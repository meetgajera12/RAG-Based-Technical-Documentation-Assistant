import os
import requests
from typing import TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_tavily import TavilySearch
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
from langgraph.graph import StateGraph, START, END

load_dotenv()


embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


class HFEmbeddings(Embeddings):

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embedding_model.encode(
            texts,
            normalize_embeddings=True
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        return embedding_model.encode(
            text,
            normalize_embeddings=True
        ).tolist()


def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)
    return chunks


vector_store = None


def create_faiss_index(documents):

    embeddings = HFEmbeddings()

    vector_store = FAISS.from_documents(
        documents,
        embeddings
    )

    return vector_store


def set_vector_store(documents):

    global vector_store

    vector_store = create_faiss_index(documents)

    return vector_store


def load_url(url: str):

    response = requests.get(
        url,
        timeout=30,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    text = soup.get_text(
        separator="\n",
        strip=True
    )

    return Document(
        page_content=text,
        metadata={
            "source": url
        }
    )


model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_retries=5,
    reasoning_effort="low"
)

web_search_tool = TavilySearch(
    max_results=5
)


#-------Agent State--------#
class State(TypedDict):
    question: str
    rewritten_query: str
    documents: list
    url: str
    relevant_documents: list
    web_results: list
    answer: str
    retry_count: int
    search_method: str
    chat_history: list


#----------------Agent workflow building-----------------#

def route_input(state: State):

    if state.get("url"):
        return "url_context"

    return "query_analysis"


def url_context(state: State):

    document = load_url(state["url"])

    chunks = split_documents([document])

    return {
        "documents": chunks,
        "search_method": "url"
    }


def url_retrieval(state: State):

    documents = state["documents"]

    url_store = FAISS.from_documents(
        documents,
        HFEmbeddings()
    )

    results = url_store.similarity_search(
        state["question"],
        k=5
    )

    return {
    "relevant_documents": results,
    "search_method": "url"
}


def query_analysis(state: State):

    history = state.get("chat_history", [])

    response = model.invoke(
        [
            SystemMessage(
                content="""
You are a Query Analysis Agent.

Rewrite the user's question into a short and clear search query
for technical documentation.

Use the conversation history when the question is a follow-up question.

Keep the important technical terms.

Return only the search query.
Do not answer the question.
"""
            ),
            HumanMessage(
                content=f"""
Conversation history:
{history}

Current question:
{state["question"]}
"""
            )
        ]
    )

    return {
        "rewritten_query": response.content.strip()
    }



def retrieval(state: State):

    if vector_store is None:
        raise ValueError(
            "Vector store is not initialized. "
            "Please ingest documents first."
        )

    results = vector_store.similarity_search(state["rewritten_query"],k=5)

    return {"documents": results,"search_method": "document_search"}



class DocGrading(BaseModel):
    relevant: bool

def document_grading(state: State):

    query = state["rewritten_query"]
    docs = state["documents"]

    relevant_documents = []

    for doc in docs:

        response = model.with_structured_output(
            DocGrading,
            method="json_schema"
        ).invoke(
            [
                SystemMessage(
                    content="""
You are a document relevance grader.

Decide whether this document chunk is relevant to the
user's question.

Mark (relevant = true) if:
- The chunk directly answers the question.
- The chunk explains the concept asked about.
- The chunk contains useful information needed to answer
  the question.

Mark (relevant = false) only if the chunk is clearly unrelated
to the question.

Do not answer the question.
"""
                ),
                HumanMessage(
                    content=f"""
Question:
{query}

Document:
{doc.page_content}
"""
                )
            ]
        )

        if response.relevant:
            relevant_documents.append(doc)

    return {"relevant_documents": relevant_documents}



class QueryRewrite(BaseModel):
    rewritten_query: str


def rewrite_query(state: State):

    response = model.with_structured_output(QueryRewrite, method="json_schema").invoke(
        [
            SystemMessage(
                content="""
You are a query rewriting agent.

The previous search did not retrieve relevant
technical documentation.

Rewrite the query to improve retrieval.

Keep the meaning of the original question,
but use clearer and more specific technical
terms.

Do not answer the question.
"""
            ),
            HumanMessage(
                content=f"""
Original question:
{state["question"]}

Previous search query:
{state["rewritten_query"]}
"""
            )
        ]
    )

    return {"rewritten_query": response.rewritten_query, "retry_count": state["retry_count"] + 1}


def web_search(state: State):

    response = web_search_tool.invoke({
        "query": state["question"]
    })

    if isinstance(response, dict):
        results = response.get("results", [])
    else:
        results = response

    return {
        "web_results": results,
        "search_method": "web_search"
    }


def check_relevance(state: State):

    if state["relevant_documents"]:
        return "generation"

    if state["retry_count"] < 3:
        return "rewrite_query"

    return "web_search"


class AnsGen(BaseModel):
    ans: str


def ans_gen(state: State):

    context_parts = []

    for doc in state["relevant_documents"]:
        context_parts.append(
            f"Source: {doc.metadata.get('source', 'Unknown')}\n"
            f"{doc.page_content}"
        )

    for result in state.get("web_results", []):
        if isinstance(result, dict):
            context_parts.append(
                f"Web Source: {result.get('title', 'Unknown')}\n"
                f"URL: {result.get('url', '')}\n"
                f"{result.get('content', '')}"
            )

    context = "\n\n".join(context_parts)

    response = model.with_structured_output(AnsGen, method="json_schema").invoke(
        [
            SystemMessage(
                content=f"""
You are a Question-Answering Agent.

Answer the user's question using ONLY the provided context.

The context may contain:
1. Technical documentation retrieved from the local
   document collection.
2. Information retrieved from a user-provided URL.
3. Information retrieved from web search.

Rules:
- Do not use outside knowledge.
- Do not make up information.
- Give a clear and concise answer.
- Use the provided source information when citing
  the answer.

Document Context:
{context}
"""
            ),
            HumanMessage(content=f"Question: {state['question']}")
        ]
    )

    return {
        "answer": response.ans
    }


# -------Graph------
graph = StateGraph(State)

graph.add_node("query_analysis", query_analysis)
graph.add_node("retrieval", retrieval)
graph.add_node("url_context", url_context)
graph.add_node("url_retrieval", url_retrieval)
graph.add_node("document_grading", document_grading)
graph.add_node("rewrite_query", rewrite_query)
graph.add_node("web_search", web_search)
graph.add_node("generation", ans_gen)

graph.add_conditional_edges(START,route_input,
    {
        "query_analysis": "query_analysis",
        "url_context": "url_context"
    })
graph.add_edge("query_analysis", "retrieval")
graph.add_edge("retrieval", "document_grading")
graph.add_conditional_edges("document_grading",check_relevance,
    {
        "generation": "generation",
        "rewrite_query": "rewrite_query",
        "web_search": "web_search"
    })
graph.add_edge("rewrite_query", "retrieval")
graph.add_edge("url_context", "url_retrieval")
graph.add_edge("url_retrieval", "generation")
graph.add_edge("web_search", "generation")
graph.add_edge("generation", END)

app = graph.compile()
