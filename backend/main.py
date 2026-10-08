import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from pypdf import PdfReader
from langchain_core.documents import Document
from backend.schema import QueryRequest, FeedbackRequest
from agent.agent import app,load_url,split_documents,set_vector_store


api = FastAPI(
    title="RAG Technical Documentation Assistant"
)


indexed_documents = []



def add_documents(documents):
    global indexed_documents

    chunks = split_documents(documents)

    indexed_documents.extend(chunks)

    set_vector_store(indexed_documents)

    return chunks



@api.get('/')
def home():
    return {'message': 'This is RAG Technical Documentation Assistant application'}

@api.get('/health')
def health_check():
    return {'status': 'OK'}

@api.post("/query")
def query(request: QueryRequest):

    initial_state = {
        "question": request.question,
        "rewritten_query": "",
        "documents": [],
        "url": request.url,
        "relevant_documents": [],
        "web_results": [],
        "answer": "",
        "retry_count": 0,
        "search_method": "",
        "chat_history": request.chat_history
    }

    try:
        result = app.invoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))


    sources = []

    if result["search_method"] == "web_search":

        for item in result.get("web_results", []):
            if item.get("url"):
                sources.append(item["url"])

    elif result["search_method"] == "url":

        if request.url:
            sources.append(request.url)

    else:

        for doc in result.get("relevant_documents", []):
            source = doc.metadata.get("source")

            if source and source not in sources:
                sources.append(source)

    if not result.get("answer"):
        raise HTTPException(
            status_code=500,
            detail="The workflow completed without generating an answer."
        )


    chunks = []

    for doc in result.get("relevant_documents", []):
        chunks.append({
            "source": doc.metadata.get("source", "Unknown"),
            "content": doc.page_content
        })

    return {
        "answer": result["answer"],
        "source": result.get("search_method", ""),
        "sources": sources,
        "chunks": chunks
    }


@api.post("/ingest")
async def ingest(
    file: UploadFile | None = File(default=None),
    url: str = ""
):

    if file is None and not url:
        raise HTTPException(
            status_code=400,
            detail="Provide either a file or a URL."
        )

    if file is not None and url:
        raise HTTPException(
            status_code=400,
            detail="Provide either a file or a URL, not both."
        )

    try:

        if url:

            document = load_url(url)

            chunks = add_documents([document])

            return {
                "message": "URL ingested successfully.",
                "source": url,
                "chunks": len(chunks)
            }

        content = await file.read()

        if file.filename.lower().endswith(".pdf"):

            reader = PdfReader(io.BytesIO(content))

            text = "\n".join(
                page.extract_text() or ""
                for page in reader.pages
            )

        elif file.filename.lower().endswith((".txt", ".md")):

            text = content.decode(
                "utf-8",
                errors="ignore"
            )

        else:
            raise HTTPException(
                status_code=400,
                detail="Supported file types: PDF, TXT, MD."
            )

        document = Document(
            page_content=text,
            metadata={
                "source": file.filename
            }
        )

        chunks = add_documents([document])

        return {
            "message": "File ingested successfully.",
            "source": file.filename,
            "chunks": len(chunks)
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@api.get("/documents")
def documents():

    sources = []

    for document in indexed_documents:

        source = document.metadata.get("source")

        if source and source not in sources:
            sources.append(source)

    return {
        "documents": sources
    }


@api.post("/feedback")
def feedback(request: FeedbackRequest):

    if request.feedback not in ["up", "down"]:
        raise HTTPException(
            status_code=400,
            detail="Feedback must be 'up' or 'down'."
        )

    return {
        "message": "Feedback received.",
        "feedback": request.feedback,
        "comment": request.comment
    }
