# RAG-Based Technical Documentation Assistant

A Retrieval-Augmented Generation (RAG) system for answering questions from technical documentation using a self-corrective LangGraph workflow.

The system retrieves relevant document chunks, grades their relevance, rewrites the query when retrieval is insufficient, and falls back to web search after multiple unsuccessful attempts.

## Tech Stack

- Python
- LangGraph
- LangChain
- FastAPI
- FAISS
- Sentence Transformers
- Groq
- Tavily
- Streamlit

## Architecture

```text
                         User Question
                              |
                              v
                     +------------------+
                     |  Query Analysis  |
                     +--------+---------+
                              |
                              v
                     +------------------+
                     |    Retrieval     |
                     |     Top-K = 5    |
                     +--------+---------+
                              |
                              v
                     +------------------+
                     | Document Grading |
                     +--------+---------+
                              |
                    +---------+---------+
                    |                   |
               Relevant            Not Relevant
                    |                   |
                    v                   v
              Generation         Query Rewriting
                    |                   |
                    |                   v
                    |              Retrieval
                    |                   |
                    |              Document
                    |               Grading
                    |                   |
                    |            Retry up to 3 times
                    |                   |
                    |                   v
                    |              Web Search
                    |                   |
                    +---------+---------+
                              |
                              v
                           Answer
```

## Workflow

### 1. Query Analysis

The user's question is rewritten into a concise search query to improve document retrieval.

Conversation history is also considered for follow-up questions.

### 2. Retrieval

FAISS performs similarity search over the indexed document chunks and retrieves the top 5 results.

### 3. Document Grading

Each retrieved chunk is evaluated by the LLM.

- `relevant = true` → chunk is kept
- `relevant = false` → chunk is discarded

If relevant chunks are found, the workflow proceeds to generation.

### 4. Query Rewriting

If no relevant documents are found, the query is rewritten and retrieval is attempted again.

The workflow allows a maximum of 3 retries.

### 5. Web Search Fallback

If relevant documentation still cannot be found after the retry limit, Tavily web search is used as a fallback.

### 6. Generation

The final answer is generated using the retrieved context.

The model is instructed to:

- Use only the provided context
- Avoid unsupported information
- Avoid hallucinating
- Provide source information with the answer

## Project Structure

```text
RAG-Based-Technical-Documentation-Assistant/
│
├── Agent/
│   └── agent.py
│
├── app/
│   ├── main.py
│   └── schema.py
│
├── frontend/
│   └── streamlit_app.py
│
├── requirements.txt
├── .env
└── README.md
```

## Embedding Model

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The embedding model runs locally and is used to generate embeddings for document chunks and user queries.

This avoids depending on an external embedding API.

## Chunking Strategy

Documents are split using:

```text
Chunk size: 800
Chunk overlap: 100
```

The overlap helps preserve context between neighboring chunks during retrieval.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/meetgajera12/RAG-Based-Technical-Documentation-Assistant.git
cd RAG-Based-Technical-Documentation-Assistant
```

### 2. Create virtual environment

```bash
python -m venv myvenv
```

Activate it on Windows:

```bash
myvenv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```


## Run the Application

Start the FastAPI application:

```bash
uvicorn backend.main:api --reload
```

After starting the backend, wait approximately **1 minute before using the application**.

The project loads the `all-MiniLM-L6-v2` embedding model and its model weights into memory. The initial model loading takes some time depending on the system.

Once the model finishes loading, the API is ready to use.

FastAPI will be available at:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

## API Endpoints

### POST `/query`

Ask a question to the RAG system.

Example request:

```json
{
    "question": "What is bias in machine learning?",
    "url": ""
}
```

Example response:

```json
{
    "answer": "Bias refers to ...",
    "source": "document_search",
    "sources": [
        "intro-to-ml.pdf"
    ]
}
```

### POST `/ingest`

Upload a technical document for indexing.

Supported document types include:

- PDF
- TXT
- Markdown

The document is processed as:

```text
Document
   ↓
Text Extraction
   ↓
Chunking
   ↓
Embedding Generation
   ↓
FAISS Index
```

### GET `/documents`

Returns the documents currently indexed by the application.

### POST `/feedback`

Submit feedback for an answer.

Example:

```json
{
    "feedback": "positive",
    "comment": "The answer was useful."
}
```

## Document Corpus

The application supports technical documentation provided as local files or URLs.

Documents are processed into chunks and stored in the FAISS vector index.

The application also supports direct URL-based questions, where the provided webpage is loaded, chunked, embedded, and searched specifically for that question.

## Design Decisions and Tradeoffs

### Local Embeddings

`sentence-transformers/all-MiniLM-L6-v2` was selected because it provides local embeddings without requiring an external embedding API.

**Tradeoff:** The model must be loaded locally during application startup and requires memory.

### FAISS

FAISS was selected as the vector store because it is simple, lightweight, and suitable for the technical-documentation corpus used in this project.

**Tradeoff:** The current implementation keeps the vector index in application memory rather than using a persistent production database.

### LLM-Based Document Grading

Retrieved chunks are passed through an LLM relevance grader before generation.

This reduces the chance of generating an answer from unrelated retrieved chunks.

**Tradeoff:** Document grading adds additional LLM calls and therefore increases latency.

### Query Rewriting

When retrieval fails, the query is rewritten and retrieval is attempted again.

A retry limit of 3 prevents an endless retrieval loop.

### Web Search Fallback

Web search is used only after document retrieval and query rewriting fail.

This keeps the primary behavior focused on the provided technical documentation while still allowing the system to handle questions outside the indexed corpus.

## Running the Streamlit Interface

After starting the FastAPI backend, run:

```bash
streamlit run frontend/streamlit_app.py
```

After starting the frontend, wait approximately **1 minute before using the application** because the backend's local embedding model needs time to finish loading its model weights.

The Streamlit interface can be used to:

- Upload documents
- Ingest URLs
- Ask questions
- View answers and sources
- View relevant document chunks
- Submit feedback
- Continue conversations with chat history

## Example Flow

```text
User Question
      ↓
Query Analysis
      ↓
FAISS Retrieval
      ↓
Document Grading
      ↓
Relevant?
   /       \
 Yes       No
  |         |
  ↓         ↓
Generate   Rewrite Query
  |         |
  |       Retrieve Again
  |         |
  |      Up to 3 retries
  |         |
  |       Web Search
  |         |
  +----→ Generation
           |
           ↓
         Answer
```

## Repository

https://github.com/meetgajera12/RAG-Based-Technical-Documentation-Assistant


## Meet Gajera

AI-ML Student 
