# Collective Agreements RAG

A Retrieval-Augmented Generation (RAG) API built with **FastAPI, LangChain, OpenAI, Qdrant, and MinIO** for intelligent querying of Collective Labor Agreements (CLAs).

The system processes collective agreement documents, generates vector embeddings, retrieves relevant excerpts, and produces AI-generated answers grounded in indexed content.

## Overview

Collective Agreements RAG is designed to simplify access to information contained in collective labor agreements.

The platform supports natural-language questions about contractual obligations, employee benefits, insurance requirements, and other provisions documented in collective agreements.

It combines document ingestion, vector search, and large language models to provide contextual answers supported by retrieved sources.

## Technologies

- Python 3.12
- FastAPI
- LangChain
- OpenAI API
- Qdrant
- MinIO
- Pydantic
- Pytest
- Ruff
- Docker

## Features

### Document Processing

- Upload documents through REST endpoints
- Support for TXT, PDF, DOCX, and DOC files
- Text extraction and document chunking
- OpenAI embedding generation
- Vector indexing with Qdrant
- Document storage integration with MinIO

### Batch Document Upload

- Upload up to 50 documents per request
- Maximum file size of 10 MB per document
- Independent processing of each document
- Individual success and failure results
- Batch processing summary
- Duplicate filename detection
- Validation error handling
- Continued processing when individual documents fail validation

### RAG Chat

- Natural-language document queries
- Semantic retrieval of relevant document excerpts
- AI-generated responses using retrieved context
- Supporting source information
- Configurable retrieval count

## Prerequisites

For local development:

- Python 3.12 or higher
- An OpenAI API key
- Running Qdrant and MinIO services

For containerized execution:

- Docker
- Docker Compose

## Installation

Clone the repository:

```bash
git clone https://github.com/Denilson-P/collective-agreements-rag.git
cd collective-agreements-rag
```

Create and activate a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install the project dependencies:

```bash
pip install -e .
```

## Configuration

Create a `.env` file in the project root using `.env.example` as a reference:

```bash
cp .env.example .env
```

Configure the application settings:

```env
# OpenAI
OPENAI_API_KEY=your_api_key_here
EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-5-mini

# RAG Pipeline
CHUNK_SIZE=1200
CHUNK_OVERLAP=200
RETRIEVAL_K=4

# MinIO
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=your_minio_access_key
MINIO_SECRET_KEY=your_minio_secret_key
MINIO_BUCKET=collective-agreements
MINIO_SECURE=false

# Qdrant
QDRANT_HOST=qdrant
QDRANT_PORT=6333
QDRANT_COLLECTION=collective-agreements
```

### Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API authentication key |
| `EMBEDDING_MODEL` | Model used to generate document embeddings |
| `CHAT_MODEL` | Language model used to generate answers |
| `CHUNK_SIZE` | Maximum configured chunk size |
| `CHUNK_OVERLAP` | Overlap between consecutive document chunks |
| `RETRIEVAL_K` | Default number of retrieved document chunks |
| `MINIO_ENDPOINT` | MinIO server endpoint |
| `MINIO_ACCESS_KEY` | MinIO access key |
| `MINIO_SECRET_KEY` | MinIO secret key |
| `MINIO_BUCKET` | Bucket used for document storage |
| `MINIO_SECURE` | Whether to use HTTPS for MinIO connections |
| `QDRANT_HOST` | Qdrant server hostname |
| `QDRANT_PORT` | Qdrant server port |
| `QDRANT_COLLECTION` | Collection used to store document vectors |

The default hostnames `minio` and `qdrant` are intended for Docker Compose networking.

When running the backend directly on the host machine, configure these endpoints according to your local environment.

The `.env` file contains sensitive configuration and must never be committed to version control.

## Running the API

Start the development server:

```bash
uvicorn app.main:app --reload --port 8000
```

The interactive API documentation is available at:

http://127.0.0.1:8000/docs

## API Endpoints

### Chat

`POST /chat`

Submit a natural-language question about indexed collective agreements.

Example request:

```json
{
  "question": "Does this collective labor agreement establish any obligations related to life insurance?",
  "k": 4
}
```

The API returns an AI-generated answer and supporting document excerpts.

The optional `k` parameter controls the number of document chunks retrieved during semantic search.

### Single Document Upload

`POST /documents/upload`

Upload one document using multipart form data.

Supported formats:

- `.txt`
- `.pdf`
- `.docx`
- `.doc`

Maximum file size: **10 MB**.

Example successful response:

```json
{
  "filename": "collective_agreement.pdf",
  "status": "indexed",
  "chunks_created": 42
}
```

The number of chunks depends on the document content and chunking configuration.

### Batch Document Upload

`POST /documents/upload/batch`

Upload multiple documents using multipart form data.

**Request field:** `files`

**Batch limit:** 1–50 documents

**File size limit:** 10 MB per document

The backend processes documents individually and returns a consolidated result.

Example response:

```json
{
  "total": 3,
  "indexed": 2,
  "failed": 1,
  "results": [
    {
      "filename": "agreement_01.pdf",
      "status": "indexed",
      "chunks_created": 42,
      "error": null
    },
    {
      "filename": "agreement_02.docx",
      "status": "indexed",
      "chunks_created": 28,
      "error": null
    },
    {
      "filename": "agreement_03.pdf",
      "status": "failed",
      "chunks_created": null,
      "error": "Este documento já foi enviado anteriormente e está disponível para consulta."
    }
  ]
}
```

The batch endpoint returns individual processing results, including errors for documents that could not be indexed.

Expected validation failures and duplicate filenames do not interrupt the processing of subsequent documents.

Unexpected infrastructure failures are propagated rather than silently converted into individual validation errors.

## Duplicate Document Handling

Documents are checked for existing filenames during upload.

When a document with the same filename has already been stored, the upload is rejected to prevent duplicate indexing.

For batch uploads, the API returns an informative message:

> Este documento já foi enviado anteriormente e está disponível para consulta.

The remaining documents in the batch continue processing.

The current API contract represents duplicate documents with `status: "failed"`.

## RAG Architecture

The document ingestion and retrieval pipeline follows this workflow:

```text
Document Upload
       |
       v
File Validation
       |
       v
Document Storage (MinIO)
       |
       v
Text Extraction
       |
       v
Document Chunking
       |
       v
OpenAI Embeddings
       |
       v
Qdrant Vector Database
       |
       v
Semantic Retrieval
       |
       v
OpenAI Language Model
       |
       v
Grounded Answer with Sources
```

MinIO provides object storage for uploaded documents, while Qdrant stores and searches vector representations of document content.

OpenAI embeddings enable semantic retrieval, and the language model generates answers using the retrieved context.

## Running with Docker

The backend includes a Dockerfile and can be started through the separate infrastructure repository.

The infrastructure uses Docker Compose to orchestrate:

- FastAPI backend
- Next.js frontend
- Qdrant vector database
- MinIO object storage

From the infrastructure directory:

```bash
docker compose up -d --build
```

Check running services:

```bash
docker compose ps
```

View backend logs:

```bash
docker compose logs --tail=50 backend
```

The backend is exposed locally at:

http://127.0.0.1:8000

The API documentation is available at:

http://127.0.0.1:8000/docs

Qdrant and MinIO use persistent Docker volumes to preserve indexed data and stored documents across container restarts.

## Testing

Install the development tools:

```bash
pip install pytest ruff
```

Run the complete automated test suite:

```bash
python -m pytest -q
```

Run the batch upload controller tests:

```bash
python -m pytest -q tests/test_batch_document_controller.py
```

Check code quality:

```bash
ruff check .
```

The automated tests cover document validation, upload processing, batch limits, duplicate handling, response contracts, and other backend functionality.

## Development Status

The backend currently provides document upload, batch processing, vector indexing, and RAG-based querying.

The batch upload implementation supports up to 50 documents and includes individual error handling and regression tests for duplicate filenames.

The project remains under active development.

Future improvements may include:

- Asynchronous batch processing
- Expanded document management
- Document deletion and reindexing
- Additional integration testing
- Improved error classification
- Performance optimizations for large document batches

## Additional Notes

- Embedding generation and language model requests use the OpenAI API and may incur usage costs.
- Uploaded documents are not automatically overwritten when filenames already exist.
- Batch processing currently occurs within the HTTP request.
- Large batches may require additional timeout configuration or asynchronous processing.
- Uploaded documents and persistent storage data should not be committed to version control.
- AI-generated answers should be verified against the provided document sources.

## License

Refer to the repository's license file, if available.