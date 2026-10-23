# Collective Agreements RAG

An API developed with **FastAPI, LangChain, OpenAI, and FAISS** for intelligent querying of Collective Labor Agreements (CLAs).

The project uses **Retrieval-Augmented Generation (RAG)** to identify insurance-related obligations and benefits, providing answers grounded in excerpts from the retrieved documents.

## Technologies

- Python 3.12
- FastAPI
- LangChain
- OpenAI API
- FAISS
- Pydantic
- Pytest
- Ruff

## Prerequisites

- Python 3.12 or higher
- An OpenAI API key with access to the configured models
- Collective Labor Agreement documents for indexing

## Installation

Clone the repository:

```bash
git clone https://github.com/Denilson-P/collective-agreements-rag.git
cd collective-agreements-rag
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -e ".[dev]"
```

## Configuration

Create a `.env` file in the project root directory using `.env.example` as a reference:

```bash
cp .env.example .env
```

Configure your OpenAI API key:

```env
OPENAI_API_KEY=your_api_key_here
EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-5-mini
```

The `.env` file is excluded from version control through `.gitignore` and should never be committed to the repository.

## Document Indexing

Place your Collective Labor Agreement documents in the following directory:

```text
data/documents/
```

The current implementation supports `.txt` files and HTML-based `.doc` files.

To generate embeddings and create the FAISS vector index, run the following command from the project root directory:

```bash
python - <<'PY'
from app.rag.ingestion import load_documents, split_documents
from app.rag.embeddings import create_vector_store, save_vector_store

documents = load_documents()
chunks = split_documents(documents)

vector_store = create_vector_store(chunks)
save_vector_store(vector_store)

print(f"Documents processed: {len(documents)}")
print(f"Chunks indexed: {len(chunks)}")
print("FAISS index created successfully.")
PY
```

The generated index will be stored in `data/vector_store/`.

**Note:** Embedding generation uses the OpenAI API and may incur usage costs. Once generated, the saved index can be reused for subsequent queries.

## Running the API

Start the development server:

```bash
uvicorn app.main:app --reload --port 8000
```

The interactive API documentation will be available at:

http://127.0.0.1:8000/docs

### Chat Endpoint

`POST /chat`

Example request:

```json
{
  "question": "Does this collective labor agreement establish any obligations related to life insurance?",
  "k": 4
}
```

The API returns the original question, the generated answer, and the retrieved document excerpts as supporting sources.

The optional `k` parameter specifies the number of document chunks retrieved during semantic search.

## Testing

Run the automated tests:

```bash
python -m pytest -q
```

Check code quality:

```bash
ruff check .
```

## How It Works

The RAG pipeline follows this workflow:

**Documents → Text Extraction → Chunking → OpenAI Embeddings → FAISS → Semantic Search → OpenAI → Grounded Answer with Sources**

Embeddings are generated using `text-embedding-3-small`, while responses are produced using `gpt-5-mini`, based on the relevant document excerpts retrieved from the vector index.

## Additional Notes

- The FAISS index must be generated before submitting queries.
- Local documents and generated vector indexes are excluded from version control.
- Traditional binary `.doc` files are not supported in the current implementation.
- Responses are generated based on indexed documents and should be verified against the provided source excerpts.

## License

Refer to the repository's license file, if available.
