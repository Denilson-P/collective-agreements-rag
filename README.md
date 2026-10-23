# Collective Agreements RAG

API desenvolvida com **FastAPI, LangChain, OpenAI e FAISS** para consultas inteligentes sobre Convenções Coletivas de Trabalho (CCTs).

O projeto utiliza **Retrieval-Augmented Generation (RAG)** para identificar obrigações e benefícios relacionados a seguros, retornando respostas fundamentadas em trechos dos documentos consultados.

## Tecnologias

- Python 3.12
- FastAPI
- LangChain
- OpenAI API
- FAISS
- Pydantic
- Pytest
- Ruff

## Pré-requisitos

- Python 3.12 ou superior
- Chave de API da OpenAI com acesso aos modelos configurados
- Documentos de Convenções Coletivas para indexação

## Instalação

Clone o repositório:

```bash
git clone https://github.com/Denilson-P/collective-agreements-rag.git
cd collective-agreements-rag
```

Crie e ative o ambiente virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instale as dependências:

```bash
pip install -e ".[dev]"
```

## Configuração

Crie o arquivo `.env` na raiz do projeto, utilizando o `.env.example` como referência:

```bash
cp .env.example .env
```

Configure sua chave da OpenAI:

```env
OPENAI_API_KEY=sua_chave_aqui
EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-5-mini
```

O arquivo `.env` está protegido pelo `.gitignore` e não deve ser enviado ao repositório.

## Indexação dos documentos

Adicione os documentos de Convenções Coletivas ao diretório:

```text
data/documents/
```

A implementação atual suporta arquivos `.txt` e arquivos `.doc` cujo conteúdo esteja em formato HTML.

Para gerar os embeddings e criar o índice FAISS, execute na raiz do projeto:

```bash
python - <<'PY'
from app.rag.ingestion import load_documents, split_documents
from app.rag.embeddings import create_vector_store, save_vector_store

documents = load_documents()
chunks = split_documents(documents)

vector_store = create_vector_store(chunks)
save_vector_store(vector_store)

print(f"Documentos processados: {len(documents)}")
print(f"Chunks indexados: {len(chunks)}")
print("Índice FAISS criado com sucesso.")
PY
```

O índice será armazenado em `data/vector_store/`.

**Observação:** a geração de embeddings utiliza a API da OpenAI e pode gerar custos. Após a indexação, o índice salvo poderá ser reutilizado nas consultas.

## Executando a API

Inicie o servidor:

```bash
uvicorn app.main:app --reload --port 8000
```

A documentação interativa estará disponível em:

http://127.0.0.1:8000/docs

### Endpoint de consulta

`POST /chat`

Exemplo de requisição:

```json
{
  "question": "Esta convenção coletiva estabelece alguma obrigação relacionada a seguro de vida?",
  "k": 4
}
```

A API retorna a pergunta, a resposta gerada e os trechos dos documentos recuperados como fontes.

O parâmetro `k` é opcional e define a quantidade de trechos recuperados durante a busca semântica.

## Testes

Para executar os testes automatizados:

```bash
python -m pytest -q
```

Para verificar a qualidade do código:

```bash
ruff check .
```

## Funcionamento

O processamento segue o fluxo:

**Documentos → Extração de texto → Chunking → OpenAI Embeddings → FAISS → Busca semântica → OpenAI → Resposta com fontes**

Os embeddings são gerados com `text-embedding-3-small`, enquanto as respostas são produzidas com `gpt-5-mini`, utilizando os trechos recuperados do índice vetorial.

## Observações

- O índice FAISS deve ser gerado antes de realizar consultas.
- Os documentos e índices locais não são versionados.
- A leitura de arquivos `.doc` binários tradicionais não está implementada nesta versão.
- As respostas são geradas a partir dos documentos indexados e devem ser verificadas nas fontes apresentadas.

## Licença

Consulte o arquivo de licença do repositório, caso disponível.