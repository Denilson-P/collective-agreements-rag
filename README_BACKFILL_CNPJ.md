# Backfill de metadados CNPJ

Arquivos novos:
- `scripts/backfill_cnpj_metadata.py`
- `tests/test_backfill_cnpj_metadata.py`

Execução a partir da raiz do repositório backend com o ambiente virtual ativo:

```bash
python -m pytest tests/test_backfill_cnpj_metadata.py -v
python -m pytest tests/ -q

# Se estiver executando pelo WSL com Qdrant e MinIO publicados em localhost:
QDRANT_HOST=localhost MINIO_ENDPOINT=localhost:9000 python -m scripts.backfill_cnpj_metadata

# SOMENTE após verificar o resumo, fazer backup do Qdrant e revisar os avisos:
QDRANT_HOST=localhost MINIO_ENDPOINT=localhost:9000 python -m scripts.backfill_cnpj_metadata --apply
```

**Padrão é dry-run.** O script não recria a coleção, os pontos nem os embeddings. Usa a operação Qdrant `set_payload(key="metadata", payload={"cnpjs": [...]})` para manter os demais campos do metadata. Cada grupo de chunks do documento é atualizado junto.

O script procura o original em `settings.documents_dir` e, se não achar, tenta o bucket MinIO caso as credenciais estejam configuradas. Se ambos estiverem indisponíveis, os pontos correspondentes são **ignorados**, não marcados falsamente como sem CNPJ. Se o original foi lido e não possui CNPJs válidos, recebe `cnpjs: []`.

**Atenção:** a cópia de `app/rag/ingestion.py` presente no ZIP-base fornecido não mostra a extração de CNPJ da funcionalidade anterior. Este backfill foi feito **sem substituir arquivos existentes**. Antes de abrir o PR, confirme que a branch atual contém o PR anterior de ingestão de CNPJs.

O verificador matemático não confirma que um CNPJ pertença a uma entidade na Receita Federal. O processamento não distingue signatários de outras organizações mencionadas no documento.
