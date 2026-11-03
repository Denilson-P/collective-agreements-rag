import pytest
from pydantic import ValidationError

from app.routes.document_router.dto import (
    BatchDocumentResultDTO,
    BatchDocumentUploadResponseDTO,
)


def test_batch_result_indexed() -> None:
    """Verify a successful document result is accepted."""
    result = BatchDocumentResultDTO(
        filename="convencao.pdf",
        status="indexed",
        chunks_created=3,
    )

    assert result.status == "indexed"
    assert result.chunks_created == 3
    assert result.error is None


def test_batch_result_failed() -> None:
    """Verify a failed document result includes an error."""
    result = BatchDocumentResultDTO(
        filename="invalido.exe",
        status="failed",
        error="Unsupported document format.",
    )

    assert result.status == "failed"
    assert result.chunks_created is None
    assert result.error == "Unsupported document format."


@pytest.mark.parametrize(
    "data",
    [
        {
            "filename": "arquivo.pdf",
            "status": "indexed",
        },
        {
            "filename": "arquivo.pdf",
            "status": "indexed",
            "chunks_created": 1,
            "error": "Unexpected error",
        },
        {
            "filename": "arquivo.pdf",
            "status": "failed",
        },
        {
            "filename": "arquivo.pdf",
            "status": "failed",
            "chunks_created": 1,
            "error": "Processing failed",
        },
    ],
)
def test_batch_result_rejects_inconsistent_fields(
    data: dict[str, str | int],
) -> None:
    """Reject inconsistent success and failure fields."""
    with pytest.raises(ValidationError):
        BatchDocumentResultDTO.model_validate(data)


def test_batch_response_with_partial_success() -> None:
    """Verify batch counts match successful and failed documents."""
    response = BatchDocumentUploadResponseDTO(
        total=2,
        indexed=1,
        failed=1,
        results=[
            BatchDocumentResultDTO(
                filename="valido.txt",
                status="indexed",
                chunks_created=2,
            ),
            BatchDocumentResultDTO(
                filename="invalido.exe",
                status="failed",
                error="Unsupported document format.",
            ),
        ],
    )

    assert response.total == 2
    assert response.indexed == 1
    assert response.failed == 1
    assert len(response.results) == 2


def test_batch_response_rejects_incorrect_counts() -> None:
    """Reject summary counts inconsistent with document results."""
    with pytest.raises(ValidationError):
        BatchDocumentUploadResponseDTO(
            total=2,
            indexed=2,
            failed=0,
            results=[
                BatchDocumentResultDTO(
                    filename="valido.txt",
                    status="indexed",
                    chunks_created=1,
                ),
                BatchDocumentResultDTO(
                    filename="invalido.exe",
                    status="failed",
                    error="Unsupported document format.",
                ),
            ],
        )


def test_batch_response_rejects_more_than_ten_documents() -> None:
    """Reject a batch response containing more than ten documents."""
    results = [
        BatchDocumentResultDTO(
            filename=f"documento_{index}.txt",
            status="indexed",
            chunks_created=1,
        )
        for index in range(11)
    ]

    with pytest.raises(ValidationError):
        BatchDocumentUploadResponseDTO(
            total=11,
            indexed=11,
            failed=0,
            results=results,
        )
