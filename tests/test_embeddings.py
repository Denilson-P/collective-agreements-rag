
from unittest.mock import Mock, patch

import pytest
from langchain_core.embeddings import Embeddings

from app.rag.embeddings import get_embeddings


def test_get_embeddings_returns_configured_model() -> None:
    """Verify that the embedding client receives the configured settings.

    Ensures that the configured model and API key are forwarded to
    OpenAIEmbeddings without making external API requests.

    Returns:
        None: Assertions validate the embedding configuration.
    """
    with (
        patch("app.rag.embeddings.settings") as mock_settings,
        patch("app.rag.embeddings.OpenAIEmbeddings") as mock_client,
    ):
        mock_settings.openai_api_key = "test-api-key"
        mock_settings.embedding_model = "text-embedding-3-small"
        mock_client.return_value = Mock(spec=Embeddings)

        result = get_embeddings()

    mock_client.assert_called_once_with(
        model="text-embedding-3-small",
        api_key="test-api-key",
    )
    assert result is mock_client.return_value


def test_get_embeddings_rejects_missing_api_key() -> None:
    """Verify that missing OpenAI credentials prevent initialization.

    Ensures that an embedding client is not created when the
    configured API key is empty.

    Returns:
        None: Assertions validate the expected configuration error.
    """
    with (
        patch("app.rag.embeddings.settings") as mock_settings,
        patch("app.rag.embeddings.OpenAIEmbeddings") as mock_client,
    ):
        mock_settings.openai_api_key = ""

        with pytest.raises(
            ValueError,
            match="OPENAI_API_KEY is required",
        ):
            get_embeddings()

    mock_client.assert_not_called()
