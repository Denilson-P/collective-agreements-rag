from fastapi import FastAPI

from app.routes.chat_router.route import router as chat_router


def create_app() -> FastAPI:
    """Create and configure the collective agreements RAG application.

    Initializes the FastAPI application and registers the chat router,
    exposing the endpoints used to query indexed collective agreements.

    Returns:
        FastAPI: Configured application with the chat API registered.

    Notes:
        The vector index is loaded when the chat service is first
        requested, rather than during application initialization.
    """
    application = FastAPI(
        title="Collective Agreements RAG API",
        description=(
            "Retrieval-Augmented Generation API for querying Brazilian "
            "collective labor agreements with supporting document sources."
        ),
        version="0.1.0",
    )

    application.include_router(chat_router)

    return application


app = create_app()
