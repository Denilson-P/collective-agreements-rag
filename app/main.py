
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from app.routes.chat_router.route import router as chat_router
from app.routes.document_router.route import router as document_router


def configure_openapi(application: FastAPI) -> None:
    """Configure binary file schemas for Swagger UI compatibility.

    FastAPI may describe uploaded files using the OpenAPI 3.1
    contentMediaType property instead of the traditional binary
    format.

    This customization adjusts only the document upload request
    schemas so Swagger UI can render file selection controls.

    Args:
        application: FastAPI application whose OpenAPI schema
            will be customized.

    Returns:
        None: The application's OpenAPI generator is configured.
    """

    def custom_openapi() -> dict:
        """Generate the OpenAPI schema with binary upload fields.

        Returns:
            dict: OpenAPI schema containing Swagger-compatible
                file upload definitions.
        """
        if application.openapi_schema is not None:
            return application.openapi_schema

        schema = get_openapi(
            title=application.title,
            version=application.version,
            description=application.description,
            routes=application.routes,
        )

        schemas = schema.get("components", {}).get("schemas", {})

        upload_fields = {
            "Body_upload_document_documents_upload_post": "file",
            "Body_upload_documents_batch_documents_upload_batch_post": "files",
        }

        for schema_name, field_name in upload_fields.items():
            body_schema = schemas.get(schema_name)

            if body_schema is None:
                continue

            field_schema = body_schema.get(
                "properties", {}
            ).get(field_name)

            if field_schema is None:
                continue

            if field_name == "files":
                target_schema = field_schema.get("items", {})
            else:
                target_schema = field_schema

            target_schema.pop("contentMediaType", None)
            target_schema["type"] = "string"
            target_schema["format"] = "binary"

        application.openapi_schema = schema

        return schema

    application.openapi = custom_openapi


def create_app() -> FastAPI:
    """Create and configure the collective agreements RAG application.

    Initializes the FastAPI application and registers the chat and
    document routers.

    The OpenAPI schema is customized to ensure uploaded documents
    are represented as binary files in Swagger UI.

    Returns:
        FastAPI: Configured application with chat and document APIs.

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
    application.include_router(document_router)

    configure_openapi(application)

    return application


app = create_app()
