"""HTTP API.  ``uvicorn app.api:app``"""

from app.api.app import create_app

app = create_app()

__all__ = ["app", "create_app"]
