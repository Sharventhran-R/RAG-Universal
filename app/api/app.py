"""FastAPI application factory.

The API process serves requests and enqueues ingest jobs. It never runs
ingestion. Heavy resources (embedder, index store, llm) are created once and
stashed on ``app.state``; tests pass fakes via ``create_app(...)``.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

import app.parsers  # noqa: F401  -- populate the parser registry (for file_type + resolve)
from app.api.routes import router
from app.config import Settings, get_settings
from app.index.store import IndexStore, verify_index_dimensions
from app.paths import get_paths

log = logging.getLogger("app.api")
_STATIC_DIR = Path(__file__).parent / "static"


def create_app(
    *,
    settings: Settings | None = None,
    embedder=None,
    llm=None,
    index_store: IndexStore | None = None,
) -> FastAPI:
    s = settings or get_settings()
    paths = get_paths(s)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        from app.db.connection import init_db

        paths.ensure()
        init_db(paths.db_path)
        verify_index_dimensions(paths, s.embed_dim)

        application.state.settings = s
        application.state.paths = paths
        application.state.index_store = index_store or IndexStore.from_settings(s)
        application.state.embedder = embedder or _lazy_embedder()
        application.state.llm = llm or _lazy_llm()

        if embedder is None and not s.fake_models:  # real deployment: warn if Ollama is unreachable
            try:
                from app.llm.ollama import OllamaClient

                if not await OllamaClient(s).ping():
                    log.warning("Ollama unreachable at %s -- /query will 503", s.ollama_host)
            except Exception:  # pragma: no cover
                log.warning("Ollama ping failed", exc_info=True)
        yield

    application = FastAPI(title="Universal RAG Document Agent", lifespan=lifespan)
    application.include_router(router)

    # minimal built-in dev UI to exercise the endpoints (no build step)
    application.mount("/ui", StaticFiles(directory=str(_STATIC_DIR), html=True), name="ui")

    @application.get("/", include_in_schema=False)
    async def _root() -> RedirectResponse:
        return RedirectResponse("/ui/")

    return application


def _lazy_embedder():
    from app.embed import get_embedder

    return get_embedder()


def _lazy_llm():
    from app.llm import get_llm

    return get_llm()
