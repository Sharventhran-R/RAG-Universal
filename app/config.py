"""Runtime configuration. Every knob the system has, in one place.

Loaded from environment variables (case-insensitive: ``DATA_DIR``,
``EMBED_DIM``, …) and an optional ``.env`` in the working directory. Both the
API and the worker import ``get_settings()``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # --- storage -----------------------------------------------------------
    data_dir: Path = Path("./data")

    # --- embeddings ------------------------------------------------------------
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384
    embed_query_prefix: str = "Represent this sentence for searching relevant passages: "

    # --- generation ---------------------------------------------------------
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    # --- retrieval / query ---------------------------------------------------
    top_k: int = 8
    citation_enforcement: Literal["strict", "flag"] = "strict"
    insufficient_sentinel: str = "INSUFFICIENT_CONTEXT"

    # --- chunking ---------------------------------------------------------
    chunk_tokens: int = 512
    chunk_overlap: int = 128
    prefix_max_tokens: int = 64
    sheet_rows_per_block: int = 50

    # --- ingestion worker --------------------------------------------------
    ingest_poll_interval: float = 1.0
    ingest_claim_timeout: int = 300
    ingest_max_attempts: int = 3
    ingest_workers: int = 1

    # --- index cache -------------------------------------------------------
    index_cache_size: int = 8

    @field_validator("data_dir")
    @classmethod
    def _expand(cls, v: Path) -> Path:
        return v.expanduser()

    @model_validator(mode="after")
    def _coherent(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_tokens:
            raise ValueError("chunk_overlap must be < chunk_tokens")
        if self.prefix_max_tokens >= self.chunk_tokens:
            raise ValueError("prefix_max_tokens must be < chunk_tokens")
        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
