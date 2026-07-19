#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsrag is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""Configuration."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Logging
    log_level: str = "INFO"
    log_json: bool = False

    # SQLite content index
    sqlite_index_path: Path = Path("data/index.db")

    # Chat model (Anthropic)
    chat_api_key: str | None = None
    chat_model: str = "claude-haiku-4-5-20251001"

    # Embeddings (local, via fastembed)
    embedding_model: str = "BAAI/bge-small-en"
    embedding_dim: int = 384

    # ONNX intra-op threads per local embedding inference.
    # None lets ONNX use every core for a single call
    inference_threads: int | None = None

    # LLM generation
    llm_temperature: float = 0.1
    llm_reasoning_effort: str | None = None
    llm_max_tokens: int = 4096
    llm_reasoning_max_tokens: int = 8192

    # Reranker (via Cohere hosted API)
    reranker_model: str = "rerank-v3.5"
    reranker_api_key: str | None = None

    # RAG pipeline
    retrieval_top_k: int = 20
    rerank_top_k: int = 5

    # Hybrid search (vector + FTS ensemble weights)
    vector_weight: float = 0.7
    fts_weight: float = 0.3

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = ["*"]

    # Content sources
    sitsbook_url: str = "https://e-sensing.github.io/sitsbook"
    sits_reference_url: str = "https://e-sensing.github.io/sits"

    # Data paths
    geosatdb_data_dir: Path = Path("data/geosatdb/data")
    spectral_indices_path: Path = Path("data/spectral-indices/indices.json")
    sits_collections_path: Path = Path("data/sits/collections.txt")
    articles_path: Path = Path("data/articles/articles.json")

    # Environment
    environment: str = "development"

    # SSE / connection limits
    sse_max_connections: int = 200
    sse_heartbeat_interval: float = 15.0
    sse_max_queue_size: int = 256

    # Agent iteration cap (single flat budget - all users anonymous)
    agent_max_iterations: int = 15

    # Global daily request ceiling
    daily_request_limit: int = 500  # or zero for unlimited
    mcp_url: str = ""
    daily_limit_message: str = (
        "You've reached today's usage limit for the public assistant. Please try again tomorrow."
    )

    # CORS
    cors_allow_credentials: bool = False

    # Langfuse
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_enabled: bool = True

    @property
    def index_db_url(self) -> str:
        """Get async SQLAlchemy URL for the read-only content index."""
        return f"sqlite+aiosqlite:///{self.sqlite_index_path}"
