#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsrag is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""Main application."""

from contextlib import asynccontextmanager
from datetime import datetime

from copilotkit import LangGraphAGUIAgent
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.memory import MemorySaver

from sitsrag.api.agui.router import router as agui_router
from sitsrag.api.errors import http_exception_handler
from sitsrag.api.routes.health import router as health_router
from sitsrag.api.sse.limiter import ConnectionLimiter
from sitsrag.config import Settings
from sitsrag.db.engine import create_engine, create_session_factory, init_db
from sitsrag.db.graph.retriever import create_hybrid_retriever
from sitsrag.db.graph.vector_store import create_vector_store
from sitsrag.logging import configure_logging, get_logger
from sitsrag.observability import build_trace_run, build_trace_span, configure_langfuse
from sitsrag.providers.llm import build_chat_model
from sitsrag.providers.reranker import build_reranker
from sitsrag.services.agent import SYSTEM_PROMPT, build_agent_graph
from sitsrag.services.quota import DailyQuota
from sitsrag.services.tools import (
    make_get_function_detail_tool,
    make_lookup_collection_tool,
    make_lookup_satellite_tool,
    make_lookup_spectral_index_tool,
    make_search_articles_tool,
    make_search_documentation_tool,
    make_search_reference_tool,
)

#
# Logger
#
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup and clean up on shutdown."""
    settings = app.state.settings

    # Configure logging
    configure_logging(settings)

    # Configure Langfuse
    langfuse_client = configure_langfuse(settings)

    # Build trace span (tool-level) and trace run (request-level)
    trace_span = build_trace_span(langfuse_client)
    trace_run = build_trace_run(langfuse_client)

    # Read-only content index
    index_engine = create_engine(settings.index_db_url, load_vec=True)
    index_session_factory = create_session_factory(index_engine)

    await init_db(index_engine)

    # In-memory checkpointer
    checkpointer = MemorySaver()

    # SQLite-vec vector store
    vector_store = create_vector_store(index_engine, settings)

    # Reranker
    reranker = build_reranker(settings)

    # Build agent tools
    tools = [
        make_search_documentation_tool(
            retriever=create_hybrid_retriever(
                store=vector_store,
                settings=settings,
                source="documentation",
            ),
            reranker=reranker,
            settings=settings,
            trace_span=trace_span,
        ),
        make_search_reference_tool(
            retriever=create_hybrid_retriever(
                store=vector_store,
                settings=settings,
                source="reference",
            ),
            reranker=reranker,
            settings=settings,
            trace_span=trace_span,
        ),
        make_get_function_detail_tool(
            vector_store=vector_store,
        ),
        make_search_articles_tool(
            retriever=create_hybrid_retriever(
                store=vector_store,
                settings=settings,
                source="articles",
            ),
            reranker=reranker,
            settings=settings,
            trace_span=trace_span,
        ),
        make_lookup_spectral_index_tool(
            session_factory=index_session_factory,
        ),
        make_lookup_satellite_tool(
            session_factory=index_session_factory,
        ),
        make_lookup_collection_tool(
            session_factory=index_session_factory,
        ),
    ]

    # Build LLM
    llm = build_chat_model(settings)

    # Build system prompt
    system_prompt = SYSTEM_PROMPT.format(
        current_date=datetime.now().strftime("%Y-%m-%d"),
    )

    # Build agent graph
    agent_graph = build_agent_graph(
        llm=llm,
        tools=tools,
        checkpointer=checkpointer,
        system_prompt=system_prompt,
    )

    # Services
    connection_limiter = ConnectionLimiter(max_connections=settings.sse_max_connections)

    # Global daily ceiling
    daily_quota = DailyQuota(settings.daily_request_limit)
    daily_limit_notice = settings.daily_limit_message

    # If MCP URL is set, add it to the notice
    if settings.mcp_url:
        daily_limit_notice += (
            f"\n\nFor unlimited access, use the SITS MCP server: {settings.mcp_url}"
        )

    # Build AG-UI agent
    agent = LangGraphAGUIAgent(
        name="sits_agent",
        description="SITS RAG assistant for satellite image time series analysis",
        graph=agent_graph,
    )

    # Set app state
    app.state.index_engine = index_engine
    app.state.agent = agent
    app.state.agent_graph = agent_graph
    app.state.trace_run = trace_run
    app.state.connection_limiter = connection_limiter
    app.state.daily_quota = daily_quota
    app.state.daily_limit_notice = daily_limit_notice
    app.state.agent_max_iterations = settings.agent_max_iterations
    app.state.sse_heartbeat_interval = settings.sse_heartbeat_interval
    app.state.sse_max_queue_size = settings.sse_max_queue_size

    # Log
    logger.info("SITS RAG API started")

    # Yield
    yield

    # Shutdown services
    if langfuse_client is not None:
        langfuse_client.flush()
        langfuse_client.shutdown()

    # Dispose the content-index engine (the only database)
    await index_engine.dispose()

    logger.info("SITS RAG API shutting down")


# Module-level settings
_settings = Settings()

# Build FastAPI app
app = FastAPI(
    title="SITS RAG API",
    description="RAG-powered API for the SITS R Package documentation",
    version="0.1.0",
    lifespan=lifespan,
)

# Make settings available to the lifespan handler
app.state.settings = _settings

app.add_exception_handler(HTTPException, http_exception_handler)

# CORS configuration
_cors_credentials = _settings.cors_allow_credentials and _settings.cors_origins != ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origins,
    allow_credentials=_cors_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/api", tags=["health"])
app.include_router(agui_router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=_settings.api_host, port=_settings.api_port)
