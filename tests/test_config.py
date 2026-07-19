#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsrag is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""Tests for Settings env-driven fields."""

import os
from pathlib import Path
from unittest.mock import patch

from sitsrag.config import Settings


def test_settings_defaults():
    """Defaults match the spec when env vars are unset."""
    with patch.dict(os.environ, {}, clear=True):
        s = Settings()

    # Assert result
    assert s.sqlite_index_path == Path("data/index.db")
    assert s.embedding_model == "BAAI/bge-small-en"
    assert s.embedding_dim == 384
    assert s.chat_model.startswith("claude-")
    assert s.agent_max_iterations == 15
    assert s.sse_max_connections == 200


def test_settings_index_db_url():
    """The content index SQLite URL is derived from the configured path."""
    with patch.dict(os.environ, {"SQLITE_INDEX_PATH": "/tmp/i.db"}, clear=True):
        s = Settings()

    assert s.index_db_url == "sqlite+aiosqlite:////tmp/i.db"


def test_settings_overrides_from_env():
    """Env vars override the defaults."""
    env = {
        "SSE_MAX_CONNECTIONS": "500",
        "AGENT_MAX_ITERATIONS": "30",
        "EMBEDDING_DIM": "768",
        "CORS_ALLOW_CREDENTIALS": "true",
    }

    with patch.dict(os.environ, env, clear=True):
        s = Settings()

    assert s.sse_max_connections == 500
    assert s.agent_max_iterations == 30
    assert s.embedding_dim == 768
    assert s.cors_allow_credentials is True
