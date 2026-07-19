#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsrag is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""Tests for the global daily request ceiling and its AG-UI notice."""

import httpx

from sitsrag.services.quota import DailyQuota

AGUI_PAYLOAD = {
    "threadId": "4e2bcb7e-157c-4b4c-af3b-9ed37c496b6f",
    "runId": "r1",
    "state": {},
    "messages": [],
    "tools": [],
    "context": [],
    "forwardedProps": {},
}


def test_quota_allows_up_to_limit_then_denies():
    q = DailyQuota(2)

    # Assert result
    assert q.allow() is True
    assert q.allow() is True
    assert q.allow() is False

    # Assert count
    assert q.count == 2


def test_quota_zero_limit_is_unlimited():
    q = DailyQuota(0)

    # Assert result
    assert all(q.allow() for _ in range(50))


def test_quota_resets_on_new_day():
    q = DailyQuota(1)

    # Assert result
    assert q.allow() is True
    assert q.allow() is False

    # Simulate the UTC day rolling over.
    q._day = "1999-01-01"
    assert q.allow() is True


async def test_agui_serves_limit_notice_when_over_quota(test_app):
    """Once the ceiling is hit, the endpoint streams the notice."""
    test_app.state.daily_quota = DailyQuota(1)
    test_app.state.daily_limit_notice = "LIMIT REACHED — use the MCP server."

    # Build transport
    transport = httpx.ASGITransport(app=test_app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        first = await client.post("/agui/agent", json=AGUI_PAYLOAD)
        second = await client.post("/agui/agent", json=AGUI_PAYLOAD)

    # First request runs the agent
    assert first.status_code == 200
    assert "Hello!" in first.text

    # Second request is over quota: a valid run carrying the notice
    assert second.status_code == 200
    assert "RUN_STARTED" in second.text
    assert "RUN_FINISHED" in second.text
    assert "LIMIT REACHED" in second.text
    assert "Hello!" not in second.text
