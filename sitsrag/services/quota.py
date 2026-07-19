#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsrag is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""Global daily request ceiling."""

from __future__ import annotations

from datetime import UTC, datetime


class DailyQuota:
    """Count allowed requests per UTC day against a fixed daily limit."""

    def __init__(self, daily_limit: int) -> None:
        """Initializer.

        Args:
            daily_limit (int): Max allowed requests per UTC day.

        Notes:
            - `daily_limit` equal or less than zero means unlimited
        """
        self._limit = daily_limit
        self._day = self._today()
        self._count = 0

    @staticmethod
    def _today() -> str:
        """Get today's date in UTC."""
        return datetime.now(UTC).strftime("%Y-%m-%d")

    def _roll_day(self) -> None:
        """Roll to a new day."""
        today = self._today()

        if today != self._day:
            self._day = today
            self._count = 0

    def allow(self) -> bool:
        """Check if a request is allowed."""
        self._roll_day()

        # If unlimited, allow
        if self._limit <= 0:
            return True

        # If over the limit, deny
        if self._count >= self._limit:
            return False

        # Otherwise, allow
        self._count += 1
        return True

    @property
    def count(self) -> int:
        """Get number of requests consumed today."""
        self._roll_day()

        return self._count

    @property
    def limit(self) -> int:
        """Get configured daily limit."""
        return self._limit
