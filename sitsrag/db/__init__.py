#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsrag is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""Database module."""

from sitsrag.db.base import Base
from sitsrag.db.ontology import models as ontology_models

__all__ = (
    "Base",
    "ontology_models",
)
