# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Count-based automatic processing admission over persisted journal positions."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncConnection

from powercontext.builtin.persistence.cursors import SourceCursorRepository
from powercontext.builtin.persistence.processing_intents import ArtifactProcessingIntentRepository
from powercontext.builtin.persistence.tables import ARTIFACT_PROCESSING_INTENTS_TABLE, SOURCE_JOURNAL_HEADS_TABLE


async def pending_source_count(connection: AsyncConnection, scope_id: str, binding_name: str, /) -> int:
    """Return journal entries after one family cursor, by persisted identity."""

    head = await connection.scalar(
        select(SOURCE_JOURNAL_HEADS_TABLE.c.position).where(SOURCE_JOURNAL_HEADS_TABLE.c.scope_id == scope_id)
    )
    if head is None:
        return 0
    stored = await SourceCursorRepository().load(connection, scope_id, binding_name)
    sequence = 0 if stored is None else stored.cursor.sequence
    return max(0, int(head) - sequence)


async def request_count_admissions(
    connection: AsyncConnection,
    *,
    binding_name: str,
    threshold: int,
    scope_ids: tuple[str, ...] | None = None,
) -> list[str]:
    """Request one processing invocation for every scope at or above its threshold.

    Candidate scopes default to the binding's dirty intents; duplicate captures
    and receipt upgrades never advance the journal head, so they cannot admit.
    """

    if scope_ids is None:
        rows = await connection.execute(
            select(ARTIFACT_PROCESSING_INTENTS_TABLE.c.scope_id).where(
                ARTIFACT_PROCESSING_INTENTS_TABLE.c.binding_name == binding_name,
                ARTIFACT_PROCESSING_INTENTS_TABLE.c.dirty_generation > ARTIFACT_PROCESSING_INTENTS_TABLE.c.clean_generation,
            )
        )
        candidates = tuple(str(row[0]) for row in rows)
    else:
        candidates = scope_ids
    intents = ArtifactProcessingIntentRepository()
    requested: list[str] = []
    for scope_id in candidates:
        if await pending_source_count(connection, scope_id, binding_name) >= threshold:
            await intents.request(connection, scope_id, binding_name)
            requested.append(scope_id)
    return requested


__all__ = ["pending_source_count", "request_count_admissions"]
