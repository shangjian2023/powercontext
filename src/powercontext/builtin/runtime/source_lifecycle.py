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

"""Source lifecycle dependency graph with withdrawal and restore propagation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import insert, select, update
from sqlalchemy.exc import IntegrityError

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory.models import MemoryCitation
from powercontext.builtin.persistence.tables import SOURCE_LIFECYCLE_TABLE
from powercontext.builtin.runtime.models import GetMemoryEntryRequest
from powercontext.limits import MAX_SCOPE_ID_LENGTH
from powercontext.sources.models import SourceRef

LIFECYCLE_SCHEMA = "powercontext.source-lifecycle.v1"
_LIFECYCLE_REASON = "source-lifecycle"


class SourceLifecycleError(ValueError):
    """Raised when a lifecycle registration, event, or resolution is invalid."""


class SourceLifecycleEvent(BaseModel):
    """One durable availability change for one registered Source node."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str
    revision: int = Field(strict=True, ge=1)
    sequence: int = Field(strict=True, ge=1)
    available: bool = Field(strict=True)
    reason: str

    @field_validator("node_id", "reason")
    @classmethod
    def require_visible_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("lifecycle text fields must not be blank")  # noqa: TRY003
        return value


class _NodeRecord(BaseModel):
    """One registered node binding and its dependency declaration."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["source", "memory", "topic"]
    reference: dict[str, Any]
    parents: tuple[str, ...] = ()
    policy: Literal["all", "any"] | None = None
    revision: int | None = None


class _EventRecord(BaseModel):
    """The newest accepted event of one registered Source node."""

    model_config = ConfigDict(extra="forbid")

    sequence: int
    available: bool


class _LifecycleState(BaseModel):
    """The complete persisted lifecycle graph of one Scope."""

    model_config = ConfigDict(extra="forbid")

    protocol: str = LIFECYCLE_SCHEMA
    nodes: dict[str, _NodeRecord] = Field(default_factory=dict)
    events: dict[str, _EventRecord] = Field(default_factory=dict)
    available: set[str] = Field(default_factory=set)


async def read_unavailable_topic_artifacts(database: Any, scope_id: str, /) -> frozenset[str]:
    """Return artifact ids of registered Topic nodes that are not available."""

    state, _generation = await _load_state(database, scope_id)
    if state is None:
        return frozenset()
    return frozenset(
        str(node.reference.get("artifact_id", ""))
        for node_id, node in state.nodes.items()
        if node.kind == "topic" and node_id not in state.available
    )


def lifecycle_aware_topic_search(search: Any, database: Any, /) -> Any:
    """Wrap one Topic Memory search so unavailable topics stay out of results."""

    async def wrapped(scope_id: str, query: str, /, **kwargs: Any) -> Any:
        result = await search(scope_id, query, **kwargs)
        try:
            hidden = await read_unavailable_topic_artifacts(database, scope_id)
        except Exception:
            return result
        if not hidden:
            return result
        kept = tuple(hit for hit in result.hits if hit.artifact_ref.artifact_id not in hidden)
        return result if len(kept) == len(result.hits) else result.model_copy(update={"hits": kept})

    return wrapped


class LifecycleApplication:
    """Select the lifecycle service for one Scope."""

    def __init__(self, runtime: Any, database: Any) -> None:
        self._runtime = runtime
        self._database = database

    def for_scope(self, scope_id: str, /) -> ScopedSourceLifecycleService:
        return ScopedSourceLifecycleService(self._runtime, self._database, scope_id)


class ScopedSourceLifecycleService:
    """Register evidence dependencies and propagate Source availability."""

    def __init__(self, runtime: Any, database: Any, scope_id: str) -> None:
        if not isinstance(scope_id, str) or not scope_id or not scope_id.strip() or len(scope_id) > MAX_SCOPE_ID_LENGTH:
            raise SourceLifecycleError("scope_id must be a bounded non-blank string")
        self._runtime = runtime
        self._database = database
        self.scope_id = scope_id

    async def register_source(self, node_id: str, reference: SourceRef, /, *, revision: int = 1) -> None:
        """Bind one node id to a readable Source at its lifecycle revision."""

        await self._register(node_id, "source", reference, parents=(), policy=None, revision=revision)

    async def register_memory(
        self,
        node_id: str,
        citation: MemoryCitation,
        /,
        *,
        parents: tuple[str, ...] = (),
        policy: str = "all",
    ) -> None:
        """Bind one node id to a readable Memory citation with dependencies."""

        await self._register(node_id, "memory", citation, parents=parents, policy=policy)

    async def register_topic(
        self,
        node_id: str,
        reference: ArtifactRef,
        /,
        *,
        parents: tuple[str, ...] = (),
        policy: str = "all",
    ) -> None:
        """Bind one node id to a readable Topic Memory artifact with dependencies."""

        await self._register(node_id, "topic", reference, parents=parents, policy=policy)

    async def apply(self, event: SourceLifecycleEvent, /) -> list[str]:
        """Accept one Source event, propagate availability, and list active nodes."""

        if not isinstance(event, SourceLifecycleEvent):
            raise SourceLifecycleError("apply requires a SourceLifecycleEvent")
        state, generation = await _load_state(self._database, self.scope_id)
        state = _LifecycleState() if state is None else state
        node = state.nodes.get(event.node_id)
        if node is None or node.kind != "source":
            raise SourceLifecycleError(f"lifecycle events require a registered Source node: {event.node_id!r}")
        if node.revision != event.revision:
            raise SourceLifecycleError("lifecycle event revision must match the registered Source revision")
        replay = False
        last = state.events.get(event.node_id)
        if last is not None:
            if event.sequence < last.sequence:
                raise SourceLifecycleError("lifecycle event sequences must increase per Source")
            if event.sequence == last.sequence:
                if event.available != last.available:
                    raise SourceLifecycleError("a used lifecycle sequence cannot change its content")
                replay = True
        if not replay:
            state.events[event.node_id] = _EventRecord(sequence=event.sequence, available=event.available)
            state.available = _recompute_availability(state)
            await _store_state(self._database, self.scope_id, state, expected_generation=generation)
        else:
            state.available = _recompute_availability(state)
        await self._reconcile_memory(state)
        return sorted(state.available)

    async def resolve(self, node_id: str, /, *, purpose: str) -> Any:
        """Return the original reference for generation or audit use."""

        state, _generation = await _load_state(self._database, self.scope_id)
        if state is None or node_id not in state.nodes:
            raise SourceLifecycleError(f"unknown lifecycle node: {node_id!r}")
        if purpose not in ("generation", "audit"):
            raise SourceLifecycleError(f"unknown lifecycle purpose: {purpose!r}")
        if purpose == "generation" and node_id not in state.available:
            raise SourceLifecycleError(f"lifecycle node {node_id!r} is not available for generation")
        node = state.nodes[node_id]
        model = {"source": SourceRef, "memory": MemoryCitation, "topic": ArtifactRef}[node.kind]
        return model.model_validate(node.reference)

    async def _register(
        self,
        node_id: str,
        kind: str,
        reference: Any,
        /,
        *,
        parents: tuple[str, ...],
        policy: str | None,
        revision: int | None = None,
    ) -> None:
        _require_node_id(node_id)
        state, generation = await _load_state(self._database, self.scope_id)
        state = _LifecycleState() if state is None else state
        record = _NodeRecord(
            kind=kind,
            reference=reference.model_dump(mode="json"),
            parents=tuple(parents),
            policy=policy,
            revision=revision,
        )
        existing = state.nodes.get(node_id)
        if existing is not None:
            if existing == record:
                return
            raise SourceLifecycleError(f"lifecycle node {node_id!r} is already registered with another binding")
        if kind == "source":
            if revision is None or revision < 1:
                raise SourceLifecycleError("Source nodes require a positive lifecycle revision")
        else:
            if node_id in parents:
                raise SourceLifecycleError("lifecycle nodes cannot depend on themselves")
            if not parents:
                raise SourceLifecycleError("derived lifecycle nodes require at least one parent")
            if len(set(parents)) != len(parents):
                raise SourceLifecycleError("lifecycle parents must be unique")
            for parent in parents:
                if parent not in state.nodes:
                    raise SourceLifecycleError(f"lifecycle parent {parent!r} must be registered first")
            if policy not in ("all", "any"):
                raise SourceLifecycleError("lifecycle policy must be 'all' or 'any'")
        await self._require_readable(kind, reference)
        state.nodes[node_id] = record
        if _node_available(node_id, record, state):
            state.available.add(node_id)
        await _store_state(self._database, self.scope_id, state, expected_generation=generation)
        if kind == "memory":
            await self._reconcile_memory(state)

    async def _require_readable(self, kind: str, reference: Any, /) -> None:
        try:
            if kind == "source":
                await self._runtime._records().get_source(self.scope_id, reference.source_type, reference.source_id)
            elif kind == "memory":
                await self._runtime.memory.for_scope(self.scope_id).get(
                    GetMemoryEntryRequest(citation=reference)
                )
            else:
                await self._runtime.records.for_scope(self.scope_id).get_artifact_revision(
                    reference.family, reference.artifact_id, reference.revision
                )
        except Exception as error:
            raise SourceLifecycleError(f"registered reference must exist and be readable: {error}") from error

    async def _reconcile_memory(self, state: _LifecycleState) -> None:
        """Transition native Memory entries to the graph's target availability."""

        memory_nodes = [
            (node_id, MemoryCitation.model_validate(node.reference))
            for node_id, node in state.nodes.items()
            if node.kind == "memory"
        ]
        if not memory_nodes:
            return
        async with self._runtime._context(self.scope_id) as context:
            service = context.artifacts.memory
            artifact_id = context.artifacts.memory_artifact_id
            for node_id, citation in memory_nodes:
                head = await service.head(artifact_id)
                manifest = {item.entry_id: item for item in head.content.manifest.entries}
                item = manifest.get(citation.entry_id)
                if item is None:
                    continue
                if (item.state == "active") == (node_id in state.available):
                    continue
                version = next(
                    (entry for entry in await service.entries(head) if entry.entry_id == citation.entry_id), None
                )
                if version is None:
                    continue
                if node_id in state.available:
                    await service.reactivate(head, entries=(version,), reason=_LIFECYCLE_REASON)
                else:
                    await service.forget(head, entries=(version,), reason=_LIFECYCLE_REASON)


def _node_available(node_id: str, node: _NodeRecord, state: _LifecycleState) -> bool:
    if node.kind == "source":
        last = state.events.get(node_id)
        return True if last is None else last.available
    support = [parent in state.available for parent in node.parents]
    return all(support) if node.policy == "all" else any(support)


def _recompute_availability(state: _LifecycleState) -> set[str]:
    active: set[str] = set()
    for node_id, node in state.nodes.items():
        if node.kind == "source":
            last = state.events.get(node_id)
            if last is None or last.available:
                active.add(node_id)
            continue
        support = [parent in active for parent in node.parents]
        if all(support) if node.policy == "all" else any(support):
            active.add(node_id)
    return active


def _require_node_id(node_id: str) -> None:
    if not isinstance(node_id, str) or not node_id.strip() or len(node_id) > MAX_SCOPE_ID_LENGTH:
        raise SourceLifecycleError("lifecycle node ids must be bounded non-blank strings")


async def _load_state(database: Any, scope_id: str, /) -> tuple[_LifecycleState | None, int]:
    async with database.transaction() as connection:
        row = (
            await connection.execute(
                select(SOURCE_LIFECYCLE_TABLE.c.state, SOURCE_LIFECYCLE_TABLE.c.generation).where(
                    SOURCE_LIFECYCLE_TABLE.c.scope_id == scope_id
                )
            )
        ).mappings().one_or_none()
    if row is None:
        return None, -1
    payload = row["state"]
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8")
    return _LifecycleState.model_validate_json(payload), int(row["generation"])


async def _store_state(database: Any, scope_id: str, state: _LifecycleState, /, *, expected_generation: int) -> None:
    payload = state.model_dump_json().encode("utf-8")
    async with database.transaction() as connection:
        if expected_generation < 0:
            try:
                await connection.execute(
                    insert(SOURCE_LIFECYCLE_TABLE).values(scope_id=scope_id, state=payload, generation=0)
                )
                return
            except IntegrityError as error:
                raise SourceLifecycleError("lifecycle state was created concurrently; retry") from error
        result = await connection.execute(
            update(SOURCE_LIFECYCLE_TABLE)
            .where(SOURCE_LIFECYCLE_TABLE.c.scope_id == scope_id, SOURCE_LIFECYCLE_TABLE.c.generation == expected_generation)
            .values(state=payload, generation=expected_generation + 1)
        )
        if result.rowcount != 1:
            raise SourceLifecycleError("lifecycle state generation conflict; retry")


__all__ = [
    "LifecycleApplication",
    "SourceLifecycleError",
    "SourceLifecycleEvent",
    "lifecycle_aware_topic_search",
    "read_unavailable_topic_artifacts",
]
