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

"""Pi coding-agent session history as per-event Sources with resumable capture."""

from __future__ import annotations

import codecs
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel, JsonValue

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.connectors import ConnectorRunCompletion, ConnectorRunStatus
from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

PI_EVENT_SOURCE_NAME = "pi-event"
PI_CONNECTOR_NAME = "pi-session-history"
PI_CONNECTOR_VERSION = "1"
_CHECKPOINT_SCHEMA = "powercontext.pi-session-history.v1"


class PiSessionInput(BaseModel):
    """One complete Pi session archive and the branch leaf to capture."""

    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    """One visible Pi event with its identity, rendered body, and raw record."""

    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


class PiEventSource(Source):
    """A CAPTURED Pi event keeping the full original record."""

    source_id: str
    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


def pi_event_source_id(session_id: str, event_id: str) -> str:
    """Return the stable identity ``pi:<session>:<event>`` for one event."""

    for name, value in (("session_id", session_id), ("event_id", event_id)):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")  # noqa: TRY003
    identity = f"pi:{quote(session_id, safe='')}:{quote(event_id, safe='')}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"event identity must not exceed {MAX_SOURCE_ID_LENGTH} characters")  # noqa: TRY003
    return identity


def parse_pi_session(value: PiSessionInput, /) -> tuple[PiEventInput, ...]:
    """Parse one archive and return the visible events of the selected branch.

    The whole archive is validated before anything is returned; any encoding,
    JSON, graph, or leaf error raises ``ValueError`` without partial results.
    """

    payload = value.content[len(codecs.BOM_UTF8):] if value.content.startswith(codecs.BOM_UTF8) else value.content
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Pi session content must be strict UTF-8") from error
    records: list[Any] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except ValueError as error:
            raise ValueError(f"Pi session line is not valid JSON: {error}") from error
    if not records:
        raise ValueError("Pi session archive is empty")  # noqa: TRY003
    header = records[0]
    if not isinstance(header, dict) or header.get("type") != "session" or header.get("version") != 3:
        raise ValueError("Pi session archive must start with a version 3 session record")  # noqa: TRY003
    session_id = header.get("id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("Pi session id must be a non-empty string")  # noqa: TRY003

    nodes: list[dict[str, JsonValue]] = []
    seen: set[str] = set()
    for record in records[1:]:
        if not isinstance(record, dict):
            raise ValueError("Pi session records must be JSON objects")  # noqa: TRY003
        node_id = record.get("id")
        if not isinstance(node_id, str) or not node_id:
            raise ValueError("Pi session records must carry a non-empty string id")  # noqa: TRY003
        if node_id in seen:
            raise ValueError(f"duplicate Pi session record id {node_id!r}")  # noqa: TRY003
        parent_id = record.get("parentId")
        if parent_id is not None and (not isinstance(parent_id, str) or parent_id not in seen):
            raise ValueError(f"Pi session record {node_id!r} references an unknown parent")  # noqa: TRY003
        seen.add(node_id)
        nodes.append(record)

    if not nodes:
        raise ValueError("Pi session archive must contain at least one event record")  # noqa: TRY003
    leaf_id = value.leaf_id if value.leaf_id is not None else nodes[-1]["id"]
    if leaf_id not in seen:
        raise ValueError(f"Pi session leaf {leaf_id!r} does not exist")  # noqa: TRY003

    by_id = {node["id"]: node for node in nodes}
    path: list[dict[str, JsonValue]] = []
    cursor: str | None = leaf_id
    while cursor is not None:
        node = by_id[cursor]
        path.append(node)
        parent = node.get("parentId")
        cursor = parent if isinstance(parent, str) else None
    path.reverse()

    events: list[PiEventInput] = []
    for node in path:
        rendered = _render_event(node)
        if rendered is None:
            continue
        role, body = rendered
        events.append(
            PiEventInput(
                session_id=session_id,
                event_id=node["id"],
                parent_id=node.get("parentId"),
                event_type=node.get("type", ""),
                role=role,
                content=f"{role}: {body}",
                record=node,
            )
        )
    return tuple(events)


def _render_event(node: dict[str, JsonValue]) -> tuple[str, str] | None:
    """Return (role, body) for one visible event, or None for invisible ones."""

    node_type = node.get("type")
    if node_type == "message":
        message = node.get("message")
        if not isinstance(message, dict):
            raise ValueError("Pi message records must carry a message object")  # noqa: TRY003
        role = message.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError("Pi message records must carry a non-empty role")  # noqa: TRY003
        if role == "bashExecution":
            command = message.get("command")
            output = message.get("output")
            if not isinstance(command, str) or not isinstance(output, str):
                raise ValueError("Pi bashExecution records must carry command and output strings")  # noqa: TRY003
            return role, f"$ {command}\n{output}"
        body = _visible_body(message)
        if role == "toolResult":
            tool_name = message.get("toolName")
            tool_call_id = message.get("toolCallId")
            if not isinstance(tool_name, str) or not isinstance(tool_call_id, str):
                raise ValueError("Pi toolResult records must carry toolName and toolCallId")  # noqa: TRY003
            body = f"[tool_result {tool_name} {tool_call_id}] {body}"
        return (role, body) if body.strip() else None
    if node_type in ("compaction", "branch_summary"):
        summary = node.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            return None
        return node_type, summary
    if node_type == "custom_message":
        body = _visible_body(node)
        return ("custom", body) if body.strip() else None
    return None


def _visible_body(container: dict[str, JsonValue]) -> str:
    """Join the visible string and tool-call parts of one content field."""

    content = container.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
        elif block_type == "toolCall":
            name = block.get("name")
            call_id = block.get("id")
            arguments = block.get("arguments")
            if not isinstance(name, str) or not isinstance(call_id, str) or not isinstance(arguments, dict):
                raise ValueError("Pi toolCall blocks must carry name, id, and object arguments")  # noqa: TRY003
            encoded = json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            parts.append(f"[tool_call {name} {call_id}] {encoded}")
    return "\n".join(parts)


class PiEventSourceAdapter:
    """Resolve and read per-event Pi Sources."""

    input_class = PiEventInput
    name = PI_EVENT_SOURCE_NAME
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        identity = pi_event_source_id(value.session_id, value.event_id)
        return PiEventSource(
            name=identity,
            materialization=SourceMaterialization.CAPTURED,
            source_id=identity,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=value.record,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


class PiEventTextEvidenceProjection:
    """Project one Pi event as canonical text evidence."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PI_EVENT_SOURCE_NAME,
            source_id=source.source_id,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


PI_EVENT_SOURCE_DEFINITION = AdapterSourceDefinition(
    PiEventSourceAdapter(),
    version="1",
    projections=(PiEventTextEvidenceProjection(),),
)


def _record_digest(record: dict[str, JsonValue]) -> str:
    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _imported_digests(checkpoint: JsonValue) -> dict[str, str]:
    if not isinstance(checkpoint, dict) or checkpoint.get("schema") != _CHECKPOINT_SCHEMA:
        return {}
    imported = checkpoint.get("imported")
    if not isinstance(imported, dict):
        return {}
    return {key: value for key, value in imported.items() if isinstance(key, str) and isinstance(value, str)}


class PiSessionConnector:
    """Capture the selected Pi branch incrementally with durable checkpoints."""

    name = PI_CONNECTOR_NAME
    version = PI_CONNECTOR_VERSION
    source_definitions = frozenset({PI_EVENT_SOURCE_NAME})

    def __init__(self, path: Path, leaf_id: str | None = None) -> None:
        self._path = Path(path)
        self._leaf_id = leaf_id

    async def run(self, session, /) -> ConnectorRunCompletion:
        events = parse_pi_session(
            PiSessionInput(content=self._path.read_bytes(), leaf_id=self._leaf_id)
        )
        imported = _imported_digests(session.checkpoint)
        current = {event.event_id: _record_digest(event.record) for event in events}
        # Detect rewrites of every already-imported event that still exists in the file.
        for event_id, digest in imported.items():
            if event_id in current and current[event_id] != digest:
                raise ValueError(f"imported Pi event {event_id!r} was rewritten in the session file")  # noqa: TRY003
        for event in events:
            if event.event_id in imported:
                continue
            await session.submit(event.event_id, PI_EVENT_SOURCE_NAME, event)
            imported[event.event_id] = current[event.event_id]
        session_id = events[-1].session_id if events else _session_id_of(session.checkpoint)
        checkpoint = {
            "schema": _CHECKPOINT_SCHEMA,
            "session_id": session_id,
            "imported": imported,
        }
        return ConnectorRunCompletion(status=ConnectorRunStatus.COMPLETE, checkpoint=checkpoint)


def _session_id_of(checkpoint: JsonValue) -> str:
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("session_id"), str):
        return checkpoint["session_id"]
    return ""


__all__ = [
    "PI_EVENT_SOURCE_DEFINITION",
    "PiEventInput",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
    "pi_event_source_id",
]
