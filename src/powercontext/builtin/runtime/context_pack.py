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

"""Context Pack wire protocol: full packs, prefix-reuse deltas, and rebuilds."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from dataclasses import dataclass

PACK_SCHEMA = "powercontext.context-pack.v1"
DELTA_SCHEMA = "powercontext.context-pack-delta.v1"

_PACK_KEYS = {"schema", "items", "digest", "byte_count"}
_DELTA_KEYS = {"schema", "base_digest", "target_digest", "prefix_items", "appended"}
_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_DEFAULT_CHUNK_BYTES = 512


class ContextPackError(ValueError):
    """Raised when a pack or delta violates the wire protocol or chunking rules."""


def _canonical_json(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _encode_items(items: tuple[bytes, ...]) -> list[str]:
    return [base64.b64encode(item).decode("ascii") for item in items]


def _items_digest(items: tuple[bytes, ...]) -> str:
    unsigned = {"schema": PACK_SCHEMA, "items": _encode_items(items)}
    return f"sha256:{hashlib.sha256(_canonical_json(unsigned)).hexdigest()}"


def _pack_byte_count(items: tuple[bytes, ...], digest: str) -> int:
    count = 0
    while True:
        payload = {"schema": PACK_SCHEMA, "items": _encode_items(items), "digest": digest, "byte_count": count}
        encoded_length = len(_canonical_json(payload))
        if encoded_length == count:
            return count
        count = encoded_length


def _require_digest(value: object, field: str) -> str:
    if not isinstance(value, str) or not _DIGEST_PATTERN.fullmatch(value):
        raise ContextPackError(f"{field} must be a sha256:<hex> digest")
    return value


def _require_items(value: object, field: str) -> tuple[bytes, ...]:
    if not isinstance(value, list):
        raise ContextPackError(f"{field} must be a list of base64 strings")
    items = []
    for entry in value:
        if not isinstance(entry, str):
            raise ContextPackError(f"{field} entries must be base64 strings")
        try:
            items.append(base64.b64decode(entry, validate=True))
        except (binascii.Error, ValueError) as error:
            raise ContextPackError(f"{field} entries must be valid base64") from error
    return tuple(items)


def _chunk_utf8(data: bytes, limit: int) -> tuple[bytes, ...]:
    if type(limit) is not int or limit < 1:
        raise ContextPackError("chunk_bytes must be a positive integer")
    chunks: list[bytes] = []
    position = 0
    total = len(data)
    while position < total:
        end = min(position + limit, total)
        if end < total:
            while end > position and (data[end] & 0xC0) == 0x80:
                end -= 1
            if end == position:
                raise ContextPackError("chunk_bytes cannot hold one complete UTF-8 sequence")
        chunks.append(data[position:end])
        position = end
    return tuple(chunks)


@dataclass(frozen=True, slots=True)
class ContextPack:
    """A complete chunked context with digest and exact wire byte count."""

    items: tuple[bytes, ...]
    digest: str
    byte_count: int

    @classmethod
    def build(cls, items: tuple[bytes, ...]) -> ContextPack:
        normalized = tuple(items) if isinstance(items, tuple) else tuple(items)
        if any(not isinstance(item, bytes) for item in normalized):
            raise ContextPackError("pack items must be bytes")
        digest = _items_digest(normalized)
        return cls(items=normalized, digest=digest, byte_count=_pack_byte_count(normalized, digest))

    @classmethod
    def from_content(cls, content: str, *, chunk_bytes: int = _DEFAULT_CHUNK_BYTES) -> ContextPack:
        return cls.build(_chunk_utf8(content.encode("utf-8"), chunk_bytes))

    @classmethod
    def from_bytes(cls, raw: bytes) -> ContextPack:
        if not isinstance(raw, (bytes, bytearray)):
            raise ContextPackError("pack wire bytes are required")
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, ValueError) as error:
            raise ContextPackError("pack wire bytes must be JSON") from error
        if not isinstance(payload, dict) or set(payload) != _PACK_KEYS:
            raise ContextPackError("pack payload must contain exactly the protocol fields")
        if payload["schema"] != PACK_SCHEMA:
            raise ContextPackError("pack payload schema mismatch")
        items = _require_items(payload["items"], "items")
        digest = _require_digest(payload["digest"], "digest")
        if digest != _items_digest(items):
            raise ContextPackError("pack digest does not match its items")
        if type(payload["byte_count"]) is not int or payload["byte_count"] != len(raw):
            raise ContextPackError("pack byte_count must equal its wire length")
        return cls(items=items, digest=digest, byte_count=payload["byte_count"])

    def to_bytes(self) -> bytes:
        payload = {
            "schema": PACK_SCHEMA,
            "items": _encode_items(self.items),
            "digest": self.digest,
            "byte_count": self.byte_count,
        }
        return _canonical_json(payload)


@dataclass(frozen=True, slots=True)
class ContextPackDelta:
    """One prefix-reuse delta transforming a base pack into the target pack."""

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack) -> ContextPackDelta:
        prefix = 0
        limit = min(len(base.items), len(target.items))
        while prefix < limit and base.items[prefix] == target.items[prefix]:
            prefix += 1
        return cls(
            base_digest=base.digest,
            target_digest=target.digest,
            prefix_items=prefix,
            appended=target.items[prefix:],
        )

    @classmethod
    def from_bytes(cls, raw: bytes) -> ContextPackDelta:
        if not isinstance(raw, (bytes, bytearray)):
            raise ContextPackError("delta wire bytes are required")
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, ValueError) as error:
            raise ContextPackError("delta wire bytes must be JSON") from error
        if not isinstance(payload, dict) or set(payload) != _DELTA_KEYS:
            raise ContextPackError("delta payload must contain exactly the protocol fields")
        if payload["schema"] != DELTA_SCHEMA:
            raise ContextPackError("delta payload schema mismatch")
        if type(payload["prefix_items"]) is not int or payload["prefix_items"] < 0:
            raise ContextPackError("prefix_items must be a non-negative integer")
        return cls(
            base_digest=_require_digest(payload["base_digest"], "base_digest"),
            target_digest=_require_digest(payload["target_digest"], "target_digest"),
            prefix_items=payload["prefix_items"],
            appended=_require_items(payload["appended"], "appended"),
        )

    def to_bytes(self) -> bytes:
        payload = {
            "schema": DELTA_SCHEMA,
            "base_digest": self.base_digest,
            "target_digest": self.target_digest,
            "prefix_items": self.prefix_items,
            "appended": _encode_items(self.appended),
        }
        return _canonical_json(payload)

    @property
    def byte_count(self) -> int:
        return len(self.to_bytes())


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta) -> ContextPack:
    """Rebuild the target pack from one base pack and a validated delta."""

    if not isinstance(base, ContextPack) or not isinstance(delta, ContextPackDelta):
        raise ContextPackError("apply requires a ContextPack and a ContextPackDelta")
    if delta.base_digest != base.digest:
        raise ContextPackError("delta base digest does not match the base pack")
    if delta.prefix_items < 0 or delta.prefix_items > len(base.items):
        raise ContextPackError("delta prefix exceeds the base pack chunk count")
    items = (*base.items[: delta.prefix_items], *delta.appended)
    if _items_digest(items) != delta.target_digest:
        raise ContextPackError("rebuilt items do not match the target digest")
    return ContextPack.build(items)


__all__ = [
    "ContextPack",
    "ContextPackDelta",
    "ContextPackError",
    "apply_context_pack_delta",
]
