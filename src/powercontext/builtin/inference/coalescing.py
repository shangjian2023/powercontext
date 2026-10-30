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

"""Cross-call embedding request coalescing with a bounded LRU vector cache."""

from __future__ import annotations

import asyncio
import itertools
from collections import OrderedDict
from typing import Any

from powercontext.builtin.inference.models import EmbeddingResult, InferenceUsage


class CoalescingEmbeddingModel:
    """Share in-flight embedding work across calls and cache successful vectors.

    Concurrent calls sharing the same texts wait on one underlying request;
    completed vectors are kept in an LRU cache bounded by ``cache_size``.
    Cancelling one caller never cancels the shared underlying work.
    """

    def __init__(
        self,
        model: Any,
        *,
        max_batch_size: int = 64,
        coalesce_ms: float = 5.0,
        cache_size: int = 1024,
    ) -> None:
        if type(max_batch_size) is not int or max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")  # noqa: TRY003
        if isinstance(coalesce_ms, bool) or not isinstance(coalesce_ms, (int, float)) or coalesce_ms < 0:
            raise ValueError("coalesce_ms must be a non-negative number")  # noqa: TRY003
        if type(cache_size) is not int or cache_size < 0:
            raise ValueError("cache_size must be a non-negative integer")  # noqa: TRY003
        self._model = model
        self._max_batch_size = max_batch_size
        self._coalesce_seconds = coalesce_ms / 1000
        self._cache_size = cache_size
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._inflight: dict[str, asyncio.Future] = {}
        self._pending: list[str] = []
        self._lock = asyncio.Lock()
        self._dispatch_scheduled = False
        self._batch_ids = itertools.count(1)

    @property
    def profile(self) -> Any:
        return self._model.profile

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        """Return vectors in the original input order, sharing and caching work."""

        if not texts:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))
        unique = list(dict.fromkeys(texts))
        resolved: dict[str, Any] = {}
        waits: list[tuple[str, asyncio.Future]] = []
        registered_pending = False
        async with self._lock:
            for text in unique:
                if text in self._cache:
                    self._cache.move_to_end(text)
                    resolved[text] = self._cache[text]
                    continue
                future = self._inflight.get(text)
                if future is None:
                    future = asyncio.get_running_loop().create_future()
                    self._inflight[text] = future
                    self._pending.append(text)
                    registered_pending = True
                waits.append((text, future))
            if registered_pending and not self._dispatch_scheduled:
                self._dispatch_scheduled = True
                if self._coalesce_seconds > 0:
                    asyncio.create_task(self._delayed_dispatch())
                else:
                    asyncio.create_task(self._drain())

        if waits:
            await asyncio.gather(*(asyncio.shield(future) for _text, future in waits))
            for text, future in waits:
                resolved[text] = future.result()

        batch_usages: dict[int, EmbeddingResult] = {}
        for _text, future in waits:
            batch_usages[getattr(future, "batch_id", 0)] = getattr(future, "batch_usage", None)
        requests = 0
        input_tokens: int | None = None
        output_tokens: int | None = None
        for usage in batch_usages.values():
            if usage is None:
                continue
            requests += usage.requests
            if usage.input_tokens is not None:
                input_tokens = (input_tokens or 0) + usage.input_tokens
            if usage.output_tokens is not None:
                output_tokens = (output_tokens or 0) + usage.output_tokens
        return EmbeddingResult(
            vectors=tuple(resolved[text] for text in texts),
            usage=InferenceUsage(requests=requests, input_tokens=input_tokens, output_tokens=output_tokens),
        )

    async def _delayed_dispatch(self) -> None:
        await asyncio.sleep(self._coalesce_seconds)
        await self._drain()

    async def _drain(self) -> None:
        """Dispatch pending texts in bounded batches until the queue is empty."""

        while True:
            async with self._lock:
                if not self._pending:
                    self._dispatch_scheduled = False
                    return
                batch_texts = self._pending[: self._max_batch_size]
                del self._pending[: self._max_batch_size]
                futures = [self._inflight[text] for text in batch_texts]
            batch_id = next(self._batch_ids)
            for future in futures:
                future.batch_id = batch_id
            try:
                result = await self._model.embed(tuple(batch_texts))
            except BaseException as error:
                await self._fail_batch(batch_texts, futures, error)
                continue
            if len(result.vectors) != len(batch_texts):
                await self._fail_batch(
                    batch_texts,
                    futures,
                    ValueError(
                        f"embedding model returned {len(result.vectors)} vectors for {len(batch_texts)} texts"
                    ),
                )
                continue
            async with self._lock:
                if self._cache_size:
                    for text, vector in zip(batch_texts, result.vectors, strict=True):
                        self._cache[text] = vector
                        self._cache.move_to_end(text)
                    while len(self._cache) > self._cache_size:
                        self._cache.popitem(last=False)
                for text in batch_texts:
                    self._inflight.pop(text, None)
            for future, vector in zip(futures, result.vectors, strict=True):
                future.batch_usage = result.usage
                if not future.done():
                    future.set_result(vector)

    async def _fail_batch(
        self,
        batch_texts: list[str],
        futures: list[asyncio.Future],
        error: BaseException,
    ) -> None:
        async with self._lock:
            for text in batch_texts:
                self._inflight.pop(text, None)
        for future in futures:
            future.batch_usage = None
            if not future.done():
                future.set_exception(error)


__all__ = ["CoalescingEmbeddingModel"]
