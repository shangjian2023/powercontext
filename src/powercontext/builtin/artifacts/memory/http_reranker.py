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

"""Cross-encoder HTTP scoring adapter for Memory reranking."""

from __future__ import annotations

import math
import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import (
    MemoryRerankDecision,
    _validate_bounds,
)
from powercontext.builtin.inference import InferenceUsage

_DEFAULT_POLICY_ID = "http-cross-encoder-v1"


class HTTPMemoryReranker:
    """Score Memory candidates through one batched HTTP service.

    Candidates are sent in batches of at most ``batch_size`` global ranks while
    the final selection is made once across every batch by score descending,
    ties broken by the original global rank ascending.
    """

    policy_id: str

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = _DEFAULT_POLICY_ID,
    ) -> None:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")  # noqa: TRY003
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size
        self.policy_id = policy_id

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        """Score every candidate once and select the globally best ranks."""

        _validate_bounds(len(candidates), limit)
        scores: dict[int, float] = {}
        request_count = 0
        for start in range(0, len(candidates), self._batch_size):
            batch = candidates[start : start + self._batch_size]
            payload = {
                "query": query,
                "candidates": [
                    {"rank": start + offset + 1, "text": candidate.text}
                    for offset, candidate in enumerate(batch)
                ],
            }
            response = await self._client.post(self._endpoint, json=payload)
            response.raise_for_status()
            expected_ranks = tuple(item["rank"] for item in payload["candidates"])
            scores.update(_parse_scores_response(response, expected_ranks=expected_ranks))
            request_count += 1
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return MemoryRerankDecision(
            selected_ranks=tuple(rank for rank, _score in ordered[:limit]),
            usage=InferenceUsage(requests=request_count),
        )


def _parse_scores_response(response: httpx.Response, *, expected_ranks: tuple[int, ...]) -> dict[int, float]:
    try:
        body = response.json()
    except ValueError as error:
        raise ValueError("reranker response is not valid JSON") from error
    if not isinstance(body, dict) or set(body) != {"scores"}:
        raise ValueError("reranker response must contain exactly the 'scores' field")  # noqa: TRY003
    entries = body["scores"]
    if not isinstance(entries, list) or len(entries) != len(expected_ranks):
        raise ValueError("reranker scores must cover exactly the requested batch ranks")  # noqa: TRY003
    expected = set(expected_ranks)
    scores: dict[int, float] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"rank", "score"}:
            raise ValueError("each reranker score must contain exactly 'rank' and 'score'")  # noqa: TRY003
        rank = entry["rank"]
        score = entry["score"]
        if type(rank) is not int or rank not in expected or rank in scores:
            raise ValueError("reranker score ranks must match the requested batch ranks exactly once")  # noqa: TRY003
        if type(score) not in (int, float) or not math.isfinite(score):
            raise ValueError("reranker scores must be finite numbers")  # noqa: TRY003
        scores[rank] = float(score)
    return scores


__all__ = ["HTTPMemoryReranker"]
