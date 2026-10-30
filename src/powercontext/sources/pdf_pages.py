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

"""Page-level PDF Sources with exact evidence pages and offline question search."""

from __future__ import annotations

import hashlib
import re
from io import BytesIO
from urllib.parse import quote

from pydantic import BaseModel

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

try:  # pypdf is provided by the shared self-test environment.
    from pypdf import PasswordType, PdfReader
except ImportError:  # pragma: no cover - keeps import-time behavior explicit
    PdfReader = None
    PasswordType = None

PDF_PAGE_SOURCE_NAME = "pdf-page"
_WORD_PATTERN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be been by did do does for from had has have how in is it its of on or that the this to was "
    "were what when where which who will with you your company corporation inc ltd".split()
)


class PDFDocumentInput(BaseModel):
    """One complete original PDF document captured by identity."""

    document_id: str
    content: bytes


class PDFPageInput(BaseModel):
    """One physical PDF page with its document digest and extracted text."""

    document_id: str
    document_sha256: str
    page_number: int
    text: str


class PDFPageSource(Source):
    """A CAPTURED physical PDF page keeping its extractable text."""

    source_id: str
    document_id: str
    document_sha256: str
    page_number: int
    text: str


def pdf_page_source_id(document_id: str, document_sha256: str, page_number: int) -> str:
    """Return the stable identity for one page of one exact document revision."""

    if not isinstance(document_id, str) or not document_id or not document_id.strip():
        raise ValueError("document_id must be a non-blank string")  # noqa: TRY003
    identity = f"pdf:{quote(document_id, safe='')}:{document_sha256}:{page_number}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"page identity must not exceed {MAX_SOURCE_ID_LENGTH} characters")  # noqa: TRY003
    return identity


def parse_pdf_document(value: PDFDocumentInput, /) -> tuple[PDFPageInput, ...]:
    """Parse every physical page of one complete PDF, or fail atomically."""

    if PdfReader is None:
        raise ValueError("pypdf is required to parse PDF documents")  # noqa: TRY003
    document_id = value.document_id
    if not isinstance(document_id, str) or not document_id or not document_id.strip():
        raise ValueError("document_id must be a non-blank string")  # noqa: TRY003
    digest = hashlib.sha256(value.content).hexdigest()
    try:
        reader = PdfReader(BytesIO(value.content))
        if getattr(reader, "is_encrypted", False) and reader.decrypt("") == PasswordType.NOT_DECRYPTED:
            raise ValueError("encrypted PDF requires a password")  # noqa: TRY003
        pages = tuple(reader.pages)
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"PDF document cannot be parsed: {error}") from error
    parsed: list[PDFPageInput] = []
    try:
        for index, page in enumerate(pages, start=1):
            text = page.extract_text() or ""
            parsed.append(
                PDFPageInput(
                    document_id=document_id,
                    document_sha256=digest,
                    page_number=index,
                    text=text,
                )
            )
    except Exception as error:
        raise ValueError(f"PDF page {len(parsed) + 1} cannot be extracted: {error}") from error
    return tuple(parsed)


async def search_pdf_pages(memory: object, question: str, limit: int = 10) -> tuple[object, ...]:
    """Search stored page evidence for one natural-language question.

    Issues several deterministic keyword views of the question through the
    public Memory search API and merges their hits by reciprocal-rank fusion,
    so pages relevant under any view surface within the requested limit.
    """

    from powercontext.builtin.artifacts.memory.models import MemoryCitation
    from powercontext.builtin.runtime import SearchMemoryRequest

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-blank string")  # noqa: TRY003
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be a positive integer")  # noqa: TRY003

    pool = 50
    scores: dict[tuple[str, str, str], float] = {}
    citations: dict[tuple[str, str, str], object] = {}
    order: dict[tuple[str, str, str], int] = {}
    seen = 0
    for query in _query_views(question):
        result = await memory.search(SearchMemoryRequest(query=query, mode="fts", limit=pool))
        for rank, hit in enumerate(result.hits):
            key = (str(hit.memory_ref), str(hit.entry_id), str(hit.entry_version_id))
            if key not in order:
                order[key] = seen
                seen += 1
            scores[key] = scores.get(key, 0.0) + 1.0 / (60.0 + rank)
            citations[key] = MemoryCitation(
                memory_ref=hit.memory_ref, entry_id=hit.entry_id, entry_version_id=hit.entry_version_id
            )
    ranked = sorted(scores, key=lambda key: (-scores[key], order[key]))
    return tuple(citations[key] for key in ranked[:limit])


def _query_views(question: str) -> tuple[str, ...]:
    """Deterministic keyword views of one question for full-text search."""

    lowered = question.lower()
    words = _WORD_PATTERN.findall(lowered)
    content = [word for word in words if word not in _STOPWORDS and len(word) > 1]
    distinctive = sorted(set(content), key=lambda word: (-len(word), word))
    views = [question.strip()]
    if content:
        views.append(" ".join(content))
    if len(distinctive) >= 3:
        views.append(" ".join(distinctive[:3]))
    if len(distinctive) >= 6:
        views.append(" ".join(distinctive[:6]))
    seen: set[str] = set()
    unique = []
    for view in views:
        if view and view not in seen:
            seen.add(view)
            unique.append(view)
    return tuple(unique)


class PDFPageSourceAdapter:
    """Resolve and read page-level PDF Sources."""

    input_class = PDFPageInput
    name = PDF_PAGE_SOURCE_NAME
    source_class = PDFPageSource

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        identity = pdf_page_source_id(value.document_id, value.document_sha256, value.page_number)
        return PDFPageSource(
            name=identity,
            materialization=SourceMaterialization.CAPTURED,
            source_id=identity,
            document_id=value.document_id,
            document_sha256=value.document_sha256,
            page_number=value.page_number,
            text=value.text,
        )

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return PDFPageInput(
            document_id=source.document_id,
            document_sha256=source.document_sha256,
            page_number=source.page_number,
            text=source.text,
        )


class PDFPageTextEvidenceProjection:
    """Project one PDF page as canonical text evidence."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PDFPageSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PDF_PAGE_SOURCE_NAME,
            source_id=source.source_id,
            content=f"{source.document_id}\n{source.text}",
            metadata={
                "document_id": source.document_id,
                "document_sha256": source.document_sha256,
                "page_number": source.page_number,
            },
        )


PDF_PAGE_SOURCE_DEFINITION = AdapterSourceDefinition(
    PDFPageSourceAdapter(),
    version="1",
    projections=(PDFPageTextEvidenceProjection(),),
)


__all__ = [
    "PDF_PAGE_SOURCE_DEFINITION",
    "PDFDocumentInput",
    "PDFPageInput",
    "PDFPageSource",
    "parse_pdf_document",
    "pdf_page_source_id",
    "search_pdf_pages",
]
