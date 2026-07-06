"""Load the transcribed Captain Cook journal pages from data/transcriptions/.

Each file is named B<book>_P<page>.txt and starts with a "=== Buch N, Seite NNN ==="
header line, e.g. data/transcriptions/B3_P113.txt:

    === Buch 3, Seite 113 ===
    South-Sea
    June ye 24
    ...

Each page's month (if any) is parsed from that first date mention so pages can
be grouped/filtered by month or season, in addition to by book. Year is NOT
parsed — it's stated explicitly on only ~19% of pages, so month/season is the
reliable date facet this corpus supports.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .config import TRANSCRIPTIONS_ROOT

_MONTH_WORD = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:obr|ober)?|Nov(?:ember)?|Dec(?:embr|ember)?)"
)
# Anchor on "<month> [ye|y.e] <day>" so we only match real dates — e.g. this
# rejects the modal verb "may" (never followed by a day number) and "Deck"/
# "March"-adjacent words like "Marine" (broken by \b before the day digits).
_DATE_PATTERN = re.compile(rf"\b({_MONTH_WORD})\.?\s*(?:y\.?e\.?)?\s*\d{{1,2}}\b", re.IGNORECASE)
_MONTH_PREFIX_TO_NUM = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
MONTH_NAMES_DE = {
    1: "Januar", 2: "Februar", 3: "März", 4: "April", 5: "Mai", 6: "Juni",
    7: "Juli", 8: "August", 9: "September", 10: "Oktober", 11: "November", 12: "Dezember",
}
# Northern-hemisphere calendar seasons — a simplification: the voyage crosses
# both hemispheres (Antarctic ice, South Pacific islands), where "winter"
# means the opposite months. Good enough as a default; flag if this matters.
SEASON_OF_MONTH = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Frühling", 4: "Frühling", 5: "Frühling",
    6: "Sommer", 7: "Sommer", 8: "Sommer",
    9: "Herbst", 10: "Herbst", 11: "Herbst",
}
SEASON_ORDER = ["Winter", "Frühling", "Sommer", "Herbst"]

_MONTH_NAME_LOOKUP = {
    **{name.lower(): num for num, name in MONTH_NAMES_DE.items()},
    **{
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    },
}
_MONTH_NAME_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(_MONTH_NAME_LOOKUP, key=len, reverse=True)) + r")\b", re.IGNORECASE
)
_SEASON_NAME_LOOKUP = {
    "winter": "Winter",
    "frühling": "Frühling", "fruehling": "Frühling", "frühjahr": "Frühling", "fruehjahr": "Frühling",
    "spring": "Frühling",
    "sommer": "Sommer", "summer": "Sommer",
    "herbst": "Herbst", "autumn": "Herbst", "fall": "Herbst",
}
_SEASON_NAME_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(_SEASON_NAME_LOOKUP, key=len, reverse=True)) + r")\b", re.IGNORECASE
)


def parse_month(text: str) -> int | None:
    m = _DATE_PATTERN.search(text)
    return _MONTH_PREFIX_TO_NUM[m.group(1)[:3].lower()] if m else None


def parse_month_name(question: str) -> int | None:
    """Find a month name mentioned in a user's question (German or English)."""
    m = _MONTH_NAME_PATTERN.search(question)
    return _MONTH_NAME_LOOKUP[m.group(1).lower()] if m else None


def parse_season_name(question: str) -> str | None:
    """Find a season name mentioned in a user's question (German or English)."""
    m = _SEASON_NAME_PATTERN.search(question)
    return _SEASON_NAME_LOOKUP[m.group(1).lower()] if m else None


@dataclass
class Page:
    page_id: str  # e.g. "B3_P113" — used as the citation label
    book: int
    page_no: int
    text: str
    month: int | None = None  # 1-12, parsed from the first date mention, if any

    @property
    def season(self) -> str | None:
        return SEASON_OF_MONTH.get(self.month) if self.month else None


def load_pages(root: Path = TRANSCRIPTIONS_ROOT) -> list[Page]:
    pages: list[Page] = []
    for path in sorted(root.glob("*.txt")):
        stem = path.stem  # "B3_P113"
        book_part, _, page_part = stem.partition("_P")
        raw = path.read_text(encoding="utf-8").strip()
        # Drop the "=== Buch N, Seite NNN ===" header line; the body is the text.
        lines = raw.splitlines()
        body = "\n".join(lines[1:]).strip() if lines and lines[0].startswith("===") else raw
        if not body:
            continue
        pages.append(
            Page(
                page_id=stem,
                book=int(book_part.lstrip("B")),
                page_no=int(page_part),
                text=body,
                month=parse_month(body),
            )
        )
    return pages


def pages_by_book(pages: list[Page]) -> dict[int, list[Page]]:
    """Group pages by book, each book's pages sorted by page_no (reading order)."""
    by_book: dict[int, list[Page]] = {}
    for page in pages:
        by_book.setdefault(page.book, []).append(page)
    for book_pages in by_book.values():
        book_pages.sort(key=lambda p: p.page_no)
    return by_book


def pages_by_month(pages: list[Page]) -> dict[int, list[Page]]:
    """Group pages with a known month, sorted by book/page (reading order).
    Pages with no parseable date are excluded (~55% of the corpus)."""
    by_month: dict[int, list[Page]] = {}
    for page in pages:
        if page.month is not None:
            by_month.setdefault(page.month, []).append(page)
    for month_pages in by_month.values():
        month_pages.sort(key=lambda p: (p.book, p.page_no))
    return by_month


def pages_by_season(pages: list[Page]) -> dict[str, list[Page]]:
    by_season: dict[str, list[Page]] = {}
    for page in pages:
        if page.season is not None:
            by_season.setdefault(page.season, []).append(page)
    for season_pages in by_season.values():
        season_pages.sort(key=lambda p: (p.book, p.page_no))
    return by_season


_FACET_GROUPERS = {
    "book": pages_by_book,
    "month": pages_by_month,
    "season": pages_by_season,
}
_FACET_LABELS = {
    "book": lambda k: f"Buch {k}",
    "month": lambda k: MONTH_NAMES_DE[k],
    "season": lambda k: k,
}
_FACET_SORT_KEYS = {
    "book": lambda k: k,
    "month": lambda k: k,
    "season": lambda k: SEASON_ORDER.index(k),
}


def group_pages(pages: list[Page], facet: str) -> list[tuple[str, list[Page]]]:
    """Group pages by facet ("book" | "month" | "season") and return
    (label, pages) pairs in a sensible display order for that facet."""
    grouped = _FACET_GROUPERS[facet](pages)
    label_of = _FACET_LABELS[facet]
    return [(label_of(k), grouped[k]) for k in sorted(grouped, key=_FACET_SORT_KEYS[facet])]


def sample_evenly(pages: list[Page], n: int) -> list[Page]:
    """Pick n pages evenly spread across the (already-sorted) list, in order —
    covers the whole book's timeline instead of just the semantically closest
    pages, which matters for whole-book summarization (not a retrieval task)."""
    if n <= 0 or not pages:
        return []
    if len(pages) <= n:
        return list(pages)
    step = (len(pages) - 1) / (n - 1) if n > 1 else 0
    indices = sorted({round(i * step) for i in range(n)})
    return [pages[i] for i in indices]
