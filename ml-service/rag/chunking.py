"""Heading-aware chunking for the KB (P5.1).

Documents (HTML help-centre articles or Markdown files) are split into
sections at headings (h1-h4 / #..####). Consecutive short sections are packed
into one chunk of at most MAX_WORDS words (~400 tokens), each keeping its
heading line; a section longer than that is windowed on its own with
OVERLAP_WORDS words (~60 tokens) of overlap. No chunk cuts a short section in
two, and overlap never crosses a heading. (Help-centre sections are short:
one chunk per section gave a median of 79 tokens on WixQA.)

Token counts are approximate (words x 4/3, typical for English with the bge
WordPiece tokenizer); the embedder truncates at 512 tokens, so a chunk is
never silently cut in half.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

MAX_WORDS = 300  # ~400 tokens
OVERLAP_WORDS = 45  # ~60 tokens
TOKENS_PER_WORD = 4 / 3

_HEADINGS = {"h1", "h2", "h3", "h4"}
_BLOCKS = {"p", "div", "li", "tr", "br", "pre", "blockquote", "h5", "h6", "td", "dd", "dt"}
_SKIP = {"script", "style", "img", "svg", "noscript"}


@dataclass
class Section:
    heading: str
    paragraphs: list[str] = field(default_factory=list)


@dataclass
class Chunk:
    ordinal: int
    heading: str
    text: str
    token_count: int


def approx_tokens(text: str) -> int:
    return math.ceil(len(text.split()) * TOKENS_PER_WORD)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


class _SectionParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.sections = [Section("")]
        self._buf: list[str] = []
        self._in_heading = False
        self._skip = 0

    def _flush(self):
        text = _clean("".join(self._buf))
        self._buf = []
        if not text:
            return
        if self._in_heading:
            self.sections.append(Section(text))
        else:
            self.sections[-1].paragraphs.append(text)

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP:
            self._skip += 1
            return
        if tag in _HEADINGS or tag in _BLOCKS:
            self._flush()
            self._in_heading = tag in _HEADINGS
        if tag == "li":
            self._buf.append("- ")

    def handle_endtag(self, tag):
        if tag in _SKIP:
            self._skip = max(0, self._skip - 1)
            return
        if tag in _HEADINGS or tag in _BLOCKS:
            self._flush()
            if tag in _HEADINGS:
                self._in_heading = False

    def handle_data(self, data):
        if not self._skip:
            self._buf.append(data)

    def close(self):
        super().close()
        self._flush()


def html_sections(html: str) -> list[Section]:
    p = _SectionParser()
    p.feed(html)
    p.close()
    return [s for s in p.sections if s.paragraphs or s.heading]


_MD_HEADING = re.compile(r"^(#{1,4})\s+(.*)$")


def markdown_sections(md: str) -> list[Section]:
    sections = [Section("")]
    para: list[str] = []

    def flush():
        if para:
            sections[-1].paragraphs.append(_clean(" ".join(para)))
            para.clear()

    for line in md.splitlines():
        m = _MD_HEADING.match(line.strip())
        if m:
            flush()
            sections.append(Section(_clean(m.group(2))))
        elif not line.strip():
            flush()
        else:
            para.append(line.strip())
    flush()
    return [s for s in sections if s.paragraphs or s.heading]


def _windows(words: list[str], max_words: int, overlap: int) -> list[list[str]]:
    if len(words) <= max_words:
        return [words]
    out, start, step = [], 0, max_words - overlap
    while start < len(words):
        out.append(words[start : start + max_words])
        if start + max_words >= len(words):
            break
        start += step
    return out


def _section_units(sec: Section, budget: int, overlap_words: int) -> list[tuple[str, str]]:
    """(heading, text) units of one section: the whole section when it fits,
    otherwise word windows (with overlap) of its paragraphs, each prefixed
    with the heading."""
    words = " ".join(sec.paragraphs).split()
    head = len(sec.heading.split())
    if not words:
        return []
    if head + len(words) <= budget:
        body = " ".join(sec.paragraphs)
        return [(sec.heading, f"{sec.heading}\n{body}" if sec.heading else body)]
    units = []
    for w in _windows(words, max(50, budget - head), overlap_words):
        body = " ".join(w)
        units.append((sec.heading, f"{sec.heading}\n{body}" if sec.heading else body))
    return units


def chunk_sections(
    sections: list[Section], max_words: int = MAX_WORDS, overlap_words: int = OVERLAP_WORDS
) -> list[Chunk]:
    """Chunks of at most ~max_words words. Consecutive short sections of one
    document are packed together (each keeps its heading line); a section
    longer than the budget is windowed on its own with overlap_words of
    overlap. Section boundaries are kept: no chunk cuts a short section."""
    if overlap_words >= max_words:
        raise ValueError("overlap_words must be smaller than max_words")
    units = [u for sec in sections for u in _section_units(sec, max_words, overlap_words)]
    chunks: list[Chunk] = []
    cur_texts: list[str] = []
    cur_heading = ""
    cur_words = 0

    def emit():
        if cur_texts:
            text = "\n\n".join(cur_texts)
            chunks.append(Chunk(len(chunks), cur_heading, text, approx_tokens(text)))

    for heading, text in units:
        n = len(text.split())
        if cur_texts and cur_words + n > max_words:
            emit()
            cur_texts, cur_words = [], 0
        if not cur_texts:
            cur_heading = heading
        cur_texts.append(text)
        cur_words += n
    emit()
    return chunks
