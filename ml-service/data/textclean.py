"""Text cleaning for training data, mirroring internal/preprocess (Go) so the
classifiers see training text shaped like the text the live pipeline feeds
them: HTML to text, NFC, zero-width removal, whitespace normalisation.

Quote/signature stripping is not ported: the dataset texts are single
messages without reply chains (checked during profiling).
"""

from __future__ import annotations

import html
import re
import unicodedata

_HTMLISH = re.compile(r"(?i)<(html|body|div|p|br|span|table|b|i|a)(\s[^<>]*)?/?>")
_BLOCK = re.compile(r"(?i)<\s*(br|/p|p|/div|div|/li|li|/tr|tr|/h[1-6]|h[1-6]|hr)\b[^>]*>")
_DROP = re.compile(r"(?is)<(script|style|head|title|blockquote)\b.*?</\1\s*>")
_TAG = re.compile(r"<[^<>]+>")
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff\u00ad"))
_INLINE_SPACE = re.compile(r"[ \t\f\v\u00a0]+")
_MULTI_BLANK = re.compile(r"\n{3,}")


def looks_like_html(s: str) -> bool:
    return bool(_HTMLISH.search(s))


def html_to_text(s: str) -> str:
    s = _DROP.sub("", s)
    s = _BLOCK.sub("\n", s)
    s = _TAG.sub("", s)
    return html.unescape(s)


def normalize_whitespace(s: str) -> str:
    s = unicodedata.normalize("NFC", s).translate(_ZERO_WIDTH)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _INLINE_SPACE.sub(" ", s)
    s = "\n".join(line.strip() for line in s.split("\n"))
    return _MULTI_BLANK.sub("\n\n", s).strip()


def _is_missing(s) -> bool:
    return s is None or (isinstance(s, float) and s != s)  # None or NaN


def clean_body(s: str | None) -> str:
    if _is_missing(s):
        return ""
    s = str(s)
    if looks_like_html(s):
        s = html_to_text(s)
    # Literal "\n" sequences appear in some dataset bodies.
    s = s.replace("\\n", "\n")
    return normalize_whitespace(s)


def clean_subject(s: str | None) -> str:
    if _is_missing(s):
        return ""
    return " ".join(unicodedata.normalize("NFC", str(s)).translate(_ZERO_WIDTH).split())


def dedupe_key(body: str) -> str:
    """Normalised body used for duplicate detection (case/whitespace-insensitive)."""
    return " ".join(body.lower().split())
