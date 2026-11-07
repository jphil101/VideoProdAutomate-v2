"""Script text helpers: sentence splitting, caption tokens, normalisation."""
from __future__ import annotations

import re
import unicodedata

_ABBREV = {"e.g.", "i.e.", "approx.", "no.", "dr.", "mr.", "mrs.", "st.", "vs.", "etc.", "fig."}


def clean_script(text: str) -> str:
    """Strip markdown, stage directions and speaker labels; keep only spoken words."""
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("<!--"):
            continue
        line = re.sub(r"\[[^\]]*\]|\([^)]*(cut to|music|sfx|scene)[^)]*\)", "", line, flags=re.I)
        line = re.sub(r"^\s*(narrator|voice ?over|vo)\s*:\s*", "", line, flags=re.I)
        line = re.sub(r"^[-*>\d.)\s]+(?=[A-Za-z\"'])", "", line)  # bullets / numbering
        line = line.replace("**", "").replace("__", "").replace("`", "")
        lines.append(line.strip())
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def split_sentences(text: str) -> list[str]:
    text = clean_script(text)
    out, buf = [], []
    for tok in text.split(" "):
        buf.append(tok)
        low = tok.lower()
        if re.search(r"[.!?]['\")\]]*$", tok) and low not in _ABBREV and not re.fullmatch(r"[A-Z]\.", tok):
            out.append(" ".join(buf))
            buf = []
    if buf:
        out.append(" ".join(buf))
    return [s for s in (x.strip() for x in out) if s]


def words(sentence: str) -> list[str]:
    """Display tokens (keep punctuation attached, as they appear in captions)."""
    return [w for w in sentence.split() if w]


def norm(word: str) -> str:
    w = unicodedata.normalize("NFKD", word).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", w)


def syllables(word: str) -> int:
    w = norm(word)
    if not w:
        return 1
    if w.isdigit():
        return max(1, len(w))  # digits are spoken as several syllables
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups)
    if w.endswith("e") and n > 1 and not w.endswith(("le", "ee")):
        n -= 1
    return max(1, n)
