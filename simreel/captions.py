"""Captions: word-synced karaoke ASS (burned in), SRT/VTT (soft track + sidecars),
and lower-third titles.  Sized and placed for the chosen resolution/orientation."""
from __future__ import annotations

from dataclasses import dataclass

from .config import CaptionSettings


@dataclass
class Card:
    words: list[tuple[str, float, float]]
    lines: list[list[int]]  # word indices per line
    start: float
    end: float

    @property
    def text(self) -> str:
        return "\n".join(" ".join(self.words[i][0] for i in ln) for ln in self.lines)


def ass_color(hex_color: str, alpha: int = 0) -> str:
    h = hex_color.lstrip("#")
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H{alpha:02X}{b}{g}{r}".upper()


def fmt_ass(t: float) -> str:
    t = max(0.0, t)
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def fmt_srt(t: float, sep=",") -> str:
    ms = int(round(max(0.0, t) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _pack(lengths: list[int], max_chars: int, max_lines: int) -> list[list[int]] | None:
    """Split word lengths into <= max_lines lines no longer than max_chars, as evenly as
    possible (minimise the longest line).  Returns index lists or None if impossible."""
    n = len(lengths)

    def width(a, b):
        return sum(lengths[a:b]) + (b - a - 1)

    if max_lines == 1 or n == 1:
        return [list(range(n))] if width(0, n) <= max_chars else None
    best, best_w = None, None
    if width(0, n) <= max_chars and max_lines >= 1:
        best, best_w = [list(range(n))], width(0, n) * 1.25  # one line only if clearly better
    for cut in range(1, n):
        head = width(0, cut)
        if head > max_chars:
            break
        rest = _pack(lengths[cut:], max_chars, max_lines - 1)
        if rest is None:
            continue
        w = max(head, max(width(cut + ln[0], cut + ln[-1] + 1) for ln in rest))
        if best_w is None or w < best_w:
            best, best_w = [list(range(cut))] + [[cut + i for i in ln] for ln in rest], w
    return best


def build_cards(sentences_words: list[list[tuple[str, float, float]]], max_chars: int, max_lines: int,
                min_dur: float = 0.7) -> list[Card]:
    """Group each sentence's words into caption cards: as full as the frame allows, broken at
    punctuation where natural, no orphan words, lines balanced, never across sentences."""
    cards: list[Card] = []
    for words in sentences_words:
        lens = [len(w[0]) for w in words]
        n = len(words)
        i = 0
        while i < n:
            k = 1
            while i + k < n and _pack(lens[i:i + k + 1], max_chars, max_lines) is not None:
                k += 1
            # break after punctuation if it keeps the card reasonably full
            if i + k < n:
                for j in range(k - 1, 0, -1):
                    if words[i + j - 1][0].endswith((",", ";", ":", "—")) and j >= 0.6 * k:
                        k = j
                        break
            left = n - (i + k)
            if 0 < left <= 2:
                if _pack(lens[i:n], int(max_chars * 1.15), max_lines) is not None:
                    k = n - i  # absorb a short tail rather than leave an orphan card
                elif left == 1 and k >= 3:
                    k -= 1  # give the last card two words
            lines = _pack(lens[i:i + k], max_chars, max_lines) or _pack(lens[i:i + k], int(max_chars * 1.15), max_lines) \
                or [list(range(k))]
            sub = words[i:i + k]
            cards.append(Card(sub, lines, sub[0][1], sub[-1][2]))
            i += k
    # timing hygiene: no overlaps, minimum duration, bridge tiny gaps
    for a, b in zip(cards, cards[1:]):
        if b.start - a.end < 0.35:
            a.end = b.start
        a.end = min(max(a.end, a.start + min_dur), b.start)
    if cards:
        cards[-1].end = max(cards[-1].end, cards[-1].start + min_dur)
    return cards


def caption_metrics(W: int, H: int, cs: CaptionSettings) -> dict:
    portrait = H > W * 1.05
    square = abs(W - H) <= 0.05 * max(W, H)
    base = min(W, H)
    if portrait:
        size = 0.072 * W
        max_chars = cs.max_chars_per_line or 20
        margin_v = int(0.24 * H) if cs.position == "bottom" else int(0.06 * H)
    elif square:
        size = 0.062 * W
        max_chars = cs.max_chars_per_line or 26
        margin_v = int(0.12 * H)
    else:
        size = 0.058 * base
        max_chars = cs.max_chars_per_line or 38
        margin_v = int(0.075 * H)
    size *= cs.size_scale
    align = {"bottom": 2, "middle": 5, "top": 8}[cs.position]
    return dict(size=int(round(size)), max_chars=max_chars, margin_v=margin_v, align=align, portrait=portrait,
                outline=max(2, int(round(size * 0.075))), shadow=max(1, int(round(size * 0.05))),
                margin_h=int(0.06 * W))


def write_ass(path, cards: list[Card], titles: list[tuple[float, float, str]], W: int, H: int, cs: CaptionSettings,
              burn_captions: bool = True):
    m = caption_metrics(W, H, cs)
    bold = -1 if cs.font_weight_bold else 0
    border_style = 3 if cs.box else 1
    back = ass_color("#000000", 0x60) if cs.box else ass_color("#000000", 0x80)
    t_size = int(m["size"] * 0.62)
    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {W}", f"PlayResY: {H}", "WrapStyle: 2",
        "ScaledBorderAndShadow: yes", "YCbCr Matrix: TV.709", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding",
        f"Style: Caption,{cs.font},{m['size']},{ass_color(cs.color)},{ass_color(cs.highlight_color)},"
        f"{ass_color(cs.outline_color)},{back},{bold},0,0,0,100,100,0,0,{border_style},{m['outline']},{m['shadow']},"
        f"{m['align']},{m['margin_h']},{m['margin_h']},{m['margin_v']},1",
        f"Style: Title,{cs.font},{t_size},{ass_color('#FFFFFF')},{ass_color(cs.highlight_color)},"
        f"{ass_color('#000000', 0x40)},{ass_color('#101418', 0x50)},-1,0,0,0,100,100,0,0,3,{max(6, t_size // 3)},0,"
        f"7,{int(0.045 * W)},{int(0.045 * W)},{int(0.06 * H)},1",
        "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    hl = ass_color(cs.highlight_color)
    base = ass_color(cs.color)
    if burn_captions:
        for c in cards:
            n = len(c.words)
            for k in range(n):
                s = c.words[k][1] if k else c.start
                e = c.words[k + 1][1] if k + 1 < n else c.end
                if e - s < 0.01:
                    continue
                text_lines = []
                for ln in c.lines:
                    toks = []
                    for wi in ln:
                        w = _esc(c.words[wi][0])
                        if cs.style == "karaoke" and wi == k:
                            toks.append(f"{{\\c{hl}\\fscx106\\fscy106}}{w}{{\\c{base}\\fscx100\\fscy100}}")
                        else:
                            toks.append(w)
                    text_lines.append(" ".join(toks))
                fade = "{\\fad(80,0)}" if k == 0 else ""
                body = "\\N".join(text_lines)
                lines.append(f"Dialogue: 0,{fmt_ass(s)},{fmt_ass(e)},Caption,,0,0,0,,{fade}{body}")
    for s, e, txt in titles:
        lines.append(f"Dialogue: 1,{fmt_ass(s)},{fmt_ass(e)},Title,,0,0,0,,{{\\fad(200,250)}}{_esc(txt)}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _esc(t: str) -> str:
    return t.replace("\\", "\\\\").replace("{", "(").replace("}", ")")


def write_srt(path, cards: list[Card]):
    out = []
    for i, c in enumerate(cards, 1):
        out += [str(i), f"{fmt_srt(c.start)} --> {fmt_srt(c.end)}", c.text, ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))


def write_vtt(path, cards: list[Card]):
    out = ["WEBVTT", ""]
    for c in cards:
        out += [f"{fmt_srt(c.start, '.')} --> {fmt_srt(c.end, '.')}", c.text, ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
