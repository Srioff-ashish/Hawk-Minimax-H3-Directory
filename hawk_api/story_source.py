"""A story's source text: the author's own novel (PDF, or plain text), split into parts that become chapters.

Nothing here calls a model. The text is read page by page, split at the novel's own chapter headings when it has
them (an 85-page novel usually does), else at paragraph breaks into parts of a readable size, and any part too long
for one chapter draft is split again at paragraph breaks."""

from __future__ import annotations

import io
import re

#: Largest file accepted, and the most text kept (about 600 pages of a novel).
MAX_BYTES = 50 * 1024 * 1024
MAX_CHARS = 1_500_000
#: A part longer than this is split again: a chapter draft reads its part whole, and much more than this is more
#: than a dozen filmed scenes can carry.
PART_CHARS = 24_000
#: Size of the parts when the text has no chapter headings.
PLAIN_PART_CHARS = 14_000
#: A heading-less chunk at the start (title page, dedication) shorter than this joins the first chapter.
FRONT_MATTER_CHARS = 1_500

_NUMBER_WORDS = ("one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
                 "seventeen|eighteen|nineteen|twenty|thirty|forty|fifty")
#: A line that is a chapter heading: "Chapter 3", "CHAPTER THREE: The Gate", "Ch. 12 -", "Part II", "Adhyay 4".
HEADING = re.compile(
    rf"^\s*(?:chapter|ch\.|part|adhyay|adhyaay|अध्याय)\s+(?:\d+|[ivxlc]+|(?:{_NUMBER_WORDS})(?:[\s-](?:{_NUMBER_WORDS}))?)"
    r"\b[\s:.\-–—]*(.{0,80})$", re.IGNORECASE)


class SourceError(ValueError):
    pass


def extract_text(data: bytes, filename: str) -> tuple[str, int]:
    """(text, pages) of a PDF, or of a .txt / .md file. A scanned PDF with no text layer is refused: there is
    nothing to read without OCR."""
    if len(data) > MAX_BYTES:
        raise SourceError(f"{filename} is {len(data) // (1024 * 1024)} MB; the limit is {MAX_BYTES // (1024 * 1024)} MB.")
    name = (filename or "").lower()
    if name.endswith((".txt", ".md", ".markdown")):
        for encoding in ("utf-8-sig", "utf-16", "cp1252"):
            try:
                text = data.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise SourceError(f"{filename} is not readable text.")
        return _tidy(text)[:MAX_CHARS], 0
    if not (name.endswith(".pdf") or data[:5] == b"%PDF-"):
        raise SourceError(f"{filename} is not a PDF or a text file (.pdf, .txt, .md).")
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - requirements-api.txt installs it
        raise SourceError("Reading PDFs needs the pypdf package on the server (pip install pypdf).") from exc
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                raise SourceError(f"{filename} is password-protected; save it without a password and upload it again.")
        pages = [page.extract_text() or "" for page in reader.pages]
    except SourceError:
        raise
    except Exception as exc:
        raise SourceError(f"{filename} could not be read as a PDF: {exc}") from exc
    text = _tidy("\n\n".join(pages))
    if len(text) < 200:
        raise SourceError(f"{filename} has {len(pages)} pages but almost no text in them: it is probably scanned images. "
                          "Export it from your writing app as a text PDF (or .txt) and upload that.")
    return text[:MAX_CHARS], len(pages)


def _tidy(text: str) -> str:
    """Join the lines a PDF breaks mid-sentence, keep paragraph breaks, drop bare page numbers."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("­", "")
    lines = [line.rstrip() for line in text.split("\n")]
    lines = [line for line in lines if not re.fullmatch(r"\s*(?:page\s*)?\d{1,4}\s*(?:of\s*\d+)?\s*", line, re.I)]
    out: list[str] = []
    for line in lines:
        if not line.strip():
            if out and out[-1] != "":
                out.append("")
            continue
        if HEADING.match(line):
            if out and out[-1] != "":
                out.append("")
            out += [line.strip(), ""]
            continue
        if out and out[-1] != "" and not re.search(r"[.!?\"'”’:;)\]]\s*$", out[-1]):
            out[-1] = (out[-1][:-1] if out[-1].endswith("-") else out[-1] + " ") + line.strip()
        else:
            out.append(line.strip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def split_parts(text: str) -> list[dict]:
    """[{"title", "text"}] in order: the novel's own chapters when it marks them, else even parts."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    sections: list[dict] = []
    current = {"title": "", "paras": []}
    for para in paragraphs:
        first = para.split("\n", 1)[0]
        if HEADING.match(first) and len(first) <= 100:
            if current["paras"] or current["title"]:
                sections.append(current)
            current = {"title": first.strip(), "paras": [para[len(first):].strip()] if "\n" in para else []}
        else:
            current["paras"].append(para)
    sections.append(current)
    titled = [s for s in sections if s["title"]]
    if len(titled) >= 2:
        if sections and not sections[0]["title"]:  # text before the first heading
            front = sections.pop(0)
            if sum(len(p) for p in front["paras"]) < FRONT_MATTER_CHARS:
                pass  # a title page or dedication: not a chapter
            else:
                sections.insert(0, {"title": "Opening", "paras": front["paras"]})
        parts = []
        for section in sections:
            parts += _split_long(section["title"], [p for p in section["paras"] if p], PART_CHARS)
    else:
        parts = _split_long("", paragraphs, PLAIN_PART_CHARS)
        for n, part in enumerate(parts, 1):
            part["title"] = f"Part {n}"
    return [p for p in parts if p["text"].strip()]


def _split_long(title: str, paragraphs: list[str], limit: int) -> list[dict]:
    chunks: list[list[str]] = [[]]
    size = 0
    for para in paragraphs:
        if chunks[-1] and size + len(para) > limit:
            chunks.append([])
            size = 0
        chunks[-1].append(para)
        size += len(para) + 2
    if len(chunks) == 1:
        return [{"title": title, "text": "\n\n".join(chunks[0])}]
    return [{"title": f"{title} ({n} of {len(chunks)})" if title else "", "text": "\n\n".join(chunk)}
            for n, chunk in enumerate(chunks, 1)]


def group_parts(parts: list[int], sizes: list[int], count: int) -> list[list[int]]:
    """Consecutive parts grouped into at most ``count`` chapters of similar length (one part each when there are
    no more parts than chapters)."""
    if count >= len(parts):
        return [[p] for p in parts]
    total = sum(sizes) or 1
    groups: list[list[int]] = [[]]
    done = 0
    for index, (part, size) in enumerate(zip(parts, sizes)):
        left_parts = len(parts) - index
        left_groups = count - len(groups)
        target = total * len(groups) / count
        if groups[-1] and (done + size / 2 > target or left_parts <= left_groups) and len(groups) < count:
            groups.append([])
        groups[-1].append(part)
        done += size
    return groups
