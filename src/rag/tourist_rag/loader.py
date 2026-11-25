"""
Knowledge-base loader for the Tourist Help RAG.

Why chunk like this (this is what keeps answers "as long as the question"):
  * Markdown is split on its own headings (## / ###), so one chunk == one topic
    (e.g. "Dress Code and Public Behavior"), never the whole document.
  * Sections longer than MAX_CHUNK_CHARS are split again on paragraph/bullet boundaries.
  * Visa CSV: one chunk per nationality row. Emergency CSV: one chunk per phone number.

Module: src/rag/loader.py
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

DB_DIR = Path(__file__).resolve().parents[1] / "db"

ADVISORY_MD = DB_DIR / "egypt_travel_advisories.md"
TRANSIT_MD = DB_DIR / "transit" / "egypt_transportation.md"
VISA_MD = DB_DIR / "visa" / "visa_rules.md"
VISA_CSV = DB_DIR / "visa" / "egypt_entry_requirements.csv"
EMERGENCY_CSV = DB_DIR / "emergency_db.csv"
TRANSPORT_CONTACTS_CSV = DB_DIR / "transit" / "transport_contacts.csv"

MAX_CHUNK_CHARS = 800
MIN_CHUNK_CHARS = 30


@dataclass
class Chunk:
    id: str
    domain: str                 # visa | advisory | transit | emergency
    title: str                  # document title
    section: str                # heading path, e.g. "Laws and Cultural Differences > Dress Code"
    text: str                   # body shown to the answerer
    source: str = ""
    source_org: str = ""
    as_of: str = ""
    kind: str = "md"            # md | visa_row | emergency_row
    meta: Dict[str, str] = field(default_factory=dict)

    @property
    def index_text(self) -> str:
        """Text that is searched: heading path + body (headings carry a lot of signal)."""
        return f"{self.section}\n{self.text}"


# --------------------------------------------------------------------------- markdown

def _parse_front_matter(raw: str) -> Tuple[Dict[str, str], str]:
    if not raw.startswith("---"):
        return {}, raw
    end = raw.find("\n---", 3)
    if end == -1:
        return {}, raw
    block, body = raw[3:end], raw[end + 4:]
    meta: Dict[str, str] = {}
    key = None
    for line in block.splitlines():
        if re.match(r"^\s+-\s+", line) and key:
            meta[key] = (meta[key] + " " + line.split("-", 1)[1].strip()).strip()
        elif ":" in line:
            k, v = line.split(":", 1)
            key = k.strip()
            meta[key] = v.strip()
    return meta, body


_HEADING = re.compile(r"^(#{1,4})\s+(.*\S)\s*$")
_URL_ONLY = re.compile(r"^\s*(?:[-*]\s*)?(?:[\w ’'/&+-]{0,40}:\s*)?<?https?://\S+>?\s*$")
_LEAD_NUMBER = re.compile(r"^\d+(?:\.\d+)*\.?\s*")


def _clean_line(line: str) -> str:
    line = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1", line)   # [text](url) -> text
    line = line.replace("**", "").replace("__", "")
    return line.rstrip()


def _split_sections(body: str) -> List[Tuple[str, str]]:
    path: List[str] = []
    buf: List[str] = []
    sections: List[Tuple[str, str]] = []

    def flush() -> None:
        text = "\n".join(buf).strip()
        if text:
            sections.append((" > ".join(path), text))

    for line in body.splitlines():
        m = _HEADING.match(line)
        if m:
            flush()
            buf.clear()
            level = len(m.group(1))
            if level == 1:                      # document title, kept separately
                path.clear()
                continue
            title = _LEAD_NUMBER.sub("", m.group(2)).strip()
            path[:] = path[: level - 2] + [title]
        elif line.lstrip().startswith(">"):     # blockquote = editorial notes, not facts for tourists
            continue
        else:
            buf.append(_clean_line(line))
    flush()
    return sections


def _useful(text: str) -> bool:
    """Drop chunks that are only link lists / boilerplate."""
    lines = [l for l in text.splitlines() if l.strip() and not _URL_ONLY.match(l)]
    lines = [l for l in lines if not re.match(r"^\s*(related guidance|source|permit information|insurance guidance)s?:?\s*$", l, re.I)]
    return len(" ".join(lines).strip()) >= MIN_CHUNK_CHARS


def _split_long(text: str, limit: int = MAX_CHUNK_CHARS) -> List[str]:
    """Split on blank lines; pack paragraphs up to `limit`; split giant paragraphs by sentence."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: List[str] = []
    for p in paragraphs:
        if len(p) <= limit:
            pieces.append(p)
        else:
            sentences = re.split(r"(?<=[.!?])\s+|\n(?=[-*•]\s)", p)
            cur = ""
            for s in sentences:
                if cur and len(cur) + len(s) + 1 > limit:
                    pieces.append(cur.strip())
                    cur = ""
                cur += s + "\n" if s.lstrip().startswith(("-", "*", "•")) else s + " "
            if cur.strip():
                pieces.append(cur.strip())
    chunks: List[str] = []
    cur = ""
    for piece in pieces:
        if cur and len(cur) + len(piece) + 2 > limit:
            chunks.append(cur)
            cur = piece
        else:
            cur = f"{cur}\n\n{piece}" if cur else piece
    if cur:
        chunks.append(cur)
    return chunks


def load_markdown(path: Path, domain: str, id_prefix: str) -> List[Chunk]:
    if not path.exists():
        return []
    meta, body = _parse_front_matter(path.read_text(encoding="utf-8"))
    title = meta.get("title", path.stem)
    source = (meta.get("source") or meta.get("source_url") or meta.get("sources", "")).split()
    source = source[0] if source else ""
    org = meta.get("source_organization", "")
    as_of = meta.get("source_current_as_of") or meta.get("source_last_updated") or meta.get("last_verified", "")
    chunks: List[Chunk] = []
    n = 0
    for section, text in _split_sections(body):
        low = section.lower()
        if low.startswith("source and") or low == "sources" or low.endswith("> sources"):
            continue                                         # maintenance/verification notes are not tourist info
        for part in _split_long(text):
            if re.match(r"^\s*review note:", part, re.I) or not _useful(part):
                continue                                     # editorial notes / link-only chunks
            n += 1
            chunks.append(Chunk(
                id=f"{id_prefix}:{n}", domain=domain, title=title, section=section or title,
                text=part, source=source, source_org=org, as_of=as_of, kind="md",
            ))
    return chunks


# --------------------------------------------------------------------------- CSV

def _read_csv(path: Path, delimiter: str) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        # keys of None hold overflow cells from ragged rows (e.g. trailing tabs) - ignore them
        return [{k.strip(): (v or "").strip() for k, v in row.items() if k is not None and isinstance(v, (str, type(None)))}
                for row in csv.DictReader(fh, delimiter=delimiter)]


def load_visa_rows(path: Path = VISA_CSV) -> List[Chunk]:
    chunks: List[Chunk] = []
    for i, row in enumerate(_read_csv(path, ",")):
        nat = row.get("nationality", "")
        if not nat:
            continue
        text = (
            f"Nationality: {nat}. Visa type: {row.get('visa_type', '')}. "
            f"e-Visa eligible: {row.get('evisa_eligible', '')}. "
            f"Visa on arrival: {row.get('visa_on_arrival_eligible', '')}. "
            f"Fee: {row.get('visa_fee', '')} {row.get('currency', '')}. "
            f"Passport validity: {row.get('passport_validity_requirement', '')}. "
            f"Documents: {row.get('required_documents', '')}. Stay/validity: {row.get('validity_or_stay', '')}. "
            f"Conditions: {row.get('eligibility_conditions', '')}"
        )
        chunks.append(Chunk(
            id=f"visa_row:{i}", domain="visa", title="Egypt entry requirements by nationality",
            section=f"{nat} - {row.get('visa_type', '')}", text=text,
            source=row.get("source_url", ""), source_org="Egypt e-Visa portal / Ministry of Interior",
            as_of=row.get("last_verified", ""), kind="visa_row", meta=row,
        ))
    return chunks


def load_emergency_rows(path: Path = EMERGENCY_CSV) -> List[Chunk]:
    chunks: List[Chunk] = []
    for i, row in enumerate(_read_csv(path, "\t")):
        service, number = row.get("service_name", ""), row.get("phone_number", "")
        if not service or not number:
            continue                                         # file has trailing blank rows
        desc = row.get("description", "").replace("reporting,", "").strip()
        chunks.append(Chunk(
            id=f"emergency:{i}", domain="emergency", title="Egypt emergency numbers", section=service,
            text=f"{service}: {number}. {desc}", source="", source_org="Egypt emergency registry",
            kind="emergency_row",
            meta={"service": service, "number": number, "description": desc,
                  "verification_status": row.get("verification_status", "")},
        ))
    return chunks


def load_all_chunks() -> List[Chunk]:
    chunks: List[Chunk] = []
    chunks += load_markdown(VISA_MD, "visa", "visa_rules")
    chunks += load_visa_rows()
    chunks += load_markdown(ADVISORY_MD, "advisory", "advisory")
    chunks += load_markdown(TRANSIT_MD, "transit", "transit")
    chunks += load_emergency_rows()
    return chunks
