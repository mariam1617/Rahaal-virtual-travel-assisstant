"""
Answer layer of the Tourist Help RAG.

    question -> scope guard -> retrieve small chunks -> answer ONLY what was asked

Answer length control (the "don't paste the whole PDF" requirement):
  1. retrieval returns 1-3 topic-sized chunks (see kb.py);
  2. the LLM prompt demands a short answer and forbids unrelated facts, OR
  3. with no LLM key, the extractive composer keeps only the sentences/bullets that
     overlap with the question (hard-capped by MAX_ANSWER_CHARS).

Visa questions are answered per nationality from the CSV row, picking only the aspect asked
(fee / passport validity / documents / visa on arrival / general eligibility).

Module: src/rag/answer.py
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.rag.kb import Hit, KnowledgeBase, get_knowledge_base
from src.rag.scope import (
    canonical_country, check_scope, extract_nationality, is_visa_query,
)
from src.rag.text import is_arabic, query_groups

MAX_ANSWER_CHARS = 650
MAX_UNITS = 4
VISA_PORTAL = "https://www.visa2egypt.gov.eg"

CORE_EMERGENCY_NUMBERS = ("Tourist Police", "Police Emergency", "Ambulance", "Civil Protection and Fire Services")


@dataclass
class RAGResult:
    status: str                       # answered | out_of_scope | no_answer
    answer: str
    context: str = ""                 # exact text the answer was built from (fed to the verifier)
    sources: List[Dict[str, str]] = field(default_factory=list)
    domain: str = "advisories"
    pending_question: str = ""


# =========================================================================
# Canned messages (short, bilingual)
# =========================================================================

def _out_of_scope_message(place: Optional[str], arabic: bool) -> str:
    if arabic:
        where = f" ({place})" if place else ""
        return (
            f"أنا مساعد رحال، ومتخصص في مساعدة الزوار اللي جايين مصر بس 🇪🇬. "
            f"مقدرش أساعدك في تأشيرات أو سفر لبلد تانية{where}؛ الأفضل تراجع سفارة البلد دي أو موقعها الرسمي. "
            "لو عندك أي سؤال عن رحلتك لمصر (التأشيرة، المواصلات، الأمان، الطوارئ) أنا موجود."
        )
    where = f" for {place}" if place else ""
    return (
        f"I'm Ra7al, and I only help visitors travelling to Egypt 🇪🇬, so I can't help with visas or travel{where}. "
        "For that, please check that country's embassy or official government website. "
        "If you have a question about your trip to Egypt (visa, getting around, safety, emergencies), I'm happy to help."
    )


def _no_answer_message(arabic: bool) -> str:
    if arabic:
        return ("معنديش معلومة مؤكدة عن السؤال ده في قاعدة بياناتي. "
                "تقدر تتواصل مع الخط الساخن لهيئة السياحة المصرية على 19654، أو تسألني عن التأشيرة، المواصلات، الأمان أو الطوارئ في مصر.")
    return ("I don't have reliable information on that in my Egypt travel knowledge base. "
            "You can call the Egyptian tourism hotline on 19654, or ask me about visas, getting around, safety or emergencies in Egypt.")


# =========================================================================
# Source line
# =========================================================================

def _source_line(hits: List[Hit]) -> str:
    seen, parts = set(), []
    for h in hits:
        c = h.chunk
        key = (c.source_org, c.source)
        if key in seen or not (c.source or c.source_org):
            continue
        seen.add(key)
        label = c.source_org or c.title
        if c.as_of:
            label += f", as of {c.as_of}"
        parts.append(f"{label} ({c.source})" if c.source else label)
    return ("Source: " + "; ".join(parts[:2])) if parts else ""


def _sources_meta(hits: List[Hit]) -> List[Dict[str, str]]:
    return [{"id": h.chunk.id, "title": h.chunk.title, "section": h.chunk.section, "url": h.chunk.source} for h in hits]


# =========================================================================
# Extractive composer: keep only what matches the question
# =========================================================================

_UNIT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_BOILERPLATE = re.compile(r"^(related (?:guidance|source)|source|permit information|insurance guidance)s?:?$", re.I)
_SOURCE_SAYS = re.compile(r"^The source (?:states|says|notes|explains) that ", re.I)
_SOURCE_VERB = re.compile(r"^The source ", re.I)


@dataclass
class Unit:
    text: str
    bullet: bool = False


def _short_org(chunk) -> str:
    org = chunk.source_org or "The source"
    m = re.search(r"\(([A-Z]{2,})\)", org)
    return m.group(1) if m else org


def _attribute(text: str, chunk) -> str:
    """The markdown says 'The source advises ...'. Name the source instead (FCDO / Experience Egypt)."""
    if _SOURCE_SAYS.match(text):
        text = _SOURCE_SAYS.sub("", text)
        return text[:1].upper() + text[1:]
    return _SOURCE_VERB.sub(f"{_short_org(chunk)} ", text)


def _row_units(chunk) -> List[Unit]:
    """CSV rows -> one compact, readable unit, dropping fields the data says are 'Not specified'."""
    kept = []
    for sentence in _UNIT_SPLIT.split(chunk.text):
        sentence = sentence.strip()
        if not sentence or "not specified" in sentence.lower() or re.search(r":\s*\.?$", sentence):
            continue
        if sentence.startswith(("Nationality:", "e-Visa eligible:", "Visa on arrival:")):
            continue
        kept.append(re.sub(r"^Fee: 0 USD\.?$", "Fee: free.", sentence))
    return [Unit(" ".join(kept))] if kept else []


def _units(chunk) -> List[Unit]:
    if chunk.kind == "visa_row":
        return _row_units(chunk)
    out: List[Unit] = []
    for line in chunk.text.splitlines():
        line = line.strip()
        if not line or re.match(r"^(?:[-*]\s*)?https?://\S+$", line) or _BOILERPLATE.match(line):
            continue
        if re.match(r"^[-*•]\s+", line):
            out.append(Unit(re.sub(r"^[-*•]\s+", "", line), bullet=True))
        else:
            out.extend(Unit(_attribute(x.strip(), chunk)) for x in _UNIT_SPLIT.split(line) if x.strip())
    return out


def _list_block(units: List[Unit]) -> Optional[List[Unit]]:
    """'Intro sentence: + 5 or more bullets' -> [intro, *bullets] (a list question wants the whole list)."""
    run, best = [], []
    for i, u in enumerate(units):
        if u.bullet:
            run.append(i)
            if len(run) > len(best):
                best = list(run)
        else:
            run = []
    if len(best) < 5:
        return None
    first = best[0]
    lead = [units[first - 1]] if first > 0 and not units[first - 1].bullet else []
    return lead + [units[i] for i in best]


def _render(units: List[Unit]) -> str:
    if len(units) == 1:
        return units[0].text
    lines = []
    for i, u in enumerate(units):
        lines.append(u.text if (i == 0 and not u.bullet and units[i + 1].bullet) else f"• {u.text}")
    return "\n".join(lines)


SMALL_CHUNK_CHARS = 520


def extractive_answer(kb: KnowledgeBase, query: str, hits: List[Hit],
                      max_units: int = MAX_UNITS, max_chars: int = MAX_ANSWER_CHARS) -> str:
    """
    1. Top chunk is small (one short topic, e.g. "Drones", "Alcohol Laws") -> that topic IS the answer.
    2. Top chunk is long -> keep only the sentences/bullets that overlap with the question.
    3. Question asks for a list ("which airports") -> give that whole list, nothing else.
    """
    groups = query_groups(query)
    units_by_hit = [_units(h.chunk) for h in hits]

    top_units = units_by_hit[0]
    if top_units and len(hits[0].chunk.text) <= SMALL_CHUNK_CHARS:
        return _render(top_units)

    candidates = []                                   # (score, hit_rank, order, unit)
    for rank, units in enumerate(units_by_hit):
        for order, unit in enumerate(units):
            cov = kb.text_coverage(unit.text, groups)
            if cov > 0:
                candidates.append((cov - 0.15 * rank, rank, order, unit))
    if not candidates:                                 # chunk matched on its heading only
        candidates = [(0.0, 0, i, u) for i, u in enumerate(top_units[:2])]
    best_c = max(candidates, key=lambda c: c[0])

    units = units_by_hit[best_c[1]]
    block = _list_block(units)
    if block:
        items = [u for u in block if u.bullet]
        in_block = any(best_c[3] is u for u in block)
        items_match = sum(kb.text_coverage(u.text, groups) > 0 for u in items) >= 0.5 * len(items)
        if in_block or items_match:                     # the question is about the list's contents
            return _render(block)[:1200]

    floor = max(best_c[0] * 0.75, 0.34 if len(groups) > 1 else 0.0)
    keep = [c for c in sorted(candidates, key=lambda c: -c[0]) if c[0] >= floor] or [best_c]
    chosen, used = [], 0
    for c in keep:
        if len(chosen) >= max_units or (chosen and used + len(c[3].text) > max_chars):
            continue
        chosen.append(c)
        used += len(c[3].text)
    chosen.sort(key=lambda c: (c[1], c[2]))            # back to document order
    return _render([c[3] for c in chosen])


# =========================================================================
# LLM composer (optional)
# =========================================================================

ANSWER_SYSTEM_PROMPT = """You are Ra7al (رحال), a travel assistant for people who are VISITING Egypt.

Rules:
1. Answer ONLY the question that was asked, as briefly as possible: usually 1-3 sentences, or at most 4 short bullets. \
Do not add related facts the user did not ask for. Never paste or restate whole sections of the context.
2. Use ONLY the context below. If the context does not contain the answer, say you don't have reliable information on it \
and suggest the Egyptian tourism hotline 19654. Never invent numbers, fees, dates, phone numbers or rules.
3. You only help with travel in/to Egypt. If the question is about another country, politely decline in one sentence.
4. Reply in the same language as the user's question.
5. Do not mention "the context" or "the documents". No headings. Do not add a source line (it is added for you)."""


def get_answer_llm() -> Optional[Any]:
    """Same provider selection as the verification guardrail; None => extractive mode."""
    if os.getenv("RAG_USE_LLM", "1").lower() in ("0", "false", "no"):
        return None
    try:
        from src.agents.tourist_help.verification import get_verification_llm
        return get_verification_llm()
    except Exception:
        return None


def llm_answer(llm: Any, query: str, hits: List[Hit], extra_instruction: str = "") -> Optional[str]:
    from langchain_core.messages import HumanMessage, SystemMessage
    context = "\n\n".join(f"[{i + 1}] ({h.chunk.section})\n{h.chunk.text}" for i, h in enumerate(hits))
    user = f"Context:\n{context}\n\nQuestion: {query}\n{extra_instruction}".strip()
    try:
        reply = llm.invoke([SystemMessage(content=ANSWER_SYSTEM_PROMPT), HumanMessage(content=user)])
        content = getattr(reply, "content", reply)
        if isinstance(content, list):
            content = " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
        content = str(content).strip()
        return content or None
    except Exception:
        return None


# =========================================================================
# Visa answers (per nationality, per aspect)
# =========================================================================

def _fee_text(raw: str) -> str:
    parts = [p.strip() for p in raw.split(";") if p.strip()]
    pretty = []
    for p in parts:
        m = re.match(r"(\d+)\s+(.*)", p)
        pretty.append(f"USD {m.group(1)} ({m.group(2)})" if m else p)
    return " / ".join(pretty)


_SPECIFIC_VISA_TOPICS = re.compile(r"transit|stamp|five[- ]year|5[- ]year|sinai|dahab|taba|nuweiba|diplomatic|700|work|student|study|residen", re.I)


def _visa_aspects(query: str) -> List[str]:
    q = query.lower()
    aspects = []
    if re.search(r"fee|cost|price|how much|pay|\$|usd|دفع|رسوم|بكام", q):
        aspects.append("fee")
    if re.search(r"valid|expir|6 months|six months|months", q):
        aspects.append("validity")
    if re.search(r"document|bring|need to (?:bring|have|show)|requirement|papers|hotel|itinerar|print", q):
        aspects.append("documents")
    if re.search(r"arrival|\bvoa\b|at the airport|on landing", q):
        aspects.append("arrival")
    if re.search(r"how (?:do i|to|can i) (?:apply|get)|apply|portal|website|where", q):
        aspects.append("apply")
    return aspects


def _visa_row_answer(kb: KnowledgeBase, nationality: str, query: str) -> Optional[RAGResult]:
    row_chunk = kb.visa_by_nationality.get(nationality.lower())
    if row_chunk is None:
        return None
    r = row_chunk.meta
    aspects = _visa_aspects(query)
    portal = VISA_PORTAL
    sentences: List[str] = []
    if not aspects:
        sentences.append(f"Passport holders from {nationality} are on the list of nationalities eligible for the Egypt tourist e-Visa, "
                         f"which you apply for online before you travel ({portal}).")
        sentences.append(f"Fees: {_fee_text(r.get('visa_fee', ''))}.")
        sentences.append("Your passport must be valid for at least 6 months after arrival.")
    else:
        if "fee" in aspects:
            sentences.append(f"For a {nationality} passport, the tourist e-Visa costs {_fee_text(r.get('visa_fee', ''))}.")
        if "validity" in aspects:
            sentences.append("Your passport must be valid for at least 6 months after your arrival date.")
        if "documents" in aspects:
            sentences.append("You should have a printed copy of your e-Visa, your travel itinerary and your hotel booking details.")
        if "arrival" in aspects:
            sentences.append(f"I can't confirm visa-on-arrival for {nationality} passports in my data - the listed option is the e-Visa, "
                             "so it's safest to apply online in advance.")
        if "apply" in aspects and "fee" not in aspects:
            sentences.append(f"With a {nationality} passport you can apply for the e-Visa online at {portal}.")
    if aspects and "arrival" not in aspects:
        sentences.append("Please confirm on the official portal before you fly.")
    else:
        sentences.append("Entry can still be refused at the border, and rules can change, so please confirm on the official portal before you fly.")
    return RAGResult(
        status="answered", answer=" ".join(sentences), context=row_chunk.text,
        sources=[{"id": row_chunk.id, "title": row_chunk.title, "section": row_chunk.section, "url": row_chunk.source}],
        domain="advisories",
    )


def _visa_answer(kb: KnowledgeBase, query: str, nationality: Optional[str], arabic: bool, llm: Optional[Any]) -> Optional[RAGResult]:
    """Returns a result for visa questions, or None to fall through to generic retrieval."""
    specific = bool(_SPECIFIC_VISA_TOPICS.search(query))
    if nationality and not specific:
        res = _visa_row_answer(kb, nationality, query)
        if res:
            return res
        # Known nationality but not on the e-Visa list we hold
        hits = kb.search("e-visa eligible nationalities listed countries", domains=["visa"], top_k=2, min_coverage=0.3)
        ctx = "\n\n".join(h.chunk.text for h in hits[:1]) or "Visa and entry rules depend on nationality."
        msg = (f"{nationality} isn't on the e-Visa eligibility list I have, so I can't confirm your entry requirements. "
               "Please check with your nearest Egyptian embassy or consulate before travelling "
               f"(official e-Visa portal: {VISA_PORTAL}).")
        return RAGResult("answered", msg, ctx, _sources_meta(hits[:1]), "advisories")
    if not nationality and not specific:
        # Needs the passport to be precise. Give the nationality-independent basics, then ask.
        hits = kb.search(query, domains=["visa"], top_k=2)
        basics = ("Most visitors apply for the Egypt tourist e-Visa online (single entry USD 30, multiple entry USD 65), "
                  "with a passport valid for at least 6 months after arrival.")
        ask = "Which passport do you hold? I'll check whether it's eligible."
        ctx = "\n\n".join(h.chunk.text for h in hits) or ""
        # Add the exact facts the basics sentence relies on, so the verifier can ground it
        fee_hits = kb.search("tourism e-visa fees single multiple entry", domains=["visa"], top_k=1, min_coverage=0.3)
        pv_hits = kb.search("passport valid 6 months after arrival", domains=["visa"], top_k=1, min_coverage=0.3)
        ctx = "\n\n".join(dict.fromkeys([ctx] + [h.chunk.text for h in fee_hits + pv_hits]))
        return RAGResult("answered", f"{basics} {ask}", ctx, _sources_meta(hits + fee_hits + pv_hits), "advisories", pending_question=ask)
    return None


# =========================================================================
# Public entry points
# =========================================================================

def answer_tourist_query(
    query: str,
    user_nationality: Optional[str] = None,
    kb: Optional[KnowledgeBase] = None,
    llm: Optional[Any] = None,
    use_llm: bool = True,
) -> RAGResult:
    """Visa / safety / transport / laws questions."""
    kb = kb or get_knowledge_base()
    arabic = is_arabic(query)

    scope = check_scope(query)
    if not scope.in_scope:
        return RAGResult("out_of_scope", _out_of_scope_message(scope.foreign_place, arabic), domain="advisories")

    if is_visa_query(query):
        nationality = extract_nationality(query) or canonical_country(user_nationality)
        res = _visa_answer(kb, query, nationality, arabic, llm)
        if res:
            return res

    domains = kb.infer_domains(query)
    hits = kb.search(query, domains=domains)
    if not hits and len(domains) < 3:                  # domain guess may have been too narrow
        hits = kb.search(query, domains=["visa", "advisory", "transit"])
    if not hits:
        return RAGResult("no_answer", _no_answer_message(arabic), domain="advisories")

    llm = llm if llm is not None else (get_answer_llm() if use_llm else None)
    text = llm_answer(llm, query, hits) if llm else None
    if not text:
        text = extractive_answer(kb, query, hits)
    source = _source_line(hits)
    answer = f"{text}\n\n{source}" if source else text
    context = "\n\n".join(h.chunk.text for h in hits)
    return RAGResult("answered", answer, context, _sources_meta(hits), domain="advisories")


_EMERGENCY_EXPANSION = [
    (r"\b(?:rob|stole|steal|theft|thief|mug|assault|attack|harass|scam|cheat|fraud|threat|police)|سرق|نصب|تحرش|اعتد|شرطة", ["Tourist Police", "Police Emergency"]),
    (r"\b(?:sick|ill|injur|hurt|bleed|faint|collaps|unconscious|heart|breath|pain|poison|hospital|doctor|medical|ambulance|overdose)|اسعاف|إسعاف|مستشفى|مستشفي|دكتور|تعب", ["Ambulance"]),
    (r"\b(?:fire|smoke|burning|flames|trapped|rescue|building collapse)|حريق|مطافئ|دخان", ["Civil Protection and Fire Services"]),
    (r"\b(?:accident|crash|collision|cars?|road|highway|driver|traffic)\b|حادث|عربية", ["Traffic Accidents", "Ambulance"]),
    (r"\b(?:gas|leak|smell)\b|غاز", ["Gas Emergency"]),
    (r"\b(?:hack|cyber|phishing)|هاك", ["Cybercrime Reporting"]),
    (r"\b(?:passport|immigration|visa|overstay)|جواز", ["Passport and Immigration Services", "Tourist Police"]),
    (r"\b(?:tourism|hotline|complain|tour guide)", ["Tourism Information Hotline", "Tourist Police"]),
]


def answer_emergency_query(query: str, location: Optional[str] = None, kb: Optional[KnowledgeBase] = None) -> RAGResult:
    """
    Emergency numbers are NEVER generated by an LLM: we pick the matching rows from the verified
    emergency registry (CSV) and print the numbers exactly as stored.
    """
    kb = kb or get_knowledge_base()
    rows = [c for c in kb.chunks if c.kind == "emergency_row"]
    by_service = {c.meta["service"]: c for c in rows}
    arabic = is_arabic(query)

    wanted: List[str] = []
    for pattern, services in _EMERGENCY_EXPANSION:
        if re.search(pattern, query, re.I):
            wanted += [s for s in services if s not in wanted]
    selected = [by_service[s] for s in wanted if s in by_service]
    if not selected:                                    # generic "help!" => the four core numbers
        selected = [by_service[s] for s in CORE_EMERGENCY_NUMBERS if s in by_service]
    seen_numbers, unique = set(), []                    # "Fire" and "Civil Protection" are both 180
    for c in selected:
        if c.meta["number"] not in seen_numbers:
            seen_numbers.add(c.meta["number"])
            unique.append(c)
    selected = unique[:4]

    place = location if location and not location.strip().lower().endswith(" unavailable") else None
    if arabic:
        header = "🚨 اتصل فورًا" + (f" (قريب من {place})" if place else "") + ":"
    else:
        header = "🚨 Call now" + (f" (near {place})" if place else "") + ":"
    lines = [header] + [f"• {c.meta['service']}: {c.meta['number']}" for c in selected]
    if any(c.meta["service"] in ("Tourist Police", "Police Emergency") for c in selected):
        lines.append("If you are in immediate danger, call 122 (Police) or 126 (Tourist Police) right away."
                     if not arabic else "لو في خطر مباشر اتصل بالشرطة 122 أو شرطة السياحة 126 فورًا.")
    return RAGResult(
        "answered", "\n".join(lines), "\n".join(c.text for c in selected),
        [{"id": c.id, "title": c.title, "section": c.section, "url": ""} for c in selected], domain="emergency",
    )
