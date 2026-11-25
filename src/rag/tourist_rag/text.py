"""
Text utilities shared by the Tourist Help RAG (tokenising, stemming, query expansion).
Pure Python on purpose: no model download, works offline, deterministic in tests.
Module: src/rag/text.py
"""

from __future__ import annotations

import re
from typing import Dict, List

_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u0600-\u06FF]+")
_ARABIC_DIACRITICS = re.compile(r"[\u064B-\u0652\u0640]")
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")

STOPWORDS = {
    # English function words
    "a", "an", "the", "i", "me", "my", "we", "our", "you", "your", "is", "are", "am", "was", "were",
    "be", "been", "do", "does", "did", "can", "could", "would", "should", "will", "shall", "may",
    "might", "to", "of", "in", "on", "at", "for", "from", "by", "with", "about", "and", "or", "if",
    "it", "its", "this", "that", "these", "those", "there", "here", "what", "which", "who", "whom",
    "how", "when", "where", "why", "any", "some", "get", "got", "have", "has", "had", "as", "into",
    # chatty filler
    "hello", "hi", "hey", "please", "thanks", "thank", "like", "know", "want", "need", "tell",
    "say", "ask", "question", "info", "information", "help", "just", "really", "also", "much",
    "many", "very", "go", "going", "visit", "visiting", "trip", "travel", "travelling", "traveling",
    "egypt", "egyptian", "allowed", "allow", "permitted", "possible", "okay", "ok", "anyone", "exist",
    "currently", "today", "now", "still", "there's", "isn", "aren", "don", "doesn", "s", "t",
    # Arabic filler
    "في", "من", "الى", "إلى", "على", "عن", "هل", "ما", "ماذا", "كيف", "انا", "أنا", "لو", "ممكن",
}

# Arabic (normalised) -> English retrieval terms. The knowledge base is English, but Egyptian
# teammates/tourists may still type Arabic keywords, so we bridge the most common ones.
ARABIC_BRIDGE: Dict[str, List[str]] = {
    "تاشيره": ["visa"], "فيزا": ["visa"], "جواز": ["passport"], "رسوم": ["fee", "fees"],
    "مواصلات": ["transport", "transportation"], "مترو": ["metro", "train"], "قطار": ["train", "railway"],
    "مطار": ["airport"], "طيران": ["flight", "airport"], "تاكسي": ["taxi"], "اتوبيس": ["bus"],
    "امان": ["safe", "safety"], "امن": ["safe", "safety"], "امنه": ["safe", "safety"], "خطر": ["danger", "risk"],
    "سيناء": ["sinai"], "دهب": ["dahab"], "شرم": ["sharm"], "الاقصر": ["luxor"], "اسوان": ["aswan"],
    "شرطه": ["police"], "اسعاف": ["ambulance"], "مستشفي": ["hospital", "ambulance"],
    "مستشفى": ["hospital", "ambulance"], "حريق": ["fire"], "مطافي": ["fire"], "سرقه": ["robbed", "theft"],
    "سرقوا": ["robbed", "theft"], "نصب": ["scam"], "تحرش": ["harassment"], "حادث": ["accident"],
    "لبس": ["dress", "clothing"], "ملابس": ["dress", "clothing"], "كحول": ["alcohol"], "تصوير": ["photography"],
    "درون": ["drone", "drones"], "رمضان": ["ramadan"], "تامين": ["insurance"],
}

# English synonym expansion: user word -> words that the documents really use.
SYNONYMS: Dict[str, List[str]] = {
    "cab": ["taxi"], "uber": ["ride-hailing", "taxi"], "careem": ["ride-hailing", "taxi"],
    "ridehailing": ["ride-hailing"], "metro": ["train", "railway"], "subway": ["train", "railway"],
    "plane": ["flight", "airport"], "fly": ["flight"], "flying": ["flight"],
    "boat": ["felucca", "ferry", "cruise", "nile"], "ship": ["cruise", "ferry"], "ferry": ["ferry", "boat"],
    "bike": ["bicycle", "motorcycle"], "scooter": ["motorcycle"], "rent": ["rental"], "hire": ["rental"],
    "drive": ["driving", "road", "rental"], "driving": ["road", "driving"],
    "safe": ["advises", "risk", "crime", "security"], "safety": ["advises", "risk", "crime", "security"],
    "dangerous": ["advises", "risk", "crime"], "danger": ["advises", "risk", "crime"],
    "unsafe": ["advises", "risk", "crime"], "security": ["safety", "security"],
    "scam": ["scams", "touts", "crime"], "scams": ["scams", "touts"], "tout": ["touts", "scams"],
    "pickpocket": ["crime", "belongings"], "theft": ["crime", "belongings"], "steal": ["crime", "belongings"],
    "women": ["sexual", "harassment"], "woman": ["sexual", "harassment"], "female": ["sexual", "harassment"],
    "solo": ["women", "harassment"],
    "clothes": ["dress", "code"], "clothing": ["dress", "code"], "wear": ["dress", "code"], "shorts": ["dress", "code"],
    "beer": ["alcohol"], "drink": ["alcohol"], "wine": ["alcohol"],
    "weed": ["drugs"], "cannabis": ["drugs"], "marijuana": ["drugs"],
    "camera": ["photography", "filming"], "photo": ["photography", "filming"], "photos": ["photography", "filming"],
    "gay": ["lgbt"], "lesbian": ["lgbt"], "queer": ["lgbt"], "trans": ["lgbt"],
    "balloon": ["balloons"], "dive": ["watersports"], "diving": ["watersports"], "snorkel": ["watersports"],
    "quad": ["quad"], "atv": ["quad"],
    "id": ["identification"], "passport": ["identification", "passport"],
    "fee": ["fees", "cost"], "cost": ["fees", "fee"], "price": ["fees", "fee"], "pay": ["fees", "fee"],
    "evisa": ["e-visa", "visa"], "voa": ["arrival", "visa"],
    "stay": ["validity", "stay"], "long": ["validity", "stay"],
    "arrive": ["arrival", "airport"], "arriving": ["arrival", "airport"],
    "muslim": ["ramadan", "religion"], "fasting": ["ramadan"],
    "rafah": ["gaza", "border"], "gaza": ["rafah", "border"],
    "libya": ["border", "libya"], "sudan": ["border", "hala'ib"],
    "desert": ["western", "desert", "permits"], "oasis": ["western", "desert"],
    "kids": ["children", "child"], "child": ["custody", "child"],
}


def normalize_arabic(word: str) -> str:
    word = _ARABIC_DIACRITICS.sub("", word)
    word = re.sub("[إأآا]", "ا", word)
    word = word.replace("ة", "ه").replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي")
    return word


STOPWORDS |= {normalize_arabic(w) for w in STOPWORDS}


def is_arabic(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    arabic = sum(1 for c in letters if ARABIC_RE.match(c))
    return arabic / len(letters) > 0.4


def stem(word: str) -> str:
    """Very light English stemmer. Only needs to be *consistent* between documents and queries."""
    if not word.isascii() or word.isdigit():
        return word
    n = len(word)
    if word.endswith("ies") and n > 4:
        return word[:-3] + "y"
    if word.endswith("ing") and n > 5:
        return word[:-3]
    if word.endswith("ed") and n > 4:
        return word[:-2]
    if n > 4 and word.endswith("es") and (word[-3] in "sxz" or word[-4:-2] in ("ch", "sh")):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and n > 3:
        return word[:-1]
    return word


def raw_tokens(text: str) -> List[str]:
    """Lower-cased word tokens without stemming or stop-word removal."""
    text = text.lower().replace("e-visa", "evisa").replace("e visa", "evisa")
    return [normalize_arabic(t) if not t.isascii() else t for t in _TOKEN_RE.findall(text)]


def tokenize(text: str, keep_stopwords: bool = False) -> List[str]:
    out = []
    for tok in raw_tokens(text):
        if not keep_stopwords and tok in STOPWORDS:
            continue
        out.append(stem(tok))
    return out


def query_groups(query: str) -> List[List[str]]:
    """
    Turn a user query into 'term groups'. Each group is the original (stemmed) word plus any
    synonyms / Arabic-bridge words. A document matches a group if it contains ANY member.
    """
    groups: List[List[str]] = []
    seen = set()
    for tok in raw_tokens(query):
        if tok in STOPWORDS:
            continue
        base = stem(tok)
        if base in seen:
            continue
        seen.add(base)
        members = [base]
        extra: List[str] = []
        if not tok.isascii():
            extra += ARABIC_BRIDGE.get(tok, [])
        extra += SYNONYMS.get(tok, [])
        for e in extra:
            for piece in re.split(r"[^a-z0-9]+", e.lower()):
                if piece:
                    s = stem(piece)
                    if s not in members:
                        members.append(s)
        groups.append(members)
    return groups
