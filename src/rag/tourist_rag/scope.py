"""
Scope guard + nationality extraction for the Tourist Help RAG.

Ra7al serves people who are COMING TO Egypt. So:
  * "How do I get a visa for France?"      -> out of scope (foreign destination)
  * "I'm French, do I need a visa?"        -> in scope (France is the passport, not the destination)
  * "Visa for Egypt from France?"          -> in scope (Egypt is mentioned)

Module: src/rag/scope.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.rag.text import normalize_arabic

# canonical (matches the visa CSV where possible) | place aliases | demonyms
_COUNTRY_TABLE = """
Albania|albania|albanian
Argentina|argentina|argentine,argentinian
Armenia|armenia|armenian
Australia|australia|australian
Austria|austria|austrian
Azerbaijan|azerbaijan|azerbaijani
Bahrain|bahrain|bahraini
Belarus|belarus|belarusian
Belgium|belgium|belgian
Bolivia|bolivia|bolivian
Bosnia and Herzegovina|bosnia,bosnia and herzegovina|bosnian
Brazil|brazil|brazilian
Bulgaria|bulgaria|bulgarian
Canada|canada|canadian
Chile|chile|chilean
China|china,beijing,shanghai|chinese
Taiwan|taiwan,taipei|taiwanese
Colombia|colombia|colombian
Croatia|croatia|croatian
Cyprus|cyprus|cypriot
Czech Republic|czech republic,czechia,prague|czech
Denmark|denmark,copenhagen|danish,dane
Ecuador|ecuador|ecuadorian
Estonia|estonia|estonian
Finland|finland,helsinki|finnish,finn
France|france,paris,marseille|french,frenchman,frenchwoman
Georgia|georgia,tbilisi|georgian
Germany|germany,berlin,munich,frankfurt|german
Greece|greece,athens|greek
Hungary|hungary,budapest|hungarian
Iceland|iceland|icelandic,icelander
India|india,delhi,mumbai|indian
Iraq|iraq,baghdad|iraqi
Ireland|ireland,dublin|irish
Italy|italy,rome,milan,venice,florence|italian
Japan|japan,tokyo,osaka,kyoto|japanese
South Korea|south korea,korea,seoul|korean
Kuwait|kuwait|kuwaiti
Latvia|latvia|latvian
Lithuania|lithuania|lithuanian
Luxembourg|luxembourg|luxembourgish
North Macedonia|north macedonia,macedonia|macedonian
Malaysia|malaysia,kuala lumpur|malaysian
Malta|malta|maltese
Mexico|mexico|mexican
Moldova|moldova|moldovan
Monaco|monaco|monegasque
Montenegro|montenegro|montenegrin
Netherlands|netherlands,holland,amsterdam|dutch
New Zealand|new zealand|kiwi
Norway|norway,oslo|norwegian
Oman|oman,muscat|omani
Paraguay|paraguay|paraguayan
Peru|peru,lima|peruvian
Poland|poland,warsaw|polish
Portugal|portugal,lisbon|portuguese
Qatar|qatar,doha|qatari
Kosovo|kosovo|kosovar
Romania|romania,bucharest|romanian
Russian Federation|russia,russian federation,moscow|russian
San Marino|san marino|sammarinese
Saudi Arabia|saudi arabia,saudi,riyadh,jeddah,mecca,medina|saudi
Serbia|serbia,belgrade|serbian
Singapore|singapore|singaporean
Slovakia|slovakia|slovak
Slovenia|slovenia|slovenian
South Africa|south africa,cape town,johannesburg|south african
Spain|spain,madrid,barcelona|spanish,spaniard
Sweden|sweden,stockholm|swedish,swede
Switzerland|switzerland,zurich,geneva|swiss
Ukraine|ukraine,kyiv,kiev|ukrainian
United Arab Emirates|united arab emirates,uae,dubai,abu dhabi,emirates|emirati
United Kingdom|united kingdom,uk,britain,great britain,england,scotland,wales,london|british,briton,scottish,welsh
United States|united states,usa,america,the states,new york,los angeles,california,texas,florida|american
Uruguay|uruguay|uruguayan
Vatican City|vatican,vatican city|
Venezuela|venezuela|venezuelan
Vietnam|vietnam,hanoi|vietnamese
Nigeria|nigeria,lagos|nigerian
Pakistan|pakistan,karachi,islamabad|pakistani
Turkey|turkey,turkiye,istanbul,ankara|turkish
Israel|israel,tel aviv|israeli
Iran|iran,tehran|iranian
Syria|syria,damascus|syrian
Libya|libya,tripoli|libyan
Sudan|sudan,khartoum|sudanese
Yemen|yemen|yemeni
Jordan|jordan,amman|jordanian
Lebanon|lebanon,beirut|lebanese
Palestine|palestine|palestinian
Morocco|morocco,casablanca,marrakech|moroccan
Algeria|algeria,algiers|algerian
Tunisia|tunisia,tunis|tunisian
Indonesia|indonesia,jakarta,bali|indonesian
Philippines|philippines,manila|filipino,filipina
Thailand|thailand,bangkok,phuket|thai
Bangladesh|bangladesh,dhaka|bangladeshi
Sri Lanka|sri lanka|sri lankan
Nepal|nepal|nepali,nepalese
Afghanistan|afghanistan,kabul|afghan
Ghana|ghana|ghanaian
Kenya|kenya,nairobi|kenyan
Ethiopia|ethiopia,addis ababa|ethiopian
Cuba|cuba,havana|cuban
"""

# Regions that are never an Egypt trip.
_EXTRA_FOREIGN_PLACES = ["europe", "schengen", "european union", "asia"]

# Arabic names of foreign destinations (normalised spelling: no hamza/ta-marbuta variants).
ARABIC_PLACES: Dict[str, str] = {
    "فرنسا": "France", "باريس": "France", "ايطاليا": "Italy", "روما": "Italy", "المانيا": "Germany",
    "برلين": "Germany", "اسبانيا": "Spain", "مدريد": "Spain", "برشلونه": "Spain", "بريطانيا": "United Kingdom",
    "انجلترا": "United Kingdom", "لندن": "United Kingdom", "امريكا": "United States", "الولايات المتحده": "United States",
    "كندا": "Canada", "استراليا": "Australia", "تركيا": "Turkey", "اسطنبول": "Turkey", "اليونان": "Greece",
    "دبي": "United Arab Emirates", "الامارات": "United Arab Emirates", "السعوديه": "Saudi Arabia",
    "الاردن": "Jordan", "اليابان": "Japan", "الصين": "China", "الهند": "India", "اوروبا": "Europe",
    "شنجن": "Schengen", "هولندا": "Netherlands", "سويسرا": "Switzerland", "السويد": "Sweden",
}

EGYPT_TERMS = [
    "egypt", "egyptian", "cairo", "giza", "luxor", "aswan", "alexandria", "sinai", "sharm", "hurghada",
    "dahab", "siwa", "nile", "pyramid", "pyramids", "sphinx", "marsa alam", "marsa matruh", "taba", "nuweiba",
    "fayoum", "faiyum", "bahariya", "farafra", "abu simbel", "saint catherine", "st catherine", "el gouna",
    "red sea", "white desert", "black desert", "karnak", "valley of the kings", "ras mohammed", "safaga",
    "مصر", "المصرية", "المصري", "القاهرة", "الجيزة", "الاقصر", "الأقصر", "اسوان", "أسوان", "الاسكندرية",
    "الإسكندرية", "سيناء", "شرم", "الغردقة", "دهب", "سيوة", "النيل", "الاهرامات", "الأهرامات",
]

COUNTRY_ALIASES: Dict[str, str] = {}   # place alias  -> canonical country
DEMONYMS: Dict[str, str] = {}          # demonym      -> canonical country
for _line in _COUNTRY_TABLE.strip().splitlines():
    _canon, _places, _dems = _line.split("|")
    for _p in filter(None, _places.split(",")):
        COUNTRY_ALIASES[_p.strip()] = _canon
    for _d in filter(None, _dems.split(",")):
        DEMONYMS[_d.strip()] = _canon
    COUNTRY_ALIASES.setdefault(_canon.lower(), _canon)


def _alt(words: List[str]) -> str:
    return "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))


_EGYPT_RE = re.compile(rf"(?<![a-z\u0600-\u06FF])(?:{_alt([t.lower() for t in EGYPT_TERMS])})(?![a-z])", re.I)
_PLACE_RE = re.compile(rf"\b(?:{_alt(list(COUNTRY_ALIASES) + _EXTRA_FOREIGN_PLACES)})\b", re.I)
_DEMONYM_RE = re.compile(rf"\b(?:{_alt(list(DEMONYMS))})s?\b", re.I)
_US_PASSPORT_RE = re.compile(r"\b(?:us|u\.s\.?|u\.s\.a\.?)\s+(?:passport|citizen|national|resident)", re.I)

# "from X", "citizen of X" ... => X describes the traveller, not the destination
_ORIGIN_BEFORE = re.compile(r"(?:\bfrom|citizens?\s+of|nationals?\s+of|passport\s+(?:of|from|issued\s+by)|born\s+in|live\s+in|living\s+in|resident\s+of|based\s+in)\s+(?:the\s+)?$", re.I)
_ORIGIN_AFTER = re.compile(r"^\s*(?:passport|citizen|citizens|national|nationals|resident|residents|driving\s+licen[cs]e|id\b)", re.I)

_VISA_RE = re.compile(
    r"\b(?:visa|visas|evisa|e-visa|e\s+visa|passport|entry\s+requirements?|immigration|voa|visa\s+on\s+arrival)\b"
    r"|تأشيرة|تاشيرة|فيزا|جواز", re.I)

_STRONG_NATIONALITY_CUES = [
    re.compile(r"\b(?:i\s*am|i'?m|im|we\s*are|we'?re)\s+(?:from|a\s+citizen\s+of)\s+(?:the\s+)?(?P<c>[a-z .]+?)(?=[,.?!;]|\s+(?:and|but|so|do|can|how|what|will|need|am|is|are|going|traveling|travelling|coming|visiting)\b|$)", re.I),
    re.compile(r"\b(?:citizen|national|resident)\s+of\s+(?:the\s+)?(?P<c>[a-z .]+?)(?=[,.?!;]|\s+(?:and|but|so|do|can|how|what)\b|$)", re.I),
    re.compile(r"\b(?:passport\s+(?:of|from|issued\s+by))\s+(?:the\s+)?(?P<c>[a-z .]+?)(?=[,.?!;]|\s+(?:and|but|so|do|can|how|what)\b|$)", re.I),
]


@dataclass
class ScopeDecision:
    in_scope: bool
    reason: str = "ok"                 # ok | foreign_destination
    foreign_place: Optional[str] = None


def mentions_egypt(text: str) -> bool:
    return bool(_EGYPT_RE.search(text.lower())) or bool(_EGYPT_RE.search(text))


def is_visa_query(text: str) -> bool:
    return bool(_VISA_RE.search(text))


def _is_origin_mention(text: str, start: int, end: int) -> bool:
    before = text[max(0, start - 30):start]
    after = text[end:end + 25]
    return bool(_ORIGIN_BEFORE.search(before) or _ORIGIN_AFTER.match(after))


def check_scope(text: str) -> ScopeDecision:
    """Decide whether the question is about travelling in Egypt."""
    if mentions_egypt(text):
        return ScopeDecision(True)
    for m in _PLACE_RE.finditer(text.lower()):
        if _is_origin_mention(text.lower(), m.start(), m.end()):
            continue
        place = m.group(0)
        return ScopeDecision(False, "foreign_destination", COUNTRY_ALIASES.get(place, place.title()))
    norm = normalize_arabic(text)
    for alias, canon in ARABIC_PLACES.items():
        if alias in norm and f"من {alias}" not in norm:
            return ScopeDecision(False, "foreign_destination", canon)
    return ScopeDecision(True)


def extract_nationality(text: str) -> Optional[str]:
    """Best-effort passport nationality from free text. Returns the canonical country name or None."""
    low = text.lower()
    for pattern in _STRONG_NATIONALITY_CUES:
        m = pattern.search(low)
        if m:
            raw = m.group("c").strip()
            if raw in COUNTRY_ALIASES:
                return COUNTRY_ALIASES[raw]
            if raw in DEMONYMS:
                return DEMONYMS[raw]
    for m in _PLACE_RE.finditer(low):                       # "French passport" / "US citizen" / "France national"
        if _ORIGIN_AFTER.match(low[m.end():m.end() + 25]):
            return COUNTRY_ALIASES.get(m.group(0))
    if _US_PASSPORT_RE.search(low):
        return "United States"
    m = _DEMONYM_RE.search(low)                             # "I'm American" / "Do Americans need..."
    if m:
        word = m.group(0)
        return DEMONYMS.get(word) or DEMONYMS.get(word[:-1])
    return None


def canonical_country(name: Optional[str]) -> Optional[str]:
    """Normalise a nationality/country string that came from stored user preferences."""
    if not name:
        return None
    low = name.strip().lower()
    return COUNTRY_ALIASES.get(low) or DEMONYMS.get(low)
