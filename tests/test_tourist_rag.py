"""
Tests for the Tourist Help RAG (src/rag + the advisories/emergency nodes).

Run offline:   pytest tests/test_tourist_rag.py
(no LLM key, no network, no model download: extractive mode is used)
"""

import os

os.environ.setdefault("OPENMAP_API_KEY", "test-key")
os.environ["RAG_USE_LLM"] = "0"          # make the graph tests deterministic even if a key is in .env

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.rag.answer import (
    MAX_ANSWER_CHARS, answer_emergency_query, answer_tourist_query, llm_answer,
)
from src.rag.kb import KnowledgeBase, get_knowledge_base
from src.rag.loader import MAX_CHUNK_CHARS, load_all_chunks, load_visa_rows
from src.rag.scope import (
    COUNTRY_ALIASES, DEMONYMS, check_scope, extract_nationality,
)


@pytest.fixture(scope="module")
def kb() -> KnowledgeBase:
    return get_knowledge_base()


def ask(q: str, **kw):
    return answer_tourist_query(q, use_llm=False, **kw)


# ------------------------------------------------------------------ knowledge base / chunking

def test_chunks_are_small_topic_sized_pieces():
    chunks = load_all_chunks()
    md = [c for c in chunks if c.kind == "md"]
    assert len(md) > 50                                   # documents were split into many topics
    assert max(len(c.text) for c in md) <= MAX_CHUNK_CHARS + 120
    assert {"visa", "advisory", "transit", "emergency"} <= {c.domain for c in chunks}


def test_editorial_notes_and_blank_csv_rows_are_not_indexed():
    chunks = load_all_chunks()
    assert not any("Review note" in c.text for c in chunks)
    assert not any("Source and Verification" in c.section for c in chunks)
    emergency = [c for c in chunks if c.kind == "emergency_row"]
    assert len(emergency) == 14                           # trailing blank rows of the CSV are skipped


def test_every_visa_csv_nationality_can_be_recognised():
    names = {c.meta["nationality"] for c in load_visa_rows() if c.meta["visa_type"] == "Tourism e-Visa"}
    known = set(COUNTRY_ALIASES.values()) | set(DEMONYMS.values())
    assert names <= known, f"no alias/demonym for: {sorted(names - known)}"


# ------------------------------------------------------------------ scope guard

@pytest.mark.parametrize("q", [
    "How do I get a visa for France?",
    "How can I apply for a Schengen visa?",
    "What's the weather in Paris?",
    "best pizza in Rome",
    "visa to Germany",
    "ازاي اطلع فيزا لفرنسا",
])
def test_foreign_destination_questions_are_refused(q):
    assert check_scope(q).in_scope is False
    res = ask(q)
    assert res.status == "out_of_scope"
    assert res.context == "" and res.sources == []        # nothing was retrieved or leaked
    assert "visa" not in res.answer.lower() or "can't help" in res.answer.lower() or "مصر" in res.answer


@pytest.mark.parametrize("q", [
    "I'm French, do I need a visa?",
    "Visa for Egypt from France?",
    "Can I fly from London to Cairo?",
    "انا من فرنسا وعايز اجي مصر",
    "Is it safe in Luxor?",
])
def test_passport_country_or_egypt_mention_is_in_scope(q):
    assert check_scope(q).in_scope is True


def test_refusal_language_follows_the_user():
    assert "مصر" in ask("ازاي اطلع فيزا لفرنسا").answer
    assert "Egypt" in ask("How do I get a visa for France?").answer


@pytest.mark.parametrize("q,expected", [
    ("I'm French, do I need a visa?", "France"),
    ("Do Americans need a visa?", "United States"),
    ("I have a US passport", "United States"),
    ("I am from Nigeria", "Nigeria"),
    ("my German passport expires soon", "Germany"),
    ("do I need a visa", None),
    ("visa for Egypt from France", None),                 # departure country is not a passport
])
def test_extract_nationality(q, expected):
    assert extract_nationality(q) == expected


# ------------------------------------------------------------------ visa answers

def test_visa_with_nationality_gives_that_nationalitys_answer():
    res = ask("I'm French, do I need a visa?")
    assert res.status == "answered"
    assert "France" in res.answer and "USD 30" in res.answer and "6 months" in res.answer
    assert "visa2egypt.gov.eg" in res.answer


def test_visa_fee_question_gets_only_the_fee():
    res = ask("How much is the visa for Americans?")
    assert "USD 30" in res.answer and "USD 65" in res.answer
    assert "6 months" not in res.answer                    # passport validity was not asked
    assert "itinerary" not in res.answer
    assert len(res.answer) < 260


def test_visa_documents_question_gets_only_documents():
    res = ask("I hold a German passport, what documents do I bring?")
    assert "printed" in res.answer and "hotel" in res.answer
    assert "USD" not in res.answer


def test_visa_on_arrival_is_never_promised_without_data():
    res = ask("I am from Nigeria, can I get visa on arrival?")
    assert "isn't on the e-Visa eligibility list" in res.answer
    assert "embassy" in res.answer.lower()
    res2 = ask("I hold a German passport, can I get visa on arrival?")
    assert "can't confirm visa-on-arrival" in res2.answer


def test_visa_without_nationality_asks_for_the_passport():
    res = ask("Do I need a visa?")
    assert res.pending_question and "passport" in res.pending_question.lower()
    assert "USD 30" in res.answer and "6 months" in res.answer


def test_stored_profile_nationality_is_used():
    res = ask("Do I need a visa?", user_nationality="Spanish")
    assert "Spain" in res.answer and not res.pending_question


def test_specific_visa_topic_bypasses_nationality_row():
    res = ask("How long can I stay with a transit visa?")
    assert "96 hours" in res.answer
    assert "USD 30" not in res.answer


# ------------------------------------------------------------------ answers are as long as the question

@pytest.mark.parametrize("q,must,must_not", [
    ("Is it safe to travel to North Sinai?", ["North Sinai"], ["Libya", "Hala", "Western Desert"]),
    ("Can I fly drones?", ["drones"], ["balloon", "alcohol", "Terrorism"]),
    ("Is alcohol allowed?", ["alcohol"], ["drone", "Sinai", "dress"]),
    ("Do I need a permit for the White Desert?", ["permit"], ["alcohol", "drone"]),
    ("Is there Uber in Cairo?", ["Ride-hailing"], ["Sinai", "visa"]),
])
def test_answer_covers_the_question_and_nothing_else(q, must, must_not):
    res = ask(q)
    assert res.status == "answered"
    body = res.answer.split("\n\nSource")[0]
    for word in must:
        assert word.lower() in body.lower(), (q, body)
    for word in must_not:
        assert word.lower() not in body.lower(), (q, body)
    assert len(body) <= MAX_ANSWER_CHARS + 600


def test_context_handed_to_the_verifier_is_only_the_retrieved_chunks(kb):
    res = ask("Is alcohol allowed?")
    total_corpus = sum(len(c.text) for c in kb.chunks)
    assert 0 < len(res.context) < total_corpus * 0.05


def test_list_questions_return_the_complete_list_and_nothing_else():
    res = ask("Which airports are international?")
    for airport in ("Cairo", "Luxor", "Suhag"):
        assert airport in res.answer
    assert "Domestic" not in res.answer and "Port Said" not in res.answer


def test_question_the_knowledge_base_cannot_answer_says_so():
    res = ask("what is the tipping etiquette")
    assert res.status == "no_answer"
    assert "19654" in res.answer


# ------------------------------------------------------------------ emergency

def test_emergency_numbers_match_the_situation():
    robbed = answer_emergency_query("I was robbed near Khan el-Khalili", "Cairo, Egypt")
    assert "126" in robbed.answer and "122" in robbed.answer and "123" not in robbed.answer
    medical = answer_emergency_query("my friend collapsed and is unconscious")
    assert "123" in medical.answer and "126" not in medical.answer and "180" not in medical.answer
    fire = answer_emergency_query("there is a fire in my hotel")
    assert "180" in fire.answer and "123" not in fire.answer
    crash = answer_emergency_query("car accident on the highway")
    assert "01221110000" in crash.answer


def test_generic_help_gives_the_four_core_numbers():
    res = answer_emergency_query("help")
    for number in ("126", "122", "123", "180"):
        assert number in res.answer


def test_emergency_never_prints_unknown_numbers(kb):
    registry = {c.meta["number"] for c in kb.chunks if c.kind == "emergency_row"}
    for q in ("robbed", "ambulance please", "fire", "gas leak", "lost my passport", "help"):
        import re
        printed = set(re.findall(r"\b\d{3,12}\b", answer_emergency_query(q).answer))
        assert printed <= registry, (q, printed - registry)


def test_unavailable_location_does_not_leak_into_the_answer():
    res = answer_emergency_query("help", "Location unavailable")
    assert "Location unavailable" not in res.answer


# ------------------------------------------------------------------ LLM path (fake model)

class FakeLLM:
    def __init__(self, text="Alcohol is only legal in licensed venues.", fail=False):
        self.text, self.fail, self.prompts = text, fail, []

    def invoke(self, messages):
        if self.fail:
            raise RuntimeError("quota exceeded")
        self.prompts.append("\n".join(m.content for m in messages))
        return AIMessage(content=self.text)


def test_llm_only_sees_retrieved_chunks_and_a_brevity_instruction(kb):
    llm = FakeLLM()
    res = answer_tourist_query("Is alcohol allowed?", kb=kb, llm=llm)
    prompt = llm.prompts[0]
    assert "ONLY the question that was asked" in prompt
    assert "Alcohol" in prompt and "Drones" not in prompt and "Sinai" not in prompt
    assert len(prompt) < 3500
    assert res.answer.startswith("Alcohol is only legal in licensed venues.")
    assert "Source:" in res.answer


def test_llm_failure_falls_back_to_extractive(kb):
    res = answer_tourist_query("Is alcohol allowed?", kb=kb, llm=FakeLLM(fail=True))
    assert res.status == "answered" and "alcohol" in res.answer.lower()


def test_llm_is_not_called_for_out_of_scope_questions(kb):
    llm = FakeLLM()
    res = answer_tourist_query("How do I get a visa for France?", kb=kb, llm=llm)
    assert res.status == "out_of_scope" and llm.prompts == []


# ------------------------------------------------------------------ LangGraph integration

@pytest.fixture(scope="module")
def app():
    from src.agents.tourist_help.graph import tourist_help_app
    return tourist_help_app


def test_graph_refuses_foreign_visa_question_without_calling_the_verifier(app):
    out = app.invoke({"messages": [("user", "How do I get a visa for France?")], "user_query": "How do I get a visa for France?"})
    assert out["out_of_scope"] is True
    assert out["rag_status"] == "out_of_scope"
    assert out["verification_result"]["skipped"] == "out_of_scope"
    assert "only help visitors travelling to Egypt" in out["verified_response"]


def test_graph_answers_visa_question_with_sources_and_verification(app):
    out = app.invoke({"messages": [("user", "I'm Italian, how much is the visa?")]})
    assert out["is_verified"] is True
    assert "USD 30" in out["verified_response"]
    assert out["retrieval_sources"]
    assert out["help_domain"] == "advisories"


def test_nodes_use_the_latest_message_not_a_stale_user_query():
    from src.agents.tourist_help.advisories import advisories_rag_node
    out = advisories_rag_node({
        "messages": [HumanMessage(content="Is it safe in Sinai?"), AIMessage(content="..."),
                     HumanMessage(content="Is alcohol allowed?")],
        "user_query": "Is it safe in Sinai?",              # stale value from the previous turn
    })
    assert "alcohol" in out["response"].lower() and "Sinai" not in out["response"].split("Source")[0]


@pytest.mark.parametrize("q,expected", [
    ("Is it safe to travel to North Sinai?", "North Sinai"),     # intent parser has no 'safe' keyword
    ("Is alcohol allowed?", "alcohol"),
    ("Can I fly drones?", "drones"),
])
def test_graph_fallback_tries_the_knowledge_base_before_giving_up(app, q, expected):
    out = app.invoke({"messages": [("user", q)]})
    assert expected.lower() in out["verified_response"].lower()
    assert out["is_verified"] is True and out["retrieval_sources"]


def test_graph_fallback_still_greets_when_nothing_matches(app):
    out = app.invoke({"messages": [("user", "Hello there")]})
    assert out["current_intent"] == "fallback"
    assert "help with your trip in Egypt" in out["response"]


def test_graph_refuses_foreign_question_that_reaches_the_fallback(app):
    out = app.invoke({"messages": [("user", "best pizza in Rome")]})
    assert out["rag_status"] == "out_of_scope"


def test_graph_emergency_flow_is_verified_and_specific(app):
    out = app.invoke({"messages": [("user", "I was robbed, call the police")]})
    assert out["help_domain"] == "emergency" and out["is_verified"] is True
    assert "126" in out["verified_response"] and "123" not in out["verified_response"]
