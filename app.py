"""Ra7al - Egypt Travel Concierge (Streamlit front end).

Requires streamlit >= 1.39 (container keys, column vertical alignment).

Demo sign-in: any of the DEMO_USERS usernames with the demo password
(default "ra7al2026", override with the RA7AL_DEMO_PASSWORD environment
variable). Replace `authenticate` with a real auth backend when ready.
"""

import hmac
import logging
import os
import re
import uuid

import streamlit as st
from langgraph.types import Command

from src.agents.execution.graph import execution_app
from src.agents.feedback.graph import feedback_app
from src.agents.planner.graph import planner_app
from src.agents.supervisor.graph import create_supervisor_graph
from src.agents.tourist_help.graph import tourist_help_app
from src.state.schema import TripRequirements
from src.services.auth import AuthService

logger = logging.getLogger("ra7al.ui")

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

ROUTE_LABELS = {
    "planner": "Planner",
    "tourist_help": "Tourist Help",
    "execution": "Booking and Calendar",
    "feedback": "Feedback",
    "profiling": "Travel Interview",
    "out_of_scope": "Out of Scope",
}

ROUTE_DESCRIPTIONS = {
    "planner": "Builds day-by-day itineraries that fit your budget.",
    "tourist_help": "Visas, emergencies, transport and local know-how.",
    "execution": "Bookings and calendar entries, only after you approve.",
    "feedback": "Collects and learns from your trip feedback.",
}

DESTINATIONS = [
    ("Cairo & Giza", "Pyramids, the Sphinx, museums and Khan el-Khalili.", "#C9A227", "#A8841A"),
    ("Luxor", "Karnak, the Valley of the Kings and Nile sunsets.", "#B5523B", "#8E3B28"),
    ("Aswan", "Felucca rides, Philae and Nubian villages.", "#0E6B66", "#0A4F4B"),
    ("Alexandria", "Mediterranean corniche, Qaitbay Citadel and the library.", "#2B6A9E", "#1D4E78"),
    ("Red Sea & Sinai", "Hurghada, Sharm El Sheikh and Dahab: reefs and desert.", "#2BA5A0", "#1D7E7A"),
    ("Siwa", "Palm oasis, salt lakes and dunes in the Western Desert.", "#8A6D3B", "#6B5329"),
]

SUGGESTIONS = [
    ("Plan a trip", "Plan a 5-day trip to Luxor and Aswan with a budget of $900"),
    ("Visa and entry", "What are the visa requirements for visiting Egypt?"),
    ("Emergency help", "What are the emergency numbers in Egypt?"),
    ("Local tips", "What should I know about taxis and tipping in Cairo?"),
]

# Demo accounts. username == user_id used by the agents.
DEMO_USERS = {
    "test_user": {"name": "Test Traveler"},
    "user_101": {"name": "Traveler 101"},
    "user_202": {"name": "Traveler 202"},
}
DEMO_PASSWORD = os.environ.get("RA7AL_DEMO_PASSWORD", "ra7al2026")


st.set_page_config(
    page_title="Ra7al | Egypt Travel Concierge",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------

PYRAMID_BG = (
    "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 560 280'>"
    "<circle cx='400' cy='95' r='54' fill='%23C9A227' opacity='.85'/>"
    "<polygon points='30,280 200,70 370,280' fill='%230E6B66' opacity='.75'/>"
    "<polygon points='200,280 350,115 500,280' fill='%231B5A6B' opacity='.8'/>"
    "<polygon points='340,280 450,160 560,280' fill='%23244760' opacity='.9'/>"
    "</svg>\") no-repeat right bottom / contain"
)

BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Amiri:wght@700&family=Cinzel:wght@500;700&family=Inter:wght@400;500;600&display=swap');
:root { --sand:#F4E6C3; --papyrus:#FBF6E9; --gold:#C9A227; --gold-deep:#A8841A; --teal:#0E6B66; --turq:#2BA5A0; --navy:#12263A; --terra:#B5523B; --ink:#1E2F3D; --muted:#6B7A83; --line:#E6DBB9; }
.stApp, .stMarkdown, .stButton button, input, textarea { font-family:'Inter','Segoe UI',Arial,sans-serif; }
.stApp { background:var(--papyrus); color:var(--ink); }
h1, h2, h3 { font-family:'Cinzel',Georgia,serif; color:var(--navy); letter-spacing:.5px; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display:none !important; }
[data-testid="stHeader"] { background:transparent; }
.block-container { padding-top:.9rem; padding-bottom:3rem; max-width:1180px; }

.egypt-band { height:8px; border-radius:4px; background:repeating-linear-gradient(90deg,var(--gold) 0 14px,var(--teal) 14px 28px,var(--terra) 28px 42px,var(--teal) 42px 56px); }

.st-key-navbar { background:white; border:1px solid var(--line); border-bottom:3px solid var(--gold); border-radius:12px; padding:10px 18px; box-shadow:0 4px 18px rgba(18,38,58,.06); margin-bottom:18px; }
.brand { display:flex; align-items:center; gap:12px; }
.brand-mark { width:42px; height:42px; background:linear-gradient(135deg,var(--teal),var(--navy)); color:var(--gold); display:grid; place-items:center; border-radius:9px; font-family:'Cinzel',serif; font-weight:700; font-size:21px; border:2px solid var(--gold); }
.brand-name { font-family:'Cinzel',serif; font-weight:700; font-size:22px; color:var(--navy); letter-spacing:2px; line-height:1; }
.brand-ar { font-family:'Amiri',serif; color:var(--gold-deep); font-size:20px; margin-left:6px; letter-spacing:0; }
.brand-sub { font-size:10.5px; color:var(--muted); letter-spacing:1.6px; text-transform:uppercase; margin-top:4px; }
.nav-links { display:flex; gap:28px; justify-content:center; }
.nav-links a { color:var(--navy); text-decoration:none; font-weight:500; font-size:14px; padding-bottom:3px; border-bottom:2px solid transparent; }
.nav-links a:hover { border-color:var(--gold); }
.user-chip { display:flex; align-items:center; gap:10px; justify-content:flex-end; }
.avatar { width:34px; height:34px; border-radius:50%; background:var(--teal); color:white; display:grid; place-items:center; font-weight:600; font-size:14px; border:2px solid var(--gold); }
.user-chip .who { line-height:1.15; text-align:right; }
.user-chip .who b { font-size:13.5px; color:var(--navy); }
.user-chip .who span { display:block; font-size:11.5px; color:var(--muted); }

[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] { background:var(--gold); color:var(--navy); border:1px solid var(--gold-deep); border-radius:8px; font-weight:600; }
[data-testid="stBaseButton-primary"]:hover, [data-testid="stBaseButton-primaryFormSubmit"]:hover { background:#D8B538; color:var(--navy); border-color:var(--gold-deep); }
[data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-secondaryFormSubmit"] { background:white; color:var(--navy); border:1px solid var(--line); border-radius:8px; font-weight:500; }
[data-testid="stBaseButton-secondary"]:hover { border-color:var(--gold); color:var(--teal); background:#FFFDF6; }

.st-key-hero { position:relative; overflow:hidden; background:linear-gradient(135deg,#0B4650 0%,#12263A 62%); border-radius:16px; border-bottom:6px solid var(--gold); padding:54px 48px 34px; margin-bottom:26px; }
.st-key-hero::after { content:""; position:absolute; right:0; bottom:0; width:58%; height:100%; background:__PYRAMIDS__; pointer-events:none; }
.st-key-hero > div { position:relative; z-index:1; }
.eyebrow { color:var(--gold); font-size:12.5px; font-weight:600; letter-spacing:3px; text-transform:uppercase; }
.hero-title { font-family:'Cinzel',serif; font-weight:700; color:white; font-size:46px; line-height:1.12; margin:10px 0 14px; max-width:620px; }
.hero-sub { color:#E8EEF0; font-size:17px; line-height:1.6; max-width:540px; margin-bottom:22px; }
.ghost-btn { display:inline-block; padding:7px 16px; border:1px solid rgba(255,255,255,.55); border-radius:8px; color:white !important; text-decoration:none; font-weight:500; font-size:14px; }
.ghost-btn:hover { border-color:var(--gold); color:var(--gold) !important; }
.stat-row { display:flex; gap:36px; margin-top:26px; flex-wrap:wrap; }
.stat b { display:block; color:var(--gold); font-family:'Cinzel',serif; font-size:24px; }
.stat span { color:#CFE0E2; font-size:12.5px; letter-spacing:.5px; }

.section-title { text-align:center; margin:34px 0 6px; font-family:'Cinzel',serif; font-size:28px; font-weight:700; color:var(--navy); }
.section-sub { text-align:center; color:var(--muted); margin-bottom:22px; }
.dest-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:16px; }
.dest-card { background:white; border:1px solid var(--line); border-radius:12px; overflow:hidden; box-shadow:0 2px 10px rgba(18,38,58,.05); transition:transform .15s, box-shadow .15s; }
.dest-card:hover { transform:translateY(-3px); box-shadow:0 10px 24px rgba(18,38,58,.12); }
.dest-top { height:78px; }
.dest-body { padding:14px 18px 18px; }
.dest-body h4 { margin:0 0 6px; font-family:'Cinzel',serif; color:var(--navy); font-size:18px; }
.dest-body p { margin:0; color:var(--muted); font-size:14px; line-height:1.5; }
.svc-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:16px; }
.svc-card { background:white; border:1px solid var(--line); border-top:4px solid var(--gold); border-radius:12px; padding:18px; }
.svc-card h4 { margin:0 0 6px; color:var(--teal); font-size:16px; }
.svc-card p { margin:0; font-size:13.5px; color:var(--muted); line-height:1.5; }
.step-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:16px; }
.step { text-align:center; padding:8px 12px; }
.step .num { width:44px; height:44px; margin:0 auto 10px; border-radius:50%; background:var(--navy); color:var(--gold); display:grid; place-items:center; font-family:'Cinzel',serif; font-weight:700; border:2px solid var(--gold); }
.step h4 { margin:0 0 4px; color:var(--navy); font-size:15.5px; }
.step p { margin:0; font-size:13.5px; color:var(--muted); line-height:1.5; }
.site-footer { margin-top:44px; padding:18px 0 4px; text-align:center; color:var(--muted); font-size:12.5px; }

.auth-card-title { font-family:'Cinzel',serif; font-size:26px; font-weight:700; color:var(--navy); margin:0 0 4px; }
.auth-card-sub { color:var(--muted); margin-bottom:12px; font-size:14px; }
.demo-hint { background:var(--sand); border:1px dashed var(--gold-deep); border-radius:8px; padding:9px 12px; font-size:12.5px; color:var(--navy); margin-top:10px; }
.st-key-authcard { background:white; border:1px solid var(--line); border-top:5px solid var(--gold); border-radius:14px; padding:26px 28px; box-shadow:0 10px 30px rgba(18,38,58,.08); }

.welcome h2 { margin:0; font-size:28px; }
.welcome p { color:var(--muted); margin:4px 0 14px; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#F7ECCE,#F0DFB0); border-right:3px solid var(--gold); }
.side-card { background:white; border:1px solid var(--line); border-radius:12px; padding:14px; display:flex; gap:12px; align-items:center; margin-bottom:6px; }
.side-card b { color:var(--navy); font-size:14px; }
.side-card span { display:block; color:var(--muted); font-size:12px; }
.side-h { font-family:'Cinzel',serif; font-size:13px; letter-spacing:1.5px; text-transform:uppercase; color:var(--gold-deep); margin:16px 0 8px; font-weight:700; }
.route-item { background:rgba(255,255,255,.7); border-left:3px solid var(--teal); border-radius:6px; padding:8px 10px; margin-bottom:6px; }
.route-item b { font-size:13px; color:var(--navy); }
.route-item span { display:block; font-size:12px; color:var(--muted); line-height:1.4; }

[data-testid="stChatMessage"] { background:white; border:1px solid var(--line); border-radius:12px; padding:14px 16px; box-shadow:0 1px 6px rgba(18,38,58,.04); }
[data-testid="stChatInput"] { border:2px solid var(--teal); border-radius:12px; background:white; }
.route-badge { display:inline-block; margin-top:10px; padding:3px 10px; border-radius:999px; background:var(--sand); border:1px solid var(--gold); color:var(--navy); font-size:12px; }
.st-key-suggestions [data-testid="stBaseButton-secondary"] { height:auto; text-align:left; padding:14px 16px; white-space:normal; justify-content:flex-start; border-left:4px solid var(--gold); }
.st-key-approval { border:1px solid var(--gold) !important; border-top:5px solid var(--gold) !important; border-radius:12px !important; background:white; }

@media (max-width: 900px) {
  .dest-grid, .svc-grid, .step-grid { grid-template-columns:1fr 1fr; }
  .hero-title { font-size:34px; }
  .st-key-hero { padding:36px 24px 26px; }
  .nav-links { display:none; }
}
@media (max-width: 560px) { .dest-grid, .svc-grid, .step-grid { grid-template-columns:1fr; } }
"""

HIDE_SIDEBAR_CSS = """
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="stExpandSidebarButton"] { display:none !important; }
"""


def html(markup):
    """Render HTML safely through markdown (strips indentation and blank lines)."""
    flat = "".join(line.strip() for line in markup.strip().splitlines())
    st.markdown(flat, unsafe_allow_html=True)


def inject_css(signed_in):
    css = BASE_CSS.replace("__PYRAMIDS__", PYRAMID_BG)
    if not signed_in:
        css += HIDE_SIDEBAR_CSS
    st.markdown("<style>" + " ".join(css.split("\n")) + "</style>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Cached resources
# ----------------------------------------------------------------------------

from langgraph.checkpoint.memory import MemorySaver

@st.cache_resource
def get_supervisor():
    return create_supervisor_graph(checkpointer=MemorySaver())


@st.cache_resource
def get_auth_service():
    return AuthService()


# ----------------------------------------------------------------------------
# Helpers (unchanged agent-facing logic)
# ----------------------------------------------------------------------------

def extract_destination(text):
    destinations = ["Cairo", "Giza", "Luxor", "Aswan", "Alexandria", "Hurghada", "Sharm El Sheikh", "Dahab", "Siwa"]
    for destination in destinations:
        if destination.lower() in text.lower():
            return destination
    return "Cairo"


def format_itinerary(plan):
    if not isinstance(plan, dict):
        return ""
    lines = [
        f"### {plan.get('title') or 'Your itinerary'}",
        f"Status: **{plan.get('status', 'draft')}**",
    ]
    total_cost = 0.0
    for day in plan.get("days", []):
        day_cost = day.get("estimated_cost", 0.0)
        total_cost += float(day_cost or 0)
        lines.append(f"\n#### Day {day.get('day_number', '?')} - ${float(day_cost or 0):.2f}")
        activities = day.get("activities", [])
        if not activities:
            lines.append("- No activities scheduled.")
            continue
        for activity in activities:
            name = activity.get("name", "Activity")
            time_slot = activity.get("time_slot")
            act_cost = float(activity.get("cost_usd", 0) or 0)
            transit_cost = float(activity.get("transit_cost_usd", 0) or 0)
            prefix = f"[{time_slot.title()}] " if time_slot else ""
            # Show transit cost inline so displayed costs sum to the day total
            if transit_cost > 0:
                lines.append(f"- {prefix}{name} - ${act_cost:.2f} + ${transit_cost:.2f} transport")
            else:
                lines.append(f"- {prefix}{name} - ${act_cost:.2f}")
    lines.append(f"\n**Estimated total: ${total_cost:.2f}**")
    return "\n".join(lines)


def get_interrupt_payload(result):
    interrupts = result.get("__interrupt__", [])
    if not interrupts:
        return None
    first = interrupts[0]
    return getattr(first, "value", first)


def confidence_label(confidence):
    return f"{confidence:.0%}" if isinstance(confidence, (int, float)) else "Not available"


def route_badge(route, confidence):
    html(
        f'<div class="route-badge">Routed to {ROUTE_LABELS.get(route, route)} '
        f"&middot; Confidence {confidence_label(confidence)}</div>"
    )


def initials(name):
    parts = [p for p in re.split(r"\s+", name.strip()) if p]
    return "".join(p[0].upper() for p in parts[:2]) or "R"


def dispatch_specialist(route, text, state):
    session_id = state.get("session_id") or st.session_state.get("session_id", "streamlit-default")
    graph_config = {"configurable": {"thread_id": f"{route}-{session_id}"}}

    if route == "planner":
        day_match = re.search(r"\b(\d+)\s*(?:-| )*(?:day|days|night|nights)\b", text.lower())
        duration = int(day_match.group(1)) if day_match else 3
        budget_match = re.search(r"(?:\$|usd\s*|budget\s*(?:of|is)?\s*\$?)\s*(\d+(?:,\d+)?(?:\.\d+)?)", text.lower())
        budget = float(budget_match.group(1).replace(",", "")) if budget_match else 500.0
        requirements = TripRequirements(
            destination=extract_destination(text),
            duration_days=max(1, min(duration, 14)),
            total_budget=budget,
        )
        result = planner_app.invoke(
            {
                "trip_requirements": requirements,
                "user_preferences": state.get("user_preferences"),
                "messages": [("user", text)],
            },
            config=graph_config,
        )
        interrupt_payload = get_interrupt_payload(result)
        if interrupt_payload:
            plan = interrupt_payload.get("plan") if isinstance(interrupt_payload, dict) else None
            return {
                "text": format_itinerary(plan),
                "approval_required": True,
                "plan": plan,
                "thread_id": graph_config["configurable"]["thread_id"],
                "interrupt_payload": interrupt_payload,
            }
        plan = result.get("current_plan")
        plan_dict = plan.model_dump() if hasattr(plan, "model_dump") else plan
        itinerary_text = format_itinerary(plan_dict) if plan_dict else ""
        response = result.get("response") or "The Planner Agent completed the itinerary request."
        return {"text": f"{itinerary_text}\n\n{response}".strip(), "approval_required": False}

    if route == "tourist_help":
        result = tourist_help_app.invoke(
            {
                "messages": [("user", text)],
                "current_intent": "unclassified",
                "raw_coordinates": state.get("raw_coordinates"),
            },
            config=graph_config,
        )
        return result.get("response") or "The Tourist Help Agent could not find a verified answer."

    if route == "feedback":
        result = feedback_app.invoke(
            {
                "messages": [("user", text)],
                "user_id": state.get("user_id"),
            },
            config=graph_config,
        )
        return result.get("response") or "The Feedback Agent processed your request."

    if route == "execution":
        result = execution_app.invoke(
            {
                "approval_status": "blocked",
                "messages": [("user", text)],
            },
            config=graph_config,
        )
        return result.get("response") or "This booking requires an approved itinerary before it can proceed."

    return None


# ----------------------------------------------------------------------------
# Session state and auth
# ----------------------------------------------------------------------------

def init_state():
    defaults = {
        "messages": [],
        "route_history": [],
        "session_id": str(uuid.uuid4()),
        "pending_approval": None,
        "auth_user": None,
        "show_login": False,
        "auth_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def open_login():
    st.session_state.show_login = True
    st.session_state.auth_error = None


def close_login():
    st.session_state.show_login = False
    st.session_state.auth_error = None


def sign_out():
    st.session_state.auth_user = None
    st.session_state.show_login = False
    st.session_state.auth_error = None
    st.session_state.messages = []
    st.session_state.route_history = []
    st.session_state.pending_approval = None
    st.session_state.session_id = str(uuid.uuid4())


def clear_conversation():
    st.session_state.messages = []
    st.session_state.route_history = []
    st.session_state.pending_approval = None


def queue_prompt(text):
    st.session_state.queued_prompt = text


# ----------------------------------------------------------------------------
# Shared layout pieces
# ----------------------------------------------------------------------------

def render_navbar(user):
    with st.container(key="navbar"):
        left, middle, right = st.columns([2.6, 4, 3], vertical_alignment="center")
        with left:
            html(
                """
                <div class="brand">
                  <div class="brand-mark">R</div>
                  <div>
                    <div class="brand-name">RA7AL<span class="brand-ar">رحال</span></div>
                    <div class="brand-sub">Egypt Travel Concierge</div>
                  </div>
                </div>
                """
            )
        with middle:
            if not user:
                html(
                    """
                    <div class="nav-links">
                      <a href="#destinations">Destinations</a>
                      <a href="#services">Services</a>
                      <a href="#how-it-works">How it works</a>
                    </div>
                    """
                )
        with right:
            if user:
                info, button = st.columns([2.2, 1], vertical_alignment="center")
                with info:
                    html(
                        f"""
                        <div class="user-chip">
                          <div class="who"><b>{user['name']}</b><span>{user['user_id']}</span></div>
                          <div class="avatar">{initials(user['name'])}</div>
                        </div>
                        """
                    )
                with button:
                    st.button("Sign out", on_click=sign_out, use_container_width=True, key="nav_signout")
            else:
                c_in, c_up = st.columns([1, 1.2], vertical_alignment="center")
                with c_in:
                    st.button("Sign in", on_click=open_login, use_container_width=True, key="nav_signin")
                with c_up:
                    st.button("Sign up", type="primary", on_click=open_login, use_container_width=True, key="nav_signup")


def render_footer():
    html('<div class="egypt-band"></div>')
    html(
        """
        <div class="site-footer">
          Ra7al &middot; Egypt-exclusive travel concierge &middot; Itineraries are estimates; nothing is booked or charged without your approval.
        </div>
        """
    )


# ----------------------------------------------------------------------------
# Signed-out views
# ----------------------------------------------------------------------------

def render_landing():
    with st.container(key="hero"):
        html('<div class="eyebrow">Egypt, planned for you</div>')
        html('<div class="hero-title">From the pyramids to the Red Sea, in one conversation.</div>')
        html(
            '<div class="hero-sub">Ra7al is an AI travel concierge built only for Egypt. '
            "Tell it where you want to go and it plans, answers your practical questions, "
            "and asks for your approval before anything is booked.</div>"
        )
        c1, c2, _ = st.columns([1.3, 1.3, 4], vertical_alignment="center")
        with c1:
            st.button("Start planning", type="primary", on_click=open_login, use_container_width=True, key="hero_cta")
        with c2:
            html('<a class="ghost-btn" href="#how-it-works">How it works</a>')
        html(
            """
            <div class="stat-row">
              <div class="stat"><b>9</b><span>Destinations</span></div>
              <div class="stat"><b>4</b><span>Specialist agents</span></div>
              <div class="stat"><b>100%</b><span>Approval before booking</span></div>
            </div>
            """
        )

    html('<div id="destinations"></div><div class="section-title">Where to?</div>')
    html('<div class="section-sub">Egypt is more than one trip. Start with a region.</div>')
    cards = "".join(
        f'<div class="dest-card"><div class="dest-top" style="background:linear-gradient(135deg,{a},{b})"></div>'
        f'<div class="dest-body"><h4>{name}</h4><p>{blurb}</p></div></div>'
        for name, blurb, a, b in DESTINATIONS
    )
    html(f'<div class="dest-grid">{cards}</div>')

    html('<div id="services"></div><div class="section-title">Your team of specialists</div>')
    html('<div class="section-sub">Every message is routed to the agent best suited to answer it.</div>')
    svc = "".join(
        f'<div class="svc-card"><h4>{ROUTE_LABELS[key]}</h4><p>{desc}</p></div>'
        for key, desc in ROUTE_DESCRIPTIONS.items()
    )
    html(f'<div class="svc-grid">{svc}</div>')

    html('<div id="how-it-works"></div><div class="section-title">How it works</div>')
    html('<div class="section-sub">Four steps, and you stay in control throughout.</div>')
    html(
        """
        <div class="step-grid">
          <div class="step"><div class="num">1</div><h4>Ask</h4><p>Describe your trip or question in plain language.</p></div>
          <div class="step"><div class="num">2</div><h4>Route</h4><p>Ra7al picks the right specialist for the job.</p></div>
          <div class="step"><div class="num">3</div><h4>Review</h4><p>Check the itinerary and costs, then approve or reject.</p></div>
          <div class="step"><div class="num">4</div><h4>Book</h4><p>Bookings and calendar entries happen only after approval.</p></div>
        </div>
        """
    )
    render_footer()


def render_login():
    _, middle, _ = st.columns([1, 1.35, 1])
    with middle:
        with st.container(key="authcard"):
            tab_signin, tab_signup = st.tabs(["Sign In", "Sign Up (Register)"])

            with tab_signin:
                html('<div class="auth-card-title">Welcome back</div>')
                html('<div class="auth-card-sub">Sign in to continue planning your Egypt trip.</div>')
                with st.form("signin_form", border=False):
                    username = st.text_input("Username", placeholder="e.g. user_101", key="login_username")
                    password = st.text_input("Password", type="password", key="login_password")
                    submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
                if submitted:
                    service = get_auth_service()
                    success, message, user, token = service.login(username, password)
                    if success and user:
                        st.session_state.auth_user = {
                            "user_id": user.user_id,
                            "name": user.name,
                            "token": token,
                        }
                        st.session_state.show_login = False
                        st.session_state.auth_error = None
                        st.session_state.session_id = str(uuid.uuid4())
                        st.rerun()
                    else:
                        st.error(message)
                html(f'<div class="demo-hint"><b>Demo accounts:</b> test_user, user_101, user_202. Password: <code>{DEMO_PASSWORD}</code></div>')

            with tab_signup:
                html('<div class="auth-card-title">Create Account</div>')
                html('<div class="auth-card-sub">Join Ra7al for a personalized Egypt travel journey.</div>')
                with st.form("signup_form", border=False):
                    reg_name = st.text_input("Full Name", placeholder="e.g. Fatima Noor", key="reg_name")
                    reg_username = st.text_input("Username", placeholder="e.g. fatima_travels", key="reg_username")
                    reg_nationality = st.text_input("Nationality", value="Egyptian", key="reg_nat")
                    reg_password = st.text_input("Password (min 6 characters)", type="password", key="reg_pwd")
                    reg_confirm = st.text_input("Confirm Password", type="password", key="reg_confirm")
                    reg_submit = st.form_submit_button("Create Account", type="primary", use_container_width=True)
                if reg_submit:
                    if reg_password != reg_confirm:
                        st.error("Passwords do not match.")
                    else:
                        service = get_auth_service()
                        success, message, user = service.register(
                            username=reg_username,
                            password=reg_password,
                            name=reg_name,
                            nationality=reg_nationality,
                        )
                        if success and user:
                            # Automatically log in upon successful sign-up
                            _, _, logged_user, token = service.login(reg_username, reg_password)
                            st.session_state.auth_user = {
                                "user_id": logged_user.user_id,
                                "name": logged_user.name,
                                "token": token,
                            }
                            st.session_state.show_login = False
                            st.session_state.auth_error = None
                            st.session_state.session_id = str(uuid.uuid4())
                            st.rerun()
                        else:
                            st.error(message)

            st.button("Back to home", on_click=close_login, key="login_back")


# ----------------------------------------------------------------------------
# Signed-in views
# ----------------------------------------------------------------------------

def render_sidebar(user):
    with st.sidebar:
        html(
            f"""
            <div class="side-card">
              <div class="avatar">{initials(user['name'])}</div>
              <div><b>{user['name']}</b><span>Traveler profile: {user['user_id']}</span></div>
            </div>
            """
        )
        html('<div class="side-h">Quick starts</div>')
        for label, prompt_text in SUGGESTIONS:
            st.button(label, key=f"side_{label}", on_click=queue_prompt, args=(prompt_text,), use_container_width=True)
        html('<div class="side-h">Specialists</div>')
        for key, desc in ROUTE_DESCRIPTIONS.items():
            html(f'<div class="route-item"><b>{ROUTE_LABELS[key]}</b><span>{desc}</span></div>')
        st.divider()
        st.button("Clear conversation", on_click=clear_conversation, use_container_width=True, key="side_clear")


def render_approval_panel():
    pending = st.session_state.pending_approval
    if not pending:
        return
    # TODO (Problem 7): Human-in-the-loop approval workflow disconnects:
    # 1. Resuming planner_app with Command(resume=...) requires a shared checkpointer thread across supervisor and planner.
    # 2. Approval returns next_agent: "execution_pending", which is not automatically forwarded to execution_app.
    # 3. Rejection terminates flow without prompting the user for desired changes or rerouting to feedback subgraph.
    # 4. Ensure approval is pinned to a specific plan_id / version so stale approvals do not apply after plan edits.
    with st.container(border=True, key="approval"):
        html('<div class="eyebrow" style="color:#A8841A">Action required</div>')
        st.markdown("## Review your itinerary")
        st.markdown(format_itinerary(pending["plan"]))
        st.caption("Nothing is booked or charged unless you approve.")
        col_approve, col_reject, _ = st.columns([1.2, 1.2, 4])
        with col_approve:
            approve_clicked = st.button("Approve itinerary", type="primary", use_container_width=True, key="approve_btn")
        with col_reject:
            reject_clicked = st.button("Reject itinerary", use_container_width=True, key="reject_btn")

    if approve_clicked or reject_clicked:
        resume_config = {"configurable": {"thread_id": pending["thread_id"]}}
        payload = pending.get("interrupt_payload") or {}
        plan_obj = pending.get("plan") or {}
        expected_plan_id = payload.get("plan_id") or plan_obj.get("plan_id")
        expected_plan_version = payload.get("plan_version") or plan_obj.get("plan_version")

        resume_value = {
            "approved": bool(approve_clicked),
            "plan_id": expected_plan_id,
            "plan_version": expected_plan_version,
        }
        resumed = planner_app.invoke(
            Command(resume=resume_value),
            config=resume_config,
        )

        resumed_plan = resumed.get("current_plan")
        resumed_plan_id = (
            getattr(resumed_plan, "plan_id", None)
            if resumed_plan is not None and not isinstance(resumed_plan, dict)
            else (resumed_plan or {}).get("plan_id") if isinstance(resumed_plan, dict) else None
        ) or expected_plan_id

        if approve_clicked and expected_plan_id and resumed_plan_id != expected_plan_id:
            response = (
                "Stale approval detected: the itinerary changed before confirmation. "
                "Please review the updated plan."
            )
            route_used = "planner"
        elif approve_clicked:
            approved_plan = resumed_plan or plan_obj
            exec_thread_id = f"execution-{st.session_state.get('session_id', 'streamlit-default')}"
            exec_result = execution_app.invoke(
                {
                    "approval_status": "approved",
                    "current_plan": approved_plan,
                    "user_id": (st.session_state.get("auth_user") or {}).get("user_id"),
                },
                config={"configurable": {"thread_id": exec_thread_id}},
            )
            response = (
                exec_result.get("response")
                or "Itinerary approved and booking/calendar execution completed."
            )
            route_used = "execution"
        else:
            response = (
                "Itinerary rejected. No bookings or charges were made. "
                "Tell me what you'd like to adjust and I can update or replan it."
            )
            route_used = "feedback"

        st.session_state.pending_approval = None
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "route": route_used,
        })
        st.rerun()
        

def render_history(user):
    for message in st.session_state.messages:
        role = message["role"]
        with st.chat_message(role):
            st.markdown(message["content"])
            if role == "assistant" and message.get("route"):
                route_badge(message["route"], message.get("confidence"))


def render_empty_state(user):
    first_name = user["name"].split()[0]
    html(
        f"""
        <div class="welcome">
          <h2>Marhaba, {first_name}</h2>
          <p>Ask about itineraries, visas, emergencies or bookings, or start with one of these.</p>
        </div>
        """
    )
    with st.container(key="suggestions"):
        cols = st.columns(2)
        for i, (label, prompt_text) in enumerate(SUGGESTIONS):
            with cols[i % 2]:
                st.button(
                    f"**{label}**\n\n{prompt_text}",
                    key=f"empty_{i}",
                    on_click=queue_prompt,
                    args=(prompt_text,),
                    use_container_width=True,
                )


def handle_prompt(prompt, user):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    recent_messages = []
    for message in st.session_state.messages[-3:]:
        role = "assistant" if message["role"] == "assistant" else "user"
        recent_messages.append((role, message["content"]))

    state = {
        "user_id": user["user_id"],
        "session_id": st.session_state.session_id,
        "messages": recent_messages,
        "curr_location": st.session_state.get("curr_location"),
        "current_intent": "unclassified",
    }

    should_rerun_for_approval = False
    with st.chat_message("assistant"):
        route, confidence, final_state = "profiling", None, {}
        try:
            session_id = st.session_state.get("session_id", "default")
            # Use a per-message thread_id so completed_steps from previous turns don't
            # block re-routing to the same subgraph (e.g. two consecutive tourist_help queries).
            turn_index = len(st.session_state.get("messages", []))
            supervisor_config = {"configurable": {"thread_id": f"supervisor-{session_id}-{turn_index}"}}
            with st.spinner("Ra7al is identifying the best specialist..."):
                final_state = get_supervisor().invoke(state, config=supervisor_config)
            # When the supervisor finishes with FINISH/empty target, infer the actual
            # specialist from the last completed step so the route badge is meaningful.
            raw_route = final_state.get("target_subgraph") or final_state.get("next_agent", "profiling")
            completed = final_state.get("completed_steps") or []
            route = raw_route if raw_route not in ("", "FINISH") else (completed[-1] if completed else "profiling")
            confidence = final_state.get("confidence")

            # Check if the planner subgraph raised a HITL interrupt inside the supervisor graph
            supervisor_interrupt = get_interrupt_payload(final_state)
            if supervisor_interrupt and isinstance(supervisor_interrupt, dict) and supervisor_interrupt.get("type") == "plan_approval":
                plan = supervisor_interrupt.get("plan")
                plan_id = supervisor_interrupt.get("plan_id")
                supervisor_thread_id = supervisor_config["configurable"]["thread_id"]
                st.session_state.pending_approval = {
                    "plan": plan,
                    "thread_id": supervisor_thread_id,
                    "interrupt_payload": supervisor_interrupt,
                    "use_supervisor": True,  # resume via supervisor graph, not planner directly
                }
                plan_dict = plan if isinstance(plan, dict) else (plan.model_dump() if hasattr(plan, "model_dump") else {})
                specialist_response = {
                    "text": format_itinerary(plan_dict),
                    "approval_required": True,
                    "plan": plan_dict,
                    "thread_id": supervisor_thread_id,
                    "interrupt_payload": supervisor_interrupt,
                }
            else:
                specialist_response = final_state.get("response")
                # Only call dispatch_specialist if supervisor returned nothing (None, not empty string)
                if specialist_response is None:
                    specialist_response = dispatch_specialist(route, prompt, state)
        except Exception as _exc:
            import traceback, sys
            tb = traceback.format_exc()
            logger.exception("Request failed")
            print(tb, file=sys.stderr, flush=True)
            specialist_response = f"Sorry, something went wrong while handling that request. Please try again.\n\n**Debug info (remove before release):**\n```\n{tb[-2000:]}\n```"

        if isinstance(specialist_response, dict):
            response = specialist_response.get("text", "")
            if specialist_response.get("approval_required"):
                st.session_state.pending_approval = {
                    "plan": specialist_response["plan"],
                    "thread_id": specialist_response["thread_id"],
                    "interrupt_payload": specialist_response.get("interrupt_payload"),
                }
                should_rerun_for_approval = True
        else:
            response = specialist_response
        response = response or final_state.get("response") or (
            f"Your request was routed to {ROUTE_LABELS.get(route, route)}."
        )

        st.markdown(response)
        route_badge(route, confidence)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "route": route,
        "confidence": confidence,
    })
    st.session_state.route_history.append(route)
    if should_rerun_for_approval:
        st.rerun()



def render_workspace(user):
    typed = st.chat_input("Ask Ra7al about planning, visas, emergencies, or bookings...")
    prompt = typed or st.session_state.pop("queued_prompt", None)

    render_approval_panel()
    if not st.session_state.messages and not prompt:
        render_empty_state(user)
    render_history(user)
    if prompt:
        handle_prompt(prompt, user)


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

init_state()
current_user = st.session_state.auth_user
inject_css(signed_in=bool(current_user))
render_navbar(current_user)

if current_user:
    render_sidebar(current_user)
    render_workspace(current_user)
elif st.session_state.show_login:
    render_login()
else:
    render_landing()