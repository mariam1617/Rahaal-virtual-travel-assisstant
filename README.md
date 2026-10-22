# Rahaal-virtual-travel-assisstant

## Introduction
The AI Travel Assistant is an intelligent, multi-agent travel routing and planning system designed to handle real-time tourist inquiries, generate complex itineraries, execute bookings, and learn from user preferences. Built on a modular "Subgraph" architecture (ideal for frameworks like LangGraph), the system intelligently routes user intents while managing both short-term session states and long-term memory embeddings. 

The project is divided into five core epics:
1. **Core Routing & Context Management:** Loading memory and routing requests.
2. **Tourist Help Subgraph:** Providing immediate assistance, emergency info, and local RAG-based rules.
3. **Planner Subgraph:** Generating and validating complex itineraries against budgets, time constraints, and accessibility.
4. **Execution Subgraph:** Handling Human-In-The-Loop (HITL) bookings and calendar syncing.
5. **Feedback & Memory Subgraph:** Updating active plans and extracting long-term preference insights.

---

## System Architecture

Based on the provided Epics and the system flowchart, the architecture is state-driven and highly modular. 

### 1. State Management (Memory)
The system relies on a hydrated context object before any routing occurs.
*   **Long-Term Memory:** 
    *   `G Pref` (Global Preferences): User tastes, budgets, and historical feedback.
    *   `P Plan` (Generated Plans): Past and upcoming confirmed itineraries.
*   **Short-Term Memory (Session):**
    *   `T current_location`: Parsed GPS/Location data.
    *   `T Tourist_intent`: The parsed immediate goal of the user.

### 2. Core Routing (Supervisor)
The **Supervisor Agent** acts as the main entry point. It evaluates the hydrated state and user prompt to route the flow to one of four specific domain subgraphs, or to a Fallback/Profiling agent if the intent is unclear.

### 3. Subgraphs (Agentic Workflows)
*   **Tourist Help Subgraph:** Uses an OpenMap API to resolve location, branches into Emergency RAG databases, Visa/Transport rule checks, or contextual nearby place recommendations.
*   **Planner Subgraph:** A cyclical validation loop. A Draft is built, then checked against constraints (budget/logistics). If it fails, it loops to a Repair node. It then passes to Temporal/Accessibility validation, and finally an overall Plan Evaluation.
*   **Execution Subgraph:** Requires explicit Human-in-the-Loop (HITL) approval. Once approved, it uses a Mock Booking API with a 3-try retry mechanism, finalizing by pushing to a Calendar API.
*   **Feedback Subgraph:** Extracts insights from chats (`chat -> insight -> store`) to update `G Pref`, or handles active modifications to existing plans by routing them back to the Planner Subgraph.

---

## Proposed GitHub Repository File System

To support this multi-agent, modular architecture, the repository should be organized by domains and subgraphs. Here is a recommended structure for a Python-based backend (e.g., using LangChain/LangGraph):

```text
ai-travel-assistant/
│
├── docs/                       # Documentation, architecture diagrams, and epics
│   └── architecture.md
│
├── data/                       # Vector databases, mock JSON data, embeddings
│   ├── emergency_db/
│   └── tourist_rag/
│
├── src/                        # Main application source code
│   ├── __init__.py
│   │
│   ├── state/                  # State management and memory classes
│   │   ├── __init__.py
│   │   ├── context.py          # Context hydration logic (Story 1.1)
│   │   ├── long_term.py        # G Pref and P Plan schemas/DB connections
│   │   └── short_term.py       # Session variables
│   │
│   ├── agents/                 # Individual LLM agent definitions
│   │   ├── __init__.py
│   │   ├── supervisor.py       # Supervisor Router (Story 1.2)
│   │   ├── profiling.py        # Interview Agent (Story 5.1)
│   │   └── evaluator.py        # Final Plan Evaluation (Story 3.4)
│   │
│   ├── subgraphs/              # Graph logic for each domain
│   │   ├── __init__.py
│   │   ├── tourist_help.py     # Intent parsing, RAG, Emergency (Epic 2)
│   │   ├── planner.py          # Draft, Validation, Repair loops (Epic 3)
│   │   ├── execution.py        # HITL, Booking API, Calendar (Epic 4)
│   │   └── feedback.py         # Memory updates, insight extraction (Epic 5)
│   │
│   ├── tools/                  # Tools called by agents
│   │   ├── __init__.py
│   │   ├── geo_tools.py        # OpenMap API, find_nearby_places
│   │   ├── rag_tools.py        # check_equivisa, get_country_advisors
│   │   ├── plan_tools.py       # calculate_travel_time, weather_forecast
│   │   └── booking_tools.py    # search_booking, calendar_sync
│   │
│   └── utils/                  # Helper functions, config, logging
│       ├── config.py
│       └── logger.py
│
├── tests/                      # Unit and integration tests
│   ├── test_routing.py
│   ├── test_planner.py
│   └── test_execution.py
│
├── .env.example                # Environment variables template
├── requirements.txt            # Python dependencies (LangChain, Pydantic, etc.)
├── .gitignore
└── README.md                   # Project intro and setup instructions
