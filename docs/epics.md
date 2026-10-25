# Epic 1: Core Routing & Context Management

**Goal:** Establish the foundational system state and ensure the Supervisor Router accurately directs user inquiries.

## Story 1.1: Load User Context

**As a** system router,
**I want to** load the user's short-term and long-term memory state before processing the request
**So that** all downstream agents have the necessary ID, location, and preference data to personalize their actions.

**Acceptance Criteria:**
- System retrieves G Pref (Global Preferences) and P Plan (Generated Plans) from long-term storage.
- System loads session variables: T current location and T Tourist intent.
- Context object is fully hydrated before passing to the Supervisor Router.

## Story 1.2: Supervisor Request Routing

**As a** user,
**I want to** my initial request evaluated by an LLM
**So that** I am automatically routed to the correct domain (Tourist Help, Plan Trip, Existing Trip Change, or Human Approval).

**Acceptance Criteria:**
- Supervisor Agent successfully uses the read_mem_state and route_to_agent tools.
- Inquiries are mapped correctly to one of the 4 primary subgraphs.
- Unrecognized intents trigger a fallback request for clarification via the Profiling & Interview Agent.

---

# Epic 2: Tourist Help Subgraph (Immediate Assistance)

**Goal:** Handle real-time inquiries, emergencies, and localized recommendations.

## Story 2.1: Intent & Location Resolution

**As a** traveler,
**I want to** the system to understand my query and resolve my GPS coordinates
**So that** assistance and recommendations are geographically accurate.

**Acceptance Criteria:**
- OpenMap API successfully converts raw coordinates to curr location ST.
- TouristRouter successfully branches to EmergencyDB, TouristRAG, or Places Recommendation based on the parsed intent.

## Story 2.2: Emergency Assistance

**As a** traveler in distress,
**I want to** immediate access to emergency numbers and protocols
**So that** I can get help quickly based on my current location.

**Acceptance Criteria:**
- Help Agent successfully calls get_emergency_number(location).
- RAG pipeline queries the EmergencyDB vector database.
- HelpVerification LLM step validates the output for accuracy before displaying it to the user.

## Story 2.3: General Tourist Advisories & Rules

**As a** traveler planning a route,
**I want to** query visa rules, transit options, and country advisories
**So that** I can travel safely and legally.

**Acceptance Criteria:**
- Help Agent successfully calls check_equivisa, get_country_advisors, and transport_rules.
- Responses are generated via the TouristRAG vector database and validated by the HelpVerification step.

## Story 2.4: Contextual Place Recommendations

**As a** tourist looking for activities,
**I want to** suggestions for nearby places that match my historical tastes
**So that** I can discover relevant local attractions.

**Acceptance Criteria:**
- Recommendation Agent successfully executes find_nearby_places(gps/coordinates).
- Results are filtered using match_user_taste(place, taste) against the user's loaded G Pref.

---

# Epic 3: Planner Subgraph (Itinerary Generation)

**Goal:** Generate, validate, and repair complex travel itineraries using multi-step constraints.

## Story 3.1: Draft Itinerary Generation

**As a** user,
**I want to** the system to generate a draft trip plan based on my preferences and constraints
**So that** I have a foundational itinerary to review.

**Acceptance Criteria:**
- Planner Agent utilizes search_local_tool, calculate_travel_time, and out_weather_forecast.
- Build_draft tool successfully generates a list of candidate places and activities.

## Story 3.2: Constraint & Budget Validation

**As a** planner agent,
**I want to** validate the draft against user budgets and logistical rules
**So that** the proposed plan is financially and physically feasible.

**Acceptance Criteria:**
- Drafts exceeding budget or violating logistics route to the ConstraintRepair node.
- Repaired drafts loop back for re-validation; valid drafts proceed to Temporal Validation.

## Story 3.3: Temporal & Accessibility Validation

**As a** traveler with specific scheduling and physical needs,
**I want to** my itinerary checked for time conflicts and accessibility standards
**So that** my trip is stress-free and accommodating.

**Acceptance Criteria:**
- Time conflicts route to TemporalRepair.
- Accessibility failures (verified via external API) route to AccessibilityRepair.
- Fully valid plans proceed to the PlanEvaluation node.

## Story 3.4: Final Plan Evaluation

**As a** system,
**I want to** evaluate the overall quality of the repaired itinerary
**So that** low-quality plans are rejected before the user sees them.

**Acceptance Criteria:**
- Plans evaluated as "Poor" trigger a Fallback response.
- Plans evaluated as "Good" output an executable Plan object ready for the Execution Subgraph.

---

# Epic 4: Execution Subgraph (Booking)

**Goal:** Handle external API bookings with mandatory Human-In-The-Loop (HITL) oversight.

## Story 4.1: Human-in-the-Loop Approval

**As a** user,
**I want to** review and explicitly approve an executable itinerary
**So that** no bookings or charges are made without my consent.

**Acceptance Criteria:**
- Supervisor Router directs to HumanApproval.
- If Rejected, the Execution subgraph exits gracefully.
- If Approved, the state moves to Hold ➔ Confirm.

## Story 4.2: Booking Confirmation & Retries

**As an** Execution Agent,
**I want to** execute the booking via external APIs with a retry mechanism
**So that** transient network errors don't fail the reservation process.

**Acceptance Criteria:**
- Agent utilizes the SearchBooking node to interface with the Mock Booking API.
- On failure, the system loops back to SearchBooking up to a hard limit of 3 retries.
- On 3 consecutive failures, user is notified.

## Story 4.3: Calendar Syncing

**As a** user,
**I want to** my successfully booked events pushed to my calendar
**So that** my personal schedule stays up to date automatically.

**Acceptance Criteria:**
- On booking success, system triggers the Event/Calendar API.
- Execution Subgraph reaches the EndExecution state.

---

# Epic 5: Feedback & Memory Subgraph

**Goal:** Process user changes, adjust active plans, and update long-term embeddings.

## Story 5.1: Session-Based Profiling

**As a** system,
**I want to** ask clarifying questions when data is missing
**So that** I can confidently proceed with planning or helping.

**Acceptance Criteria:**
- Profiling & Interview Agent triggers ask_user_questions.
- User responses update short-term memory via update_session_preferences.

## Story 5.2: Active Itinerary Modification

**As a** user,
**I want to** request changes to my existing trip plan
**So that** the itinerary adapts to my changing desires.

**Acceptance Criteria:**
- FeedbackRouter identifies "Planning" feedback.
- System triggers PlannerUpdate and loops data back to the Planner Subgraph for constraint and temporal re-validation.

## Story 5.3: Long-Term Preference Learning

**As a** user,
**I want to** the system to remember my feedback on places or styles
**So that** future trips reflect my evolving tastes.

**Acceptance Criteria:**
- Feedback Handler tracks chat insights (chat ➔ insight ➔ store).
- FeedbackRouter identifies "Preference" feedback.
- System triggers PreferenceUpdate and persistently saves the insight to the G Pref profile store.

  # Project Implementation Plan

| Week | Phase Goal | Task | Story Reference |
|---|---|---|---|
| Week 1 | **Architecture Foundation, State & Knowledge Bases**<br>Establish the foundational system state, primary routing logic, and manual document creation. | **Set up Backend Infrastructure:** Initialize the Python and FastAPI environment to support synchronous execution across the multi-agent system. | Project Initialization |
| | | **Manually Create Knowledge Base Documents:** Draft and compile the static text files, PDFs, or CSVs containing emergency numbers, visa rules, transit options, and country advisories for the RAG pipelines. | Story 2.2, Story 2.3 |
| | | **Initialize System State Object:** Define the core state schema to hold essential variables (Global Preferences, Generated Plans, user's current location). | Story 1.1 - Load User Context |
| | | **Implement Memory Loaders:** Create connectors to fetch the user's long-term memory profile and short-term session variables. | Story 1.1 - Load User Context |
| | | **Develop Supervisor Router LLM:** Engineer the Supervisor Agent to evaluate initial requests and map inquiries to the primary subgraphs. | Story 1.2 - Supervisor Request Routing |
| | | **Build Fallback Flow:** Connect the Supervisor to the Profiling & Interview Agent to handle unrecognized intents. (Implement as a temporary pass-through stub for now). | Story 1.2 - Supervisor Request Routing |
| Week 2 | **Read-Only Tourist Assistance & Recommendations**<br>Deliver immediate value through RAG pipelines, emergency assistance, and local recommendations. | **Implement Intent & Location Parsing:** Integrate OpenMap API to convert raw coordinates to a current location state and branch to the correct sub-agent. | Story 2.1 - Intent & Location Resolution |
| | | **Deploy EmergencyDB Pipeline:** Spin up the vector database and ingest the manually created emergency number documents for the Help Agent. | Story 2.2 - Emergency Assistance |
| | | **Build TouristRAG Document Store:** Ingest the manually created visa rules, transit options, and country advisories into the vector store. | Story 2.3 - General Tourist Advisories & Rules |
| | | **Implement Help Verification Node:** Create a verification LLM step to validate outputs against hallucinations before presenting them to the user. | Story 2.2, Story 2.3 |
| | | **Build Place Recommendation Logic:** Implement local search capabilities and filter recommendations against the user's loaded global preferences. | Story 2.4 - Contextual Place Recommendations |
| Week 3 | **Planner Subgraph Part 1 (Drafts & Logistics)**<br>Enable Planner Agent to generate base itineraries and loop through core financial and logistical constraints. | **Equip Planner with Base Tools:** Provide integrations to search locally, calculate travel times, and fetch weather forecasts. | Story 3.1 - Draft Itinerary Generation |
| | | **Implement Draft Generator:** Prompt the Planner to synthesize tool outputs into a structured candidate list of places and activities. | Story 3.1 - Draft Itinerary Generation |
| | | **Build Budget & Logistics Validator:** Create a validation node to calculate total costs and verify route feasibility. | Story 3.2 - Constraint & Budget Validation |
| | | **Develop Constraint Repair Loop:** Implement cyclic routing so drafts failing validation are sent back with explicit error context for correction. | Story 3.2 - Constraint & Budget Validation |
| | | **Develop Active Profiling Agent:** Build a tool to ask clarifying questions and gather missing constraints from the user during the planning phase. | Story 5.1 - Session-Based Profiling |
| Week 4 | **Planner Subgraph Part 2 (Advanced Validation)**<br>Resolve complex temporal conflicts, verify accessibility, and guarantee plan quality. | **Implement Temporal & Accessibility Checks:** Integrate APIs to verify opening hours, overlapping event times, and physical accessibility standards. | Story 3.3 - Temporal & Accessibility Validation |
| | | **Build Advanced Repair Nodes:** Create specific temporal and accessibility repair nodes to handle logistical failures without forcing a full draft regeneration. | Story 3.3 - Temporal & Accessibility Validation |
| | | **Implement Final Plan Evaluator:** Build a plan evaluation node to grade the repaired itinerary, routing high-quality plans to execution and rejecting poor ones. | Story 3.4 - Final Plan Evaluation |
| Week 5 | **Execution Subgraph (Booking & API Syncing)**<br>Transition approved plans into real-world reservations with strict human oversight. | **Build Human-in-the-Loop (HITL) Gate:** Implement an interrupt node pausing all execution until explicit user approval is received for the itinerary. | Story 4.1 - Human-in-the-Loop Approval |
| | | **Integrate Mock Booking API:** Equip the Execution Agent with tools to trigger external reservations. | Story 4.2 - Booking Confirmation & Retries |
| | | **Implement Booking Retry Mechanism:** Code a retry loop with a limit of 3 retries for transient network errors, paired with a user-notification fallback. | Story 4.2 - Booking Confirmation & Retries |
| | | **Calendar Sync Integration:** Connect successful bookings to automatically push events to the user's schedule. | Story 4.3 - Calendar Syncing |
| Week 6 | **Feedback Loops, Memory & System Testing**<br>Enable active memory updates, handle user modifications, and conduct comprehensive tests. | **Implement Feedback Router:** Build classification logic to distinguish between active trip planning feedback and long-term preference updates. | Story 5.2, Story 5.3 |
| | | **Develop Active Itinerary Modifier:** Create functionality to inject requested changes into the current plan and route it back through the validation loops. | Story 5.2 - Active Itinerary Modification |
| | | **Build Insight Extractor Pipeline:** Implement logic to distill conversational feedback into structured tags or embeddings. | Story 5.3 - Long-Term Preference Learning |
| | | **Implement Long-Term Preference Update:** Create the update function to save new insights persistently into the profile store for future trips. | Story 5.3 - Long-Term Preference Learning |
| | | **Final End-to-End System Tests:** Conduct comprehensive tests across the entire application, validating the synchronous agent execution and FastAPI endpoints. | All Stories |
