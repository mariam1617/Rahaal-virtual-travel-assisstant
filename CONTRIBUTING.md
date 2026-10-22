# Contributing to Virtual Travel Assistant
## 1. Getting Started

### Prerequisites

* **Python 3.10+**
* **Git**

### Local Environment Setup

1. **Fork and clone the repository:**
```bash
git clone https://github.com/your-username/virtual-travel-assistant.git
cd virtual-travel-assistant

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


3. **Set up environment variables:**
```bash
cp .env.example .env

```


Fill in required API keys (e.g., OpenAI/Anthropic/Google LLM keys, Map API keys, Vector DB connections) in your `.env` file.

---

## 2. Project Architecture & Structure

This project uses a **LangGraph multi-agent architecture** structured as follows:

```text
src/
├── agents/             # LangGraph subgraphs (Supervisor, Tourist Help, Planner, Execution, Feedback)
├── api/                # FastAPI endpoints and schemas
├── db/                 # Vector stores and profile DB interfaces
├── state/              # Shared graph state and memory schemas
├── tools/              # External API tools (Maps, Weather, Booking, Calendar)
├── config.py           # Application configurations
└── main.py             # Application entry point

```

When creating new features, make sure agent nodes communicate exclusively through the shared state defined in `src/state/overall_state.py`.

---

## 3. Git Workflow & Branch Naming

### Branch Naming Convention

Use descriptive branch names prefixed with the relevant epic or task type:

* **Feature branches:** `feature/epic-X-short-description` (e.g., `feature/epic-1-load-context`)
* **Bug fixes:** `fix/issue-description` (e.g., `fix/planner-temporal-loop`)
* **Documentation:** `docs/update-readme`

### Commit Messages

Keep commit messages concise and imperative:

* `feat(planner): add constraint validation logic`
* `fix(execution): handle 3x booking retry limit gracefully`
* `docs(readme): update environment variable setup`

---

## 4. Coding Standards & Testing

### Python Guidelines

* Follow **PEP 8** style guidelines.
* Use **Type Hints** for all function signatures and agent node definitions.
* Ensure all agent tools handle missing parameters gracefully by returning clear status messages to the calling node.

### Code Formatting

Format your code before committing:

```bash
black src/ tests/
ruff check src/ tests/

```

### Running Tests

All pull requests must pass existing unit and integration tests.

Run the test suite with `pytest`:

```bash
pytest

```

To run tests for a specific subgraph:

```bash
pytest tests/test_planner.py

```

---

## 5. Pull Request (PR) Checklist

Before opening a Pull Request, verify the following:

* [ ] Branch is up to date with `main`.
* [ ] Code follows project formatting and type standards.
* [ ] New agent nodes or tools include corresponding unit tests in `tests/`.
* [ ] All tests pass locally using `pytest`.
* [ ] Shared state schema changes (`src/state/`) are documented and backward-compatible.
* [ ] Description clearly explains **what** was changed and **which Epic/Story** it satisfies.