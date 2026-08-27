# Orbit — adaptive learning navigator

Orbit turns a stated goal into a prerequisite-locked route of real learning
resources, fitted to the hours the learner actually has — and explains every
recommendation it makes, including the ones it did not make.

There is no course catalogue in this repository. For each goal, Orbit derives
the skill graph behind it, searches the live web for resources, and plans a
route through them with a deterministic engine.

## Why it is built this way

A fixed catalogue caps the system at the roles somebody typed out in advance.
Ask such a system for "become a UX researcher" or "build a Rust game engine"
and it quietly answers about the nearest role it knows — a lookup table wearing
a recommender's clothes.

So the split is deliberate:

- **The model does what a fixed list structurally cannot** — decompose an
  arbitrary goal into its own skill graph, and read live search results into
  typed resources.
- **The algorithms decide the route** — gap weighting, budgeted set cover,
  topological ordering. These are deterministic and auditable, which is what
  makes a route defensible rather than "the model said so".

Two rules hold throughout. Nothing is fabricated: a resource is accepted only
if its URL came back from the search, durations are the source's own or are
flagged as estimates, and there are no invented ratings or enrollment counts.
And when a route cannot be produced honestly, the system says so instead of
falling back to seed data.

## How a request flows

```
goal in free text
  │
  ├─ intake        goal + constraints (hours, weeks, budget, format), domain-blind
  ├─ goal graph    Gemini → skills, target levels, importance weights, prereq DAG
  ├─ discovery     Tavily search per skill group, concurrently → typed resources
  ├─ catalogue     project the skill DAG onto the resources' prerequisites
  │
  ├─ profiler      mastery vector from claims, history and diagnostic answers
  ├─ gap           importance-weighted shortfall per skill
  ├─ retrieval     score resources against the gap, emitting reason codes
  ├─ planner       budgeted greedy set cover → topological sort → phases
  └─ explainer     prose written only from the reason codes the scorer emitted
```

The planner's guarantees are asserted in the test suite, not assumed:

- **Zero prerequisite violations.** The route is replayed and mastery is
  checked before every item.
- **The hour budget is never exceeded**, across several budgets.
- **More time never lowers projected readiness.**
- **Rejected and completed resources do not come back.**

## Grounding

Every scored term emits a reason code as it is computed — which gap it closes,
what it assumes, how its level matched, what it costs in hours. The explanation
layer is handed those codes and may write only from them, so an explanation is
a rendering of the arithmetic rather than a plausible story about it. The same
contract runs in reverse for "why is this *not* on my route", which is answered
from the prerequisite graph and the hour budget.

## Run locally

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env    # then paste your keys into .env
uvicorn app.main:app --reload --port 8000
```

Open <http://localhost:8000>. The UI is one self-contained file with no build
step: `backend/app/static/index.html`.

To see a route built from the command line:

```powershell
python scripts/demo.py --goal "Become a machine learning engineer" --hours 8 --weeks 20
```

## Configuration

| Variable | Required | Purpose |
| --- | --- | --- |
| `GEMINI_API_KEY` | yes | Skill-graph derivation, resource labelling, explanations |
| `TAVILY_API_KEY` | yes | Live web search for resources |
| `LPR_MODEL` | no | Primary Gemini model (default `gemini-3.6-flash`) |
| `LPR_OFFLINE` | no | Never call the API; cached goals still resolve |
| `LPR_USE_EMBEDDINGS` | no | Set false to force the TF-IDF retrieval fallback |
| `LPR_CACHE_DIR` | no | Where derived graphs and resources are cached |

There are no credentials in this repository and no defaults in source — the
application reads both keys from the environment and fails honestly without
them. Copy `backend/.env.example` to `backend/.env` for local work; `.env` is
gitignored.

## Deployment

Vercel serves `public/index.html` at the root and routes `/api/*` to the
FastAPI app through `api/index.py`. Both keys are set as project environment
variables.

One limitation worth stating plainly: serverless functions only have ephemeral
`/tmp`, and any request can be answered by an instance that has never seen a
given learner. The browser therefore holds the authoritative profile and sends
it with each request, guarded by a revision counter, and a workspace loads in a
single call rather than six. That keeps a session coherent, but a genuinely
multi-user deployment wants durable shared storage behind `store.py` and the
discovery caches.

## API

| Endpoint | Purpose |
| --- | --- |
| `POST /api/intake` | Capture a goal and constraints from free text |
| `GET /api/workspace/{id}` | Profile, route, catalogue and dashboard in one call |
| `POST /api/path/{id}/generate` | Derive the graph, discover resources, plan the route |
| `GET /api/path/{id}/explain/{resource_id}` | Explanation plus the reason codes behind it |
| `GET /api/path/{id}/why-not/{resource_id}` | Counterfactual from the graph and the budget |
| `POST /api/path/{id}/feedback` | Apply a signal and re-plan |
| `POST /api/path/{id}/simulate` | What-if over hours and weeks, nothing persisted |
| `GET/POST /api/profile/{id}/diagnostic` | Adaptive diagnostic and grading |
| `GET /api/dashboard/{id}` | Readiness, gaps, progress, next actions |
| `POST /api/chat` | Grounded assistant, streamed |

## Tests

```powershell
cd backend
pytest -q
```

54 tests, all offline — fixtures stand in for the model and search boundary, so
the suite needs no key and does not depend on what the live web returns today.
