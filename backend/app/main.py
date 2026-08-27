import base64
import binascii
import logging
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

from . import llm, store  # noqa: E402  - after load_dotenv so the key is visible
from .api import chat, dashboard, path, profile, workspace  # noqa: E402
from .api.deps import goal_session  # noqa: E402
from .config import settings  # noqa: E402
from .engines import discovery  # noqa: E402
from .schemas import LearnerProfile  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(
    title="Learning Path Recommender",
    version="0.3.0",
    description="Skill-graph based learning path generation with grounded explanations.",
)

# Wide open in dev. Lock this down before anything is deployed publicly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(profile.router)
app.include_router(path.router)
app.include_router(dashboard.router)
app.include_router(chat.router)
app.include_router(workspace.router)


# The browser holds the authoritative learner profile and sends it back on
# every call. On a serverless host the instance answering this request may
# never have seen this learner - see store.adopt for why that happens and why
# adopting is safe.
PROFILE_HEADER = "x-orbit-profile"
MAX_PROFILE_HEADER = 8192   # base64; a profile this large is not a real one


@app.middleware("http")
async def adopt_client_profile(request, call_next):
    raw = request.headers.get(PROFILE_HEADER)
    if raw and len(raw) <= MAX_PROFILE_HEADER:
        try:
            store.adopt(LearnerProfile.model_validate_json(base64.b64decode(raw)))
        except (ValueError, binascii.Error, UnicodeDecodeError) as exc:
            # Untrusted input. A malformed header is not worth failing a
            # request over, but it should not pass silently either.
            log.warning("ignoring unusable %s header (%s)", PROFILE_HEADER, type(exc).__name__)
    return await call_next(request)


@app.get("/api/health")
def health():
    models = llm.configured_models()
    return {
        "status": "ok",
        "catalog_mode": "per_goal",
        "courses": 0,
        "skills": 0,
        "roles": [],
        "llm": "live" if llm.available() else "offline",
        "search": "live" if discovery.available() else "offline",
        "model": f"Gemini pool ({len(models['graph'])} models)",
        "primary_model": settings.model,
        "models": models,
    }


@app.get("/api/catalog/roles")
def roles(learner_id: str | None = None):
    if not learner_id:
        return []
    return workspace.catalog_payload(goal_session(learner_id))["roles"]


@app.get("/api/catalog/courses")
def courses(learner_id: str | None = None, q: str = "", limit: int = 50):
    if not learner_id:
        return []
    cat = goal_session(learner_id).catalog
    items = cat.courses
    if q:
        needle = q.lower()
        items = [c for c in items if needle in c.title.lower() or needle in c.description.lower()]
    return [c.model_dump() for c in items[:limit]]


@app.get("/api/catalog/skills")
def skills(learner_id: str | None = None):
    if not learner_id:
        return []
    return [s.model_dump() for s in goal_session(learner_id).catalog.skills]


# The UI is one self-contained file with no build step - no npm, no bundler,
# nothing to install before a judge can open it. Mounted last so it cannot
# shadow an /api route.
STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
