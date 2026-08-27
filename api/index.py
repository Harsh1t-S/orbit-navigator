"""Vercel entrypoint for the FastAPI application.

Vercel discovers functions from the repository-root ``api/`` directory. The
application itself stays in ``backend/app`` so local development continues to
use ``uvicorn app.main:app`` from the backend directory.
"""

from backend.app.main import app
