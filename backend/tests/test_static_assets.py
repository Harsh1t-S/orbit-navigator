"""The UI ships twice and both copies must stay identical.

`backend/app/static/index.html` is what uvicorn serves locally; `public/`
is what Vercel serves at the root. They were the same file by hand, which is
exactly the kind of duplication that silently drifts - a fix applied to one and
not the other looks like "it works locally".
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SERVED = REPO / "backend" / "app" / "static" / "index.html"
PUBLIC = REPO / "public" / "index.html"


def test_the_two_copies_of_the_ui_are_identical():
    assert SERVED.exists() and PUBLIC.exists()
    served, public = SERVED.read_bytes(), PUBLIC.read_bytes()
    assert served == public, (
        "backend/app/static/index.html and public/index.html have drifted - "
        "copy the canonical backend/app/static/index.html over public/index.html."
    )
