"""Print a path built from a per-goal graph and real discovered resources.

Requires GEMINI_API_KEY and TAVILY_API_KEY for an uncached goal. Matching goal/resource
caches can be reused without a key; this script never substitutes seed data.

    python scripts/demo.py --goal "Become a machine learning engineer" --hours 8 --weeks 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(BACKEND_DIR / ".env")

from app import session  # noqa: E402
from app.engines import intake, planner  # noqa: E402
from app.engines.gap import gap_items, readiness  # noqa: E402
from app.engines.profiler import mastery  # noqa: E402
from app.schemas import LearnerProfile  # noqa: E402


def _rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a dynamic Orbit path in the terminal.")
    parser.add_argument("--goal", required=True, help="The learner's goal, in their own words.")
    parser.add_argument("--hours", type=float, default=8, help="Hours available each week (default: 8).")
    parser.add_argument("--weeks", type=int, default=20, help="Planning horizon in weeks (default: 20).")
    parser.add_argument("--budget", choices=("free", "freemium", "paid"), default="freemium")
    parser.add_argument("--known", action="append", default=[], help="Known skill phrase; repeat as needed.")
    parser.add_argument("--refresh", action="store_true", help="Bypass graph/resource caches and search again.")
    args = parser.parse_args()

    profile = LearnerProfile(
        learner_id="terminal-demo",
        goal_text=args.goal,
        weekly_hours=max(args.hours, 1),
        horizon_weeks=max(args.weeks, 1),
        budget=args.budget,
        claimed_skills=args.known,
    )

    try:
        active = session.build_for(
            profile.learner_id, profile.goal_text, profile.budget, refresh=args.refresh
        )
    except session.GoalUnavailable as exc:
        print(f"Cannot build a path: {exc}", file=sys.stderr)
        print(
            "Add GEMINI_API_KEY and TAVILY_API_KEY to backend/.env, or use a goal with matching cached graph and resources.",
            file=sys.stderr,
        )
        return 2

    intake.snap(profile, active.spec)
    profile.target_role = active.role.id
    cat = active.catalog
    target, weight = cat.role_target(active.role)
    current = mastery(profile, cat)
    path = planner.build_path(profile, cat, active.retriever)

    _rule(f"{active.role.title}")
    print(f"goal: {profile.goal_text}")
    print(f"graph: {cat.n_skills} skills; resources: {len(cat.courses)}")
    print(f"readiness now: {readiness(current, target, weight):.1%}")

    _rule("Largest weighted gaps")
    for item in gap_items(current, target, weight, cat)[:6]:
        print(f"  {item.skill_name:<34} {item.current:.2f} -> {item.target:.2f}")

    _rule("Generated path")
    for milestone in path.milestones:
        print(f"  [{milestone.index + 1}] {milestone.title} (weeks {milestone.start_week}-{milestone.end_week})")
        for item in milestone.items:
            if item.kind == "course" and item.course:
                duration = f"{item.hours:g}h" if item.course.hours_stated else f"~{item.hours:g}h"
                print(f"      {item.title} — {duration} — {item.course.url}")
            else:
                print(f"      {item.title} — {item.hours:g}h")

    _rule("Summary")
    print(f"  {path.total_hours:g}h over {path.total_weeks} weeks")
    print(f"  readiness {path.readiness_before:.1%} -> {path.readiness_after:.1%}")
    print(f"  weighted gap closed: {path.coverage:.1%}")
    for note in path.notes:
        print(f"  note: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
