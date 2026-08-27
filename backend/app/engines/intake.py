"""Conversational intake: free text in, structured constraints out.

Extraction here is deliberately domain-blind. It does not know what skills
exist, because at this point nothing does - the skill graph is derived from the
goal afterwards (engines/goal.py). So intake captures three things:

  * the goal, in the learner's own words, passed through untouched
  * hard constraints - hours a week, weeks available, budget, format
  * whatever they claimed to already know, as raw phrases

Those phrases get snapped onto the derived graph later, in `snap`, once there
is a graph to snap them onto. Keeping the two steps apart is what lets the
system take a goal in any domain: the extractor never needs a vocabulary.

There is a keyword fallback for when no model is available. It cannot derive a
graph - nothing can, without a model - but it gets the constraints right, which
is enough to keep the intake conversation moving and to reuse a cached goal.
"""

from __future__ import annotations

import difflib
import re

from .. import llm
from ..schemas import LearnerProfile, ProfileDraft, SkillClaim

SYSTEM = """You extract a learner's goal and constraints from what they tell you.

You do not know what skills exist and you must not guess a curriculum - another
step handles that. Capture only what this person actually said.

  goal_text        their goal, rewritten as one clear sentence in their own
                   terms. Do not generalise a specific goal into a job title:
                   "build a Rust game engine" must not become "software
                   engineer". If they named a role, keep the role.
  known_skills     raw phrases for things they said they already know or use.
                   Their words, not a taxonomy. Include the qualifier they used
                   - "basic Python", "five years of SQL", "some React".
  weekly_hours     only if stated. Never infer it from enthusiasm.
  horizon_weeks    only if stated. Convert months to weeks.
  time_unconstrained true only when they explicitly say there is no deadline,
                   no time constraint, or that the timeline is open-ended.
  budget           free / freemium / paid, only if they indicated one.
  format_prefs     video / text / interactive / project, only if indicated.
  motivation       job_switch / promotion / curiosity / academic, if clear.

  confidence       0-1, how complete this is. Below 0.7 you must set
                   follow_up_question to the single most useful thing to ask
                   next: the goal first if it is vague, then current experience,
                   then time available. "Complete beginner", "no experience",
                   and "starting from scratch" answer the experience question.
                   An open-ended timeline answers the time-window question.
                   One question, conversational, and never one they have
                   already answered.

A goal you have never heard of is not a problem. Record it faithfully.
"""


def extract(message: str, history: list[str], existing_goal: str = "") -> ProfileDraft:
    transcript = "\n".join(history[-6:] + [f"learner: {message}"])
    parsed = llm.parse(ProfileDraft, transcript, SYSTEM, effort="low", lane="fast")
    if parsed is None:
        return _fallback(message, history, existing_goal=existing_goal)
    return _normalise(parsed, message, history, existing_goal=existing_goal)


# ---------------------------------------------------------------------------
# offline fallback - constraints only, no curriculum
# ---------------------------------------------------------------------------

WORD_NUMBERS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "couple": 2, "few": 3,
}

LEVEL_HINTS = [
    (re.compile(r"\b(expert|advanced|strong|years of|professional)\b"), 5),
    (re.compile(r"\b(solid|comfortable|confident|good at|experienced|at work|for work|daily)\b"), 4),
    (re.compile(r"\b(some|decent|okay|ok|working knowledge|intermediate)\b"), 3),
    (re.compile(r"\b(basic|beginner|little|new to|learning|starting|dabbled)\b"), 2),
]

# "I know X and want to Y" - everything after the intent verb is aspiration,
# not experience. Without this the goal gets mined as a held skill.
ASPIRATION = re.compile(
    r"\b(?:want(?:s|ed)? to|would like to|hoping to|aiming to|aspire to|plan to|"
    r"trying to|looking to|move into|get into|switch to|transition to|become|"
    r"learn|study|pick up|build)\b"
)

# what people put after "I know" / "I use" / "I'm comfortable with"
EXPERIENCE = re.compile(
    r"\b(?:i know|i use|i've used|i have used|i am|i'm|familiar with|comfortable with|"
    r"experienced (?:in|with)|background in|work with|write)\b([^.;]{2,90})"
)

OPEN_ENDED = re.compile(
    r"\b(?:no\s+(?:time\s+)?constraints?|no\s+(?:fixed\s+)?deadline|"
    r"no\s+time\s+limit|not\s+in\s+a\s+rush|as\s+long\s+as\s+it\s+takes|"
    r"open[ -]?ended|flexible\s+(?:timeline|schedule))\b"
)
NOVICE = re.compile(
    r"\b(?:(?:complete|completely|absolute|total)\s+beginner|"
    r"no\s+(?:prior\s+)?(?:(?:programming|coding|machine\s+learning|ml)\s+)?experience|"
    r"zero\s+(?:prior\s+)?experience|starting\s+(?:completely\s+)?from\s+scratch)\b"
)


def _learner_messages(history: list[str]) -> list[str]:
    """Return learner turns only.

    Stored history includes the assistant's questions. Feeding those questions
    to the regex fallback made "How many hours a week?" look like a learner's
    one-week constraint.
    """
    messages: list[str] = []
    for turn in history[-6:]:
        role, separator, content = turn.partition(":")
        if separator and role.strip().lower() == "assistant":
            continue
        if separator and role.strip().lower() in ("user", "learner"):
            messages.append(content.strip())
        else:
            messages.append(turn)
    return messages


def _has_duration(text: str) -> bool:
    return bool(re.search(r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|"
                          r"eleven|twelve|couple|few)\s+(?:weeks?|months?)\b", text.lower()))


def _follow_up_only(message: str) -> bool:
    lowered = message.lower()
    return bool(
        re.search(
            r"\b(?:hours?|hrs?|weeks?|months?|free|paid|budget|i know|i use|"
            r"familiar with|comfortable with|experienced|experience|beginner|"
            r"starting from scratch|time constraints?|deadline)\b",
            lowered,
        )
    )


def _normalise(
    draft: ProfileDraft,
    message: str,
    history: list[str],
    *,
    existing_goal: str = "",
) -> ProfileDraft:
    """Apply literal, high-confidence signals after either extraction path."""
    learner_text = " ".join(_learner_messages(history) + [message])
    current = message.strip()
    current_lower = current.lower()

    # Short constraint/experience answers belong to the existing goal even if
    # a model rewrites the answer itself into `goal_text`.
    if existing_goal and _follow_up_only(current) and not ASPIRATION.search(current_lower):
        draft.goal_text = existing_goal.strip()

    if OPEN_ENDED.search(current_lower):
        draft.time_unconstrained = True
        draft.horizon_weeks = None
    elif _has_duration(current_lower):
        # An explicit finite window is also an explicit correction to a prior
        # open-ended answer.
        draft.time_unconstrained = False

    novice_answered = bool(NOVICE.search(learner_text.lower()))
    if novice_answered:
        draft.known_skills = [
            phrase for phrase in draft.known_skills
            if not NOVICE.search(phrase.lower()) and not OPEN_ENDED.search(phrase.lower())
        ]

    goal = (draft.goal_text or existing_goal).strip()
    goal_is_clear = len(goal) >= 20 and bool(ASPIRATION.search(goal.lower()) or existing_goal)
    time_answered = bool(
        draft.weekly_hours is not None
        or draft.horizon_weeks is not None
        or draft.time_unconstrained is True
    )

    question = draft.follow_up_question or ""
    asks_experience = bool(re.search(r"\b(?:experience|background|already|covered|know)\b", question.lower()))
    asks_time = bool(re.search(r"\b(?:time|hours?|weeks?|months?|deadline|window)\b", question.lower()))
    if (novice_answered and asks_experience) or (time_answered and asks_time):
        draft.follow_up_question = None

    if goal_is_clear and novice_answered and time_answered:
        draft.confidence = max(draft.confidence, 0.82)
        draft.follow_up_question = None
    elif draft.confidence < 0.7 and not draft.follow_up_question:
        if not goal_is_clear:
            draft.follow_up_question = "What do you want to be able to do at the end of this?"
        elif not novice_answered and not draft.known_skills:
            draft.follow_up_question = "What experience do you already have in this area?"
        elif not time_answered:
            draft.follow_up_question = "Do you have a deadline, or should I make this an open-ended route?"

    return draft


def _number(raw: str) -> int | None:
    raw = raw.strip().lower()
    return int(raw) if raw.isdigit() else WORD_NUMBERS.get(raw)


def rating_for(phrase: str) -> int:
    """'basic Python' and 'strong Python' must not produce the same profile."""
    for pattern, rating in LEVEL_HINTS:
        if pattern.search(phrase.lower()):
            return rating
    return 3


def _fallback(message: str, history: list[str], existing_goal: str = "") -> ProfileDraft:
    text = " ".join(_learner_messages(history) + [message])
    lowered = text.lower()

    # A follow-up such as "8 hours a week" must not replace the learner's
    # stated goal.  The model sees the whole transcript; this small offline
    # equivalent preserves it when the new turn is clearly a constraint or an
    # experience answer.  An explicit new aspiration still wins.
    current = message.strip()
    explicit_goal = bool(ASPIRATION.search(current.lower()))
    follow_up_only = _follow_up_only(current)
    goal_text = existing_goal.strip() if existing_goal and follow_up_only and not explicit_goal else current
    draft = ProfileDraft(goal_text=goal_text, confidence=0.25)

    number = r"(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|couple|few)"

    hours = re.search(number + r"\s*(?:hours?|hrs?|h)\s*(?:a|per|/|each)\s*week", lowered)
    if hours and (value := _number(hours.group(1))):
        draft.weekly_hours = float(value)
        draft.confidence += 0.2

    months = re.search(number + r"\s*months?", lowered)
    weeks = re.search(number + r"\s*weeks?", lowered)
    if months and (value := _number(months.group(1))):
        draft.horizon_weeks = value * 4
        draft.confidence += 0.15
    elif weeks and (value := _number(weeks.group(1))):
        draft.horizon_weeks = value
        draft.confidence += 0.15

    if OPEN_ENDED.search(lowered):
        draft.time_unconstrained = True
        draft.horizon_weeks = None
        draft.confidence += 0.15

    if re.search(r"\b(free|no budget|cannot pay|can't pay|without paying)\b", lowered):
        draft.budget = "free"
    elif re.search(r"\b(paid|happy to pay|budget is fine|will pay)\b", lowered):
        draft.budget = "paid"

    for fmt, pattern in (
        ("project", r"\b(building|hands.?on|by doing|projects?|practical|portfolio)\b"),
        ("video", r"\b(videos?|lectures?|watching)\b"),
        ("text", r"\b(reading|books?|articles?|docs)\b"),
        ("interactive", r"\b(interactive|exercises?|challenges?)\b"),
    ):
        if re.search(pattern, lowered):
            draft.format_prefs.append(fmt)

    # only mine experience clauses, and stop at the first aspiration verb
    for match in EXPERIENCE.finditer(lowered):
        clause = match.group(1)
        cut = ASPIRATION.search(clause)
        if cut:
            clause = clause[: cut.start()]
        for part in re.split(r",| and ", clause):
            part = part.strip(" .")
            if (
                2 <= len(part) <= 40
                and not ASPIRATION.search(part)
                and not NOVICE.search(part)
                and not OPEN_ENDED.search(part)
            ):
                draft.known_skills.append(part)
    if draft.known_skills:
        draft.confidence += 0.1

    if len(draft.goal_text or "") > 25:
        draft.confidence += 0.15

    novice_answered = bool(NOVICE.search(lowered))
    if novice_answered:
        draft.confidence += 0.2

    if draft.confidence < 0.7:
        if len(draft.goal_text or "") < 25:
            draft.follow_up_question = "What do you want to be able to do at the end of this?"
        elif not novice_answered and not draft.known_skills:
            draft.follow_up_question = "What experience do you already have in this area?"
        elif draft.weekly_hours is None and draft.time_unconstrained is not True:
            draft.follow_up_question = "Do you have a deadline, or should I make this an open-ended route?"
        else:
            draft.follow_up_question = "What have you already covered in this area?"
    return _normalise(draft, message, history, existing_goal=existing_goal)


# ---------------------------------------------------------------------------
# merge + snap
# ---------------------------------------------------------------------------

def merge(profile: LearnerProfile, draft: ProfileDraft) -> LearnerProfile:
    """Apply a draft. Later turns win, so a correction overrides."""
    if draft.goal_text:
        profile.goal_text = draft.goal_text.strip()
    if draft.weekly_hours is not None:
        profile.weekly_hours = float(draft.weekly_hours)
    if draft.horizon_weeks is not None:
        profile.horizon_weeks = int(draft.horizon_weeks)
        profile.time_unconstrained = False
    if draft.time_unconstrained is not None:
        profile.time_unconstrained = bool(draft.time_unconstrained)
    if draft.budget in ("free", "freemium", "paid"):
        profile.budget = draft.budget
    if draft.motivation:
        profile.motivation = draft.motivation

    for fmt in draft.format_prefs:
        if fmt in ("video", "text", "interactive", "project") and fmt not in profile.format_prefs:
            profile.format_prefs.append(fmt)

    for phrase in draft.known_skills:
        phrase = phrase.strip()
        if phrase and phrase.lower() not in {p.lower() for p in profile.claimed_skills}:
            profile.claimed_skills.append(phrase)

    return profile


def snap(profile: LearnerProfile, spec) -> LearnerProfile:
    """Map the learner's own words onto the derived graph's skill ids.

    Runs once the graph exists. "some pandas" has to become whatever id the
    model chose for dataframe work in *this* goal - which is why it cannot
    happen during extraction.
    """
    lookup: dict[str, str] = {}
    for skill in spec.skills:
        lookup[skill.name.lower()] = skill.id
        lookup[skill.id.split(".")[-1].replace("_", " ")] = skill.id

    already = {c.skill_id for c in profile.known_skills}
    for phrase in profile.claimed_skills:
        sid = _match(phrase, lookup)
        if sid and sid not in already:
            profile.known_skills.append(SkillClaim(skill_id=sid, self_rating=rating_for(phrase)))
            already.add(sid)
    return profile


# Words that carry no skill identity. Without these, "write SQL at work" is
# compared whole against "Advanced SQL & Query Optimization", matches nothing,
# and the learner who does that job every day is planned for as a total
# beginner - which is exactly what the profile card was reporting.
_FILLER = {
    "a", "advanced", "am", "an", "and", "at", "basic", "beginner", "bit",
    "comfortable", "confident", "daily", "day", "every", "experience",
    "experienced", "familiar", "for", "good", "i", "in", "intermediate", "job",
    "know", "knowledge", "learning", "little", "my", "of", "on", "professional",
    "professionally", "solid", "some", "strong", "the", "to", "use", "used",
    "using", "with", "work", "working", "write", "writing", "year", "years",
}


def _tokens(text: str) -> set[str]:
    return {
        word
        for word in re.split(r"[^a-z0-9+#.]+", text.lower())
        if len(word) > 1 and word not in _FILLER
    }


def _match(phrase: str, lookup: dict[str, str]) -> str | None:
    cleaned = re.sub(
        r"\b(basic|beginner|some|solid|strong|advanced|intermediate|good at|little|"
        r"years? of|experienced in|working knowledge of)\b", " ", phrase.lower()
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    if not cleaned:
        return None
    if cleaned in lookup:
        return lookup[cleaned]
    for name, sid in lookup.items():
        if cleaned in name or name in cleaned:
            return sid
    # Overlap on the words that actually name a technology or a practice. Ranked
    # so "SQL" prefers the skill that is mostly about SQL over one that merely
    # mentions it, with ties broken on the shorter name so the result is stable.
    wanted = _tokens(cleaned)
    if wanted:
        scored = [
            (len(shared) / len(_tokens(name) | wanted), -len(name), sid)
            for name, sid in lookup.items()
            if (shared := wanted & _tokens(name))
        ]
        if scored:
            return max(scored)[2]

    hit = difflib.get_close_matches(cleaned, list(lookup), n=1, cutoff=0.8)
    return lookup[hit[0]] if hit else None
