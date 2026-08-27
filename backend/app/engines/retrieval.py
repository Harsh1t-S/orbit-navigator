"""Hybrid retrieval + the scorer that produces reason codes.

Retrieval is BM25 fused with a dense/semantic score via reciprocal rank
fusion. Dense means sentence-transformers when it is installed, TF-IDF when it
is not - the fallback is noticeably worse on paraphrases but it keeps the
install to ~50MB and the app always runs.

The scorer is the important half. Every term in the final score is named and
kept, and each one emits a reason code as a side effect. Nothing downstream
ever has to guess why a course was picked, which is what stops the explanation
layer from hallucinating.
"""

from __future__ import annotations

import logging
import math
import re

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from ..catalog import Catalog
from ..config import settings
from ..schemas import Course, LearnerProfile, ReasonCode, Recommendation

log = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9+#.]+")


def _tok(s: str) -> list[str]:
    return _TOKEN.findall(s.lower())


class Retriever:
    def __init__(self, cat: Catalog):
        self.cat = cat
        self.ids = [c.id for c in cat.courses]
        corpus = [c.search_text() for c in cat.courses]

        self.bm25 = BM25Okapi([_tok(t) for t in corpus])

        self.encoder = None
        if settings.use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer

                self.encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
                self.matrix = self.encoder.encode(corpus, normalize_embeddings=True)
                log.info("retrieval: using sentence-transformers embeddings")
            except Exception as exc:
                log.info("retrieval: sentence-transformers unavailable (%s), using TF-IDF", exc)

        if self.encoder is None:
            self.tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
            self.matrix = self.tfidf.fit_transform(corpus)
            self.matrix = self.matrix / (np.sqrt(self.matrix.multiply(self.matrix).sum(1)) + 1e-9)


    # -- retrieval ---------------------------------------------------------

    def _dense_scores(self, query: str) -> np.ndarray:
        if self.encoder is not None:
            q = self.encoder.encode([query], normalize_embeddings=True)
            return (self.matrix @ q.T).ravel()
        q = self.tfidf.transform([query])
        q = q / (np.sqrt(q.multiply(q).sum()) + 1e-9)
        return np.asarray((self.matrix @ q.T).todense()).ravel()

    def search(self, query: str, k: int = 60) -> dict[str, float]:
        """Reciprocal rank fusion of BM25 and dense. Returns id -> fused score
        in [0,1], already normalised so it can be dropped straight into the
        weighted sum."""
        if not query.strip():
            return {cid: 0.5 for cid in self.ids}

        dense = self._dense_scores(query)
        sparse = np.asarray(self.bm25.get_scores(_tok(query)), dtype=np.float32)

        fused: dict[str, float] = {}
        for scores in (dense, sparse):
            order = np.argsort(-scores)
            for rank, idx in enumerate(order[: k * 2]):
                fused[self.ids[idx]] = fused.get(self.ids[idx], 0.0) + 1.0 / (60 + rank)

        if not fused:
            return {}
        top = max(fused.values())
        return {cid: s / top for cid, s in sorted(fused.items(), key=lambda kv: -kv[1])[:k]}

    # -- scoring -----------------------------------------------------------

    def rank(
        self,
        profile: LearnerProfile,
        m: np.ndarray,
        gap: np.ndarray,
        query: str,
        selected: list[str] | None = None,
        limit: int = 15,
    ) -> list[Recommendation]:
        selected = selected or []
        semantic = self.search(query)
        gap_total = float(gap.sum()) or 1.0

        selected_teaches = (
            np.max([self.cat.teaches_vec(c) for c in selected], axis=0)
            if selected
            else np.zeros(self.cat.n_skills, dtype=np.float32)
        )

        w = _weights(profile)
        out: list[Recommendation] = []

        for course in self.cat.courses:
            if course.id in profile.completed_courses or course.id in profile.rejected_courses:
                continue

            teaches = self.cat.teaches_vec(course.id)
            reasons: list[ReasonCode] = []

            # --- gap coverage: the reason the course exists in this path at all
            covered = np.minimum(teaches, gap)
            gap_cov = float(covered.sum()) / gap_total
            covers = [
                self.cat.skills[i].id
                for i in np.argsort(-covered)[:5]
                if covered[i] > 0.02
            ]
            if covers:
                reasons.append(
                    ReasonCode(
                        type="GAP_COVERAGE",
                        detail={"skills": covers, "names": [self.cat.name(s) for s in covers]},
                        contribution=round(gap_cov, 4),
                    )
                )

            # --- semantic relevance to what they typed
            sem = semantic.get(course.id, 0.0)

            # --- level fit: penalise both directions, not just too-hard
            lvl = _level_fit(course, m, self.cat)
            reasons.append(
                ReasonCode(
                    type="LEVEL_FIT",
                    detail={"course_level": course.level, "fit": round(lvl, 3)},
                    contribution=round(lvl, 4),
                )
            )

            # --- source transparency: a resource whose provider states its
            #     own duration is a better-specified resource than one where we
            #     had to estimate. Weak signal, but it is a real one, unlike the
            #     invented star ratings this replaced.
            specificity = 1.0 if course.hours_stated else 0.55

            # --- preference fit: format, budget, and whether it fits a sane
            #     number of weeks at their pace
            pref, pref_detail = _preference_fit(course, profile)
            if pref_detail:
                reasons.append(ReasonCode(type="PREFERENCE_FIT", detail=pref_detail, contribution=round(pref, 4)))

            weeks = course.hours / max(profile.weekly_hours, 1.0)
            reasons.append(
                ReasonCode(
                    type="TIME_FIT",
                    detail={"hours": course.hours, "weeks": round(weeks, 1)},
                    contribution=0.0,
                )
            )

            # --- redundancy against what is already in the path
            overlap = float(np.minimum(teaches, selected_teaches).sum())
            redundancy = overlap / (float(teaches.sum()) + 1e-6)

            # --- unmet prerequisites are not a penalty here, the planner
            #     handles ordering. But we record them so the explainer can say
            #     "after you finish X".
            unmet = _unmet_prereqs(course, m, self.cat)
            if unmet:
                reasons.append(ReasonCode(type="NEEDS_FIRST", detail={"skills": unmet}))

            score = (
                w["gap"] * gap_cov
                + w["semantic"] * sem
                + w["level"] * lvl
                + w["quality"] * specificity
                + w["preference"] * pref
                - w["redundancy"] * redundancy
            )

            out.append(
                Recommendation(
                    course=course,
                    score=round(float(score), 4),
                    components={
                        "gap_coverage": round(gap_cov, 4),
                        "semantic": round(float(sem), 4),
                        "level_fit": round(lvl, 4),
                        "quality": round(specificity, 4),
                        "preference": round(pref, 4),
                        "redundancy": round(redundancy, 4),
                    },
                    reasons=reasons,
                    covers=covers,
                )
            )

        out.sort(key=lambda r: r.score, reverse=True)
        return out[:limit]



# ---------------------------------------------------------------------------
# scoring pieces - kept module-level so the eval notebook can import them
# ---------------------------------------------------------------------------

def _weights(profile: LearnerProfile) -> dict[str, float]:
    base = {
        "gap": settings.w_gap,
        "semantic": settings.w_semantic,
        "level": settings.w_level,
        "quality": settings.w_quality,
        "preference": settings.w_preference,
        "redundancy": settings.w_redundancy,
    }
    base.update({k: v for k, v in profile.weight_overrides.items() if k in base})
    return base


def _level_fit(course: Course, m: np.ndarray, cat: Catalog) -> float:
    """Where does this course sit relative to what they can already do?

    We compare course level against mastery of the skills it teaches, not
    against a global "learner level" - somebody can be advanced in SQL and a
    beginner in PyTorch at the same time, and a single number hides that.
    """
    idxs = [cat.index[s] for s in course.teaches if s in cat.index]
    if not idxs:
        return 0.5
    local = float(np.mean([m[i] for i in idxs]))
    # ideal course sits a little above current mastery
    delta = course.level_value - (local + 0.15)
    return float(math.exp(-(delta ** 2) / 0.08))




def _preference_fit(course: Course, profile: LearnerProfile) -> tuple[float, dict]:
    score, detail = 0.5, {}

    if profile.format_prefs:
        if course.format in profile.format_prefs:
            score += 0.25
            detail["format"] = course.format
        else:
            score -= 0.1

    order = {"free": 0, "freemium": 1, "paid": 2}
    if order[course.cost] <= order[profile.budget]:
        score += 0.15
        if course.cost == "free":
            detail["cost"] = "free"
    else:
        score -= 0.3
        detail["cost_mismatch"] = course.cost

    # a 60-hour course at 3h/week is five months. That is not a fit.
    weeks = course.hours / max(profile.weekly_hours, 1.0)
    if not profile.time_unconstrained and weeks > profile.horizon_weeks * 0.6:
        score -= 0.2
        detail["too_long"] = round(weeks, 1)

    return float(np.clip(score, 0.0, 1.0)), detail


def _unmet_prereqs(course: Course, m: np.ndarray, cat: Catalog) -> list[str]:
    out = []
    for sid, need in course.requires.items():
        i = cat.index.get(sid)
        if i is not None and m[i] < need - 0.05:
            out.append(sid)
    return out

