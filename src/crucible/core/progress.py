"""What the agent knows about its own run, and what it says when asked.

The dashboard lets the operator type at the agent while a scan is running. With
no model provider configured there is no way to *understand* an arbitrary
message, so this module does the honest thing instead of the impressive one: it
keeps a live snapshot of the run and answers the questions an operator actually
asks -- "what have you found?", "how far along are you?" -- from facts the run
really produced. A message it cannot place gets the current status plus a plain
statement that it cannot hold a conversation, which is better than a fluent
reply that means nothing.

The snapshot is driven by the event bus rather than by hand-written updates at
each stage, so it cannot drift out of step with what the dashboard is showing:
both read the same events.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from crucible.core.events import Event, EventType

#: Word sets rather than substring tests: "hi" must not match "this", and
#: "why" must not match "shy".
_STOP_WORDS = frozenset({"stop", "abort", "cancel", "halt", "kill", "quit"})
_HELP_WORDS = frozenset({"help", "commands", "capabilities", "options", "menu"})
_FINDING_WORDS = frozenset(
    {
        "bug",
        "bugs",
        "issue",
        "issues",
        "finding",
        "findings",
        "problem",
        "problems",
        "found",
        "defect",
        "defects",
        "broke",
        "broken",
        "anything",
    }
)
_EXPLAIN_WORDS = frozenset({"why", "explain", "cause", "reason", "detail", "details"})
_SCORE_WORDS = frozenset({"accuracy", "score", "recall", "precision", "benchmark"})
_VIDEO_WORDS = frozenset(
    {"video", "recording", "recorded", "browser", "screenshot", "screenshots", "watch"}
)
_LOGIN_WORDS = frozenset(
    {"login", "log", "password", "credentials", "credential", "auth", "signin", "account"}
)
_TARGET_WORDS = frozenset({"target", "url", "site", "website", "testing", "scanning", "test"})
_STATUS_WORDS = frozenset(
    {"status", "progress", "doing", "update", "going", "far", "stage", "now", "busy", "time"}
)
_GREETING_WORDS = frozenset({"hi", "hello", "hey", "yo", "hiya", "morning", "evening"})
_THANKS_WORDS = frozenset({"thanks", "thank", "cheers", "ta"})

#: Longest echo of the operator's own words repeated back in a fallback reply.
_ECHO_LIMIT = 120


@dataclass(slots=True)
class FindingNote:
    """One finding, in the words the operator would use.

    ``cause`` and ``fix`` stay empty until triage has produced them, so the
    agent can talk about a finding the moment it is recorded and get more
    specific once it knows more.
    """

    title: str
    cause: str = ""
    fix: str = ""


@dataclass(slots=True)
class RunProgress:
    """A live, plain-language snapshot of one run."""

    target: str
    stage: str = "starting up"
    routes: int = 0
    #: Pages fetched so far. Distinct from ``routes``: this climbs during the
    #: crawl, so the agent can answer a question asked mid-recon with a number
    #: instead of "nothing yet" while it is visibly working.
    pages_seen: int = 0
    forms: int = 0
    broken: int = 0
    cases_total: int = 0
    cases_run: int = 0
    notes: list[FindingNote] = field(default_factory=list)
    decision: str | None = None
    recall: float | None = None
    precision: float | None = None
    video_key: str | None = None
    finished: bool = False

    def note(self, finding: FindingNote) -> None:
        """Record a finding once, upgrading it if richer detail arrives."""
        if not finding.title:
            return
        for index, existing in enumerate(self.notes):
            if existing.title == finding.title:
                if finding.cause or finding.fix:
                    self.notes[index] = finding
                return
        self.notes.append(finding)

    def replace_notes(self, findings: list[FindingNote]) -> None:
        """Take triage's fuller version of the findings list, if it has one."""
        if findings:
            self.notes = list(findings)

    def state_brief(self) -> str:
        """A compact, factual account of the run for a model or a reader.

        Deliberately plain and complete: it is the only thing the agent is
        allowed to base a claim on, so anything missing here is something it
        will say it does not know rather than guess at.
        """
        lines = [
            f"Target: {self.target}",
            f"Stage: {self.stage}",
            f"Pages found: {self.routes}",
            f"Pages returning errors: {self.broken}",
            f"Checks run: {self.cases_run} of {self.cases_total}",
            f"Finished: {'yes' if self.finished else 'no'}",
        ]
        if self.decision:
            lines.append(f"Verdict: {self.decision}")
        if self.recall is not None and self.precision is not None:
            lines.append(f"Recall {self.recall:.0%}, precision {self.precision:.0%}")
        if self.video_key:
            lines.append(f"Browser recording: {self.video_key}")
        if not self.notes:
            lines.append("Findings: none yet")
        else:
            lines.append(f"Findings ({len(self.notes)}):")
            for note in self.notes:
                detail = f"  - {note.title}"
                if note.cause:
                    detail += f" Why: {note.cause}"
                if note.fix:
                    detail += f" Fix: {note.fix}"
                lines.append(detail)
        return "\n".join(lines)

    def reply(self, text: str) -> str:
        """Compose a reply to an operator message from observed state.

        This is not a conversation and does not pretend to be. It matches the
        handful of things an operator asks a running scan and answers them from
        facts; anything else gets the status plus a straight statement of the
        limitation.
        """
        words = _words(text)
        if words & _STOP_WORDS:
            return self._stopping()
        if words & _HELP_WORDS:
            return self._help()
        # Findings before counts so "how many bugs" lists them, not just counts.
        if words & _FINDING_WORDS:
            return self._findings()
        if words & _EXPLAIN_WORDS:
            return self._explain()
        if words & _SCORE_WORDS:
            return self._score()
        if words & _VIDEO_WORDS:
            return self._video()
        if words & _LOGIN_WORDS:
            return self._login()
        if words & _TARGET_WORDS:
            return self._target()
        if words & _STATUS_WORDS:
            return self._status()
        if words & _GREETING_WORDS:
            return f"Hello. {self._stage_sentence()} {self._stats_sentence()}"
        if words & _THANKS_WORDS:
            return f"Any time. {self._stats_sentence()}"
        return self._unknown(text)

    def _stage_sentence(self) -> str:
        if self.finished:
            return "I've finished this scan."
        return f"I'm {self.stage}."

    def _stats_sentence(self) -> str:
        parts: list[str] = []
        if self.routes:
            parts.append(f"{self.routes} page(s) mapped")
        elif self.pages_seen:
            parts.append(f"looked at {self.pages_seen} page(s) already")
        if self.broken:
            parts.append(f"{self.broken} of them returning errors")
        if self.cases_total:
            parts.append(f"{self.cases_run} of {self.cases_total} checks run")
        if self.notes:
            parts.append(f"{len(self.notes)} finding(s)")
        elif self.finished:
            parts.append("no findings")
        if not parts:
            return "Nothing has come back from the target yet."
        return "So far: " + ", ".join(parts) + "."

    def _status(self) -> str:
        return f"{self._stage_sentence()} {self._stats_sentence()}"

    def _findings(self) -> str:
        if not self.notes:
            if self.finished:
                return (
                    f"No findings. The scan finished and nothing failed a check "
                    f"across {self.routes} page(s)."
                )
            return f"Nothing yet. {self._stage_sentence()} {self._stats_sentence()}"
        listed = "\n".join(
            f"{index}. {note.title}" for index, note in enumerate(self.notes, 1)
        )
        header = f"{len(self.notes)} finding(s) so far"
        header += "." if self.finished else ", and I'm still going."
        return f"{header}\n{listed}\n\nAsk \"why\" and I'll explain the last one."

    def _explain(self) -> str:
        if not self.notes:
            return self._findings()
        note = self.notes[-1]
        lines = [f'About "{note.title}":']
        if note.cause:
            lines.append(f"Why: {note.cause}")
        if note.fix:
            lines.append(f"Fix: {note.fix}")
        if len(lines) == 1:
            lines.append("The full evidence is in the Findings panel on the dashboard.")
        return "\n".join(lines)

    def _score(self) -> str:
        if self.recall is None or self.precision is None:
            return (
                "I score myself against targets that publish a list of known "
                "defects. This one doesn't, so there is no score to report -- the "
                "findings are the whole result."
            )
        return (
            f"Recall {self.recall:.0%} -- that's how many of the known defects I "
            f"found. Precision {self.precision:.0%} -- that's how many of my "
            f"findings turned out to be real."
        )

    def _video(self) -> str:
        if self.video_key:
            return (
                f"Yes. I recorded the browser walk to {self.video_key}. It plays in "
                f'the dashboard under "What the browser saw".'
            )
        if self.finished:
            return (
                "Not this time -- the browser lane didn't run, so there is nothing "
                "recorded. The text and server checks still ran."
            )
        return (
            "Not yet. I open a real browser near the end of the scan and record "
            "that; the video comes from that step."
        )

    def _login(self) -> str:
        return (
            "When I hit a page that needs a login I stop and ask you whether to "
            "skip it or to use credentials. I'll do that here if I run into one."
        )

    def _target(self) -> str:
        return f"I'm scanning {self.target}. {self._stats_sentence()}"

    def _stopping(self) -> str:
        return (
            "I can't be stopped from the dashboard once a scan starts -- there's no "
            "cancel button yet, and closing the tab doesn't stop me. I run to the "
            "end and save the results, which usually takes under a minute."
        )

    def _help(self) -> str:
        return (
            "I answer from what I've observed so far:\n"
            '- "what have you found" / "any bugs"\n'
            '- "how far along are you" / "status"\n'
            '- "why" -- explain the most recent finding\n'
            '- "what\'s my score"\n'
            '- "where\'s the video"\n'
            '- "what are you testing"\n'
            "I can't take commands yet -- there's no cancel or retarget."
        )

    def _unknown(self, text: str) -> str:
        echo = text.strip()
        if len(echo) > _ECHO_LIMIT:
            echo = echo[: _ECHO_LIMIT - 3] + "..."
        return (
            f'You said: "{echo}"\n\n'
            "I can't hold a conversation yet -- no model provider is configured, so "
            "I answer from what I have actually observed rather than guessing at "
            f"what you meant. {self._stage_sentence()} {self._stats_sentence()}\n\n"
            'Try "what have you found", "how far along are you" or "why".'
        )


class ProgressTracker:
    """Keeps a :class:`RunProgress` in step with the run's event stream.

    Built as a bus subscriber so the snapshot is derived from the same events
    the dashboard renders. That is the point: what the agent says it is doing
    and what the operator sees it doing cannot disagree.
    """

    def __init__(self, progress: RunProgress) -> None:
        self._progress = progress

    @property
    def progress(self) -> RunProgress:
        return self._progress

    def __call__(self, event: Event) -> None:
        progress = self._progress
        payload = event.payload

        match event.type:
            case EventType.RECON_STARTED:
                progress.stage = "crawling the site"
            case EventType.RECON_PAGE_VISITED:
                progress.pages_seen += 1
            case EventType.RECON_FINISHED:
                progress.routes = _as_int(payload.get("routes"))
                progress.forms = _as_int(payload.get("forms"))
                progress.broken = _as_int(payload.get("failed"))
            case EventType.PLAN_STARTED:
                progress.stage = "deciding what to test"
            case EventType.PLAN_CASE_ADDED:
                progress.cases_total += 1
            case EventType.EXEC_STARTED:
                progress.stage = "running the checks"
            case EventType.EXEC_CASE_FINISHED:
                progress.cases_run += 1
            case EventType.BROWSER_STARTED:
                progress.stage = "recording the site in a real browser"
            case EventType.BROWSER_FINISHED:
                progress.video_key = payload.get("video_key") or None
            case EventType.FINDING_RECORDED:
                progress.note(FindingNote(title=str(payload.get("title") or "")))
            case EventType.VERDICT_REACHED:
                progress.stage = "grouping what I found"
                progress.decision = payload.get("decision") or None
            case EventType.RUN_FINISHED:
                progress.finished = True
                benchmark = payload.get("benchmark") or {}
                progress.recall = _as_float(benchmark.get("recall"))
                progress.precision = _as_float(benchmark.get("precision"))


def _words(text: str) -> set[str]:
    """Split a message into lowercase words, keeping digits and apostrophes."""
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
