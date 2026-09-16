"""Persistence schema.

The shape of this schema encodes two decisions worth stating.

First, **a verdict is a first-class row, not a field on a test result.** A test
can produce an execution without a verdict for a long time — execution happens
immediately, judgement may wait on a differential run or a human. Keeping them
separate means an unjudged execution is a queryable state rather than a null
waiting to be misinterpreted as "passed".

Second, **`Execution.classification` is an enum with an `unknown` member.** The
dangerous failure mode is a failed test being reported as a bug; forcing every
execution through a classification makes "we do not know why this failed" a
recorded outcome instead of an implicit one.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def new_id(prefix: str) -> str:
    """Return a short, readable, prefixed identifier."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Declarative base for every table."""


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ABORTED = "aborted"


class CaseStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERRORED = "errored"
    SKIPPED = "skipped"


class Classification(StrEnum):
    """Why an execution produced the result it did.

    ``UNKNOWN`` is deliberate and must not be treated as ``APP_BUG``. Most
    false positives in automated testing come from collapsing "something went
    wrong" into "the application is broken".
    """

    APP_BUG = "app_bug"
    TEST_BUG = "test_bug"
    FLAKE = "flake"
    ENV = "env"
    UNKNOWN = "unknown"


class VerdictDecision(StrEnum):
    BUG = "bug"
    NOT_A_BUG = "not_a_bug"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FindingStatus(StrEnum):
    OPEN = "open"
    CONFIRMED = "confirmed"
    FALSE_POSITIVE = "false_positive"
    FIXED = "fixed"


class Target(Base):
    """An application under test."""

    __tablename__ = "targets"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("tgt"))
    name: Mapped[str] = mapped_column(String(200))
    base_url: Mapped[str] = mapped_column(String(500))
    repo_url: Mapped[str | None] = mapped_column(String(500), default=None)
    tracked_ref: Mapped[str | None] = mapped_column(String(200), default=None)
    #: Data classification for everything derived from this target.
    data_class: Mapped[str] = mapped_column(String(20), default="proprietary")
    #: Must be opted into explicitly; production targets are refused by default.
    is_production: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    runs: Mapped[list[Run]] = relationship(back_populates="target")

    __table_args__ = (Index("ix_targets_base_url", "base_url"),)

    def __repr__(self) -> str:
        return f"<Target {self.id} {self.name!r}>"


class Run(Base):
    """One execution of the pipeline against a target."""

    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("run"))
    target_id: Mapped[str] = mapped_column(ForeignKey("targets.id"), index=True)
    status: Mapped[RunStatus] = mapped_column(
        Enum(RunStatus, native_enum=False, validate_strings=True),
        default=RunStatus.PENDING,
    )
    seed: Mapped[int | None] = mapped_column(Integer, default=None)
    commit_sha: Mapped[str | None] = mapped_column(String(64), default=None)
    #: Which mutant the target was configured with. The benchmark needs this to
    #: know which defects were actually present.
    mutant_selection: Mapped[str | None] = mapped_column(String(500), default=None)
    spent_usd: Mapped[float] = mapped_column(Float, default=0.0)
    tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    error: Mapped[str | None] = mapped_column(Text, default=None)

    target: Mapped[Target] = relationship(back_populates="runs")
    app_map: Mapped[AppMap | None] = relationship(back_populates="run", uselist=False)
    plans: Mapped[list[TestPlan]] = relationship(back_populates="run")
    findings: Mapped[list[Finding]] = relationship(back_populates="run")

    def __repr__(self) -> str:
        return f"<Run {self.id} {self.status}>"


class AppMap(Base):
    """The world model the recon stage builds.

    Versioned per run so drift between runs is diffable — a route disappearing
    is itself a finding worth surfacing.
    """

    __tablename__ = "app_maps"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("map"))
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), unique=True, index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    #: Serialised routes, forms, entities, and discovered API operations.
    data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    route_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    run: Mapped[Run] = relationship(back_populates="app_map")

    def __repr__(self) -> str:
        return f"<AppMap {self.id} routes={self.route_count}>"


class TestPlan(Base):
    """A batch of generated test cases produced from one strategy."""

    # Tells pytest not to try to collect this as a test class. Without it the
    # Test* prefix makes pytest attempt collection and emit a warning per run.
    __test__ = False

    __tablename__ = "test_plans"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("plan"))
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    strategy: Mapped[str] = mapped_column(String(100))
    #: Free-form plan metadata: coverage criteria, risk rubric, model used.
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    run: Mapped[Run] = relationship(back_populates="plans")
    cases: Mapped[list[TestCase]] = relationship(back_populates="plan")

    def __repr__(self) -> str:
        return f"<TestPlan {self.id} {self.strategy}>"


class TestCase(Base):
    """A single executable test derived from a requirement or a discovered surface."""

    __test__ = False

    __tablename__ = "test_cases"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("case"))
    plan_id: Mapped[str] = mapped_column(ForeignKey("test_plans.id"), index=True)
    title: Mapped[str] = mapped_column(String(300))
    lane: Mapped[str] = mapped_column(String(50), default="ui")
    #: Traceability: which requirement or discovered surface this covers. A case
    #: with no reference is untraceable and should be visible as such.
    requirement_ref: Mapped[str | None] = mapped_column(String(300), default=None)
    risk: Mapped[int] = mapped_column(Integer, default=1)
    #: Executable source. Stored so a run can be reproduced without regeneration.
    code: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[CaseStatus] = mapped_column(
        Enum(CaseStatus, native_enum=False, validate_strings=True),
        default=CaseStatus.PENDING,
    )
    #: Rolling instability score. Cases above a threshold get quarantined by the
    #: critic rather than repeatedly reported as failures.
    flake_score: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    plan: Mapped[TestPlan] = relationship(back_populates="cases")
    executions: Mapped[list[Execution]] = relationship(back_populates="case")

    __table_args__ = (Index("ix_test_cases_requirement", "requirement_ref"),)

    def __repr__(self) -> str:
        return f"<TestCase {self.id} {self.title[:40]!r}>"


class Execution(Base):
    """One attempt at running a test case."""

    __tablename__ = "executions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("exec"))
    case_id: Mapped[str] = mapped_column(ForeignKey("test_cases.id"), index=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[CaseStatus] = mapped_column(
        Enum(CaseStatus, native_enum=False, validate_strings=True),
        default=CaseStatus.PENDING,
    )
    classification: Mapped[Classification] = mapped_column(
        Enum(Classification, native_enum=False, validate_strings=True),
        default=Classification.UNKNOWN,
    )
    duration_ms: Mapped[float] = mapped_column(Float, default=0.0)
    #: Artifact keys (screenshot, video, HAR, console log) in the artifact store.
    artifacts: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    stdout: Mapped[str] = mapped_column(Text, default="")
    stderr: Mapped[str] = mapped_column(Text, default="")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)

    case: Mapped[TestCase] = relationship(back_populates="executions")
    verdicts: Mapped[list[Verdict]] = relationship(back_populates="execution")

    __table_args__ = (
        Index("ix_executions_case_attempt", "case_id", "attempt", unique=True),
    )

    def __repr__(self) -> str:
        return f"<Execution {self.id} {self.status}/{self.classification}>"


class Verdict(Base):
    """A judgement about what an execution means.

    Separate from :class:`Execution` because judgement and observation have
    different lifetimes and different failure modes.
    """

    __tablename__ = "verdicts"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("vrd"))
    execution_id: Mapped[str] = mapped_column(ForeignKey("executions.id"), index=True)
    decision: Mapped[VerdictDecision] = mapped_column(
        Enum(VerdictDecision, native_enum=False, validate_strings=True)
    )
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    #: Per-signal outcomes, including disagreements. Recorded rather than
    #: discarded, because it is the only data that calibrates the thresholds.
    signals: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    #: Citations the judge was required to produce.
    evidence: Mapped[list[Any]] = mapped_column(JSON, default=list)
    model: Mapped[str | None] = mapped_column(String(120), default=None)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    execution: Mapped[Execution] = relationship(back_populates="verdicts")

    def __repr__(self) -> str:
        return f"<Verdict {self.id} {self.decision} conf={self.confidence:.2f}>"


class Finding(Base):
    """A clustered, deduplicated defect report."""

    __tablename__ = "findings"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("fnd"))
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    title: Mapped[str] = mapped_column(String(300))
    severity: Mapped[Severity] = mapped_column(
        Enum(Severity, native_enum=False, validate_strings=True), default=Severity.MEDIUM
    )
    status: Mapped[FindingStatus] = mapped_column(
        Enum(FindingStatus, native_enum=False, validate_strings=True),
        default=FindingStatus.OPEN,
    )
    root_cause: Mapped[str | None] = mapped_column(Text, default=None)
    repro_steps: Mapped[list[Any]] = mapped_column(JSON, default=list)
    evidence: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    #: For the benchmark: which declared defect this corresponds to, if any.
    #: A finding with no match is a false positive; a declared defect with no
    #: finding is a miss. That pairing is the recall/precision measurement.
    matched_bug_id: Mapped[str | None] = mapped_column(String(80), default=None, index=True)
    ticket_ref: Mapped[str | None] = mapped_column(String(200), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    run: Mapped[Run] = relationship(back_populates="findings")

    def __repr__(self) -> str:
        return f"<Finding {self.id} {self.severity} matched={self.matched_bug_id}>"


class LLMSpanRow(Base):
    """Persisted cost and token ledger entry.

    Denormalised on purpose: cost needs to be attributable per case and per
    agent long after the run's in-memory ledger is gone.
    """

    __tablename__ = "llm_spans"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("span"))
    run_id: Mapped[str | None] = mapped_column(ForeignKey("runs.id"), index=True, default=None)
    case_id: Mapped[str | None] = mapped_column(String(32), index=True, default=None)
    provider: Mapped[str] = mapped_column(String(40))
    model: Mapped[str] = mapped_column(String(120))
    tier: Mapped[str] = mapped_column(String(20))
    data_class: Mapped[str] = mapped_column(String(20))
    agent: Mapped[str] = mapped_column(String(60))
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    ok: Mapped[bool] = mapped_column(Boolean, default=True)
    escalated: Mapped[bool] = mapped_column(Boolean, default=False)
    attempts: Mapped[int] = mapped_column(Integer, default=1)
    error: Mapped[str | None] = mapped_column(Text, default=None)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)

    def __repr__(self) -> str:
        return f"<LLMSpanRow {self.provider}/{self.model} {self.cost_usd:.6f}>"


class Memory(Base):
    """Target-scoped learned state: the flake ledger and known-issue suppressors."""

    __tablename__ = "memory"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("mem"))
    target_id: Mapped[str | None] = mapped_column(ForeignKey("targets.id"), index=True, default=None)
    kind: Mapped[str] = mapped_column(String(50), index=True)
    key: Mapped[str] = mapped_column(String(300), index=True)
    value: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    __table_args__ = (
        Index("ix_memory_target_kind_key", "target_id", "kind", "key", unique=True),
    )

    def __repr__(self) -> str:
        return f"<Memory {self.kind}:{self.key}>"
