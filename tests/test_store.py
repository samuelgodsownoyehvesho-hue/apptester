"""Schema and session behaviour.

The assertions worth caring about encode design decisions: classifications must
default to ``unknown`` rather than to a bug, unjudged executions must be
representable, and foreign keys must actually be enforced (SQLite disables them
unless the pragma is set, so this is a real check rather than a formality).

Note on identifiers: SQLAlchemy applies Python-side column defaults at INSERT,
not at object construction, so ``run.id`` is ``None`` until a flush. The
helpers below flush explicitly, which is also the pattern the pipeline uses —
it needs the identifier to emit events and name artifacts before committing.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from crucible.store import (
    CaseStatus,
    Classification,
    Execution,
    Finding,
    FindingStatus,
    Memory,
    Run,
    RunStatus,
    Severity,
    Target,
    TestCase,
    TestPlan,
    Verdict,
    VerdictDecision,
    init_db,
    make_engine,
    make_session_factory,
    session_scope,
)


@pytest.fixture
def factory(tmp_path: Path) -> Iterator[sessionmaker[Session]]:
    engine = make_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    yield make_session_factory(engine)
    engine.dispose()


def _seed_target(session: Session, *, name: str = "Nimbus Supply") -> Target:
    target = Target(name=name, base_url="http://localhost:3100", data_class="public")
    session.add(target)
    session.flush()
    return target


def _seed_run(session: Session, target: Target, **overrides: object) -> Run:
    run = Run(target_id=target.id, status=RunStatus.RUNNING, **overrides)
    session.add(run)
    session.flush()
    return run


def _seed_case(session: Session, run: Run, title: str = "Add to cart updates total") -> TestCase:
    plan = TestPlan(run_id=run.id, strategy="recon")
    session.add(plan)
    session.flush()

    case = TestCase(plan_id=plan.id, title=title)
    session.add(case)
    session.flush()
    return case


class TestSchemaCreation:
    def test_init_db_is_idempotent(self, tmp_path: Path) -> None:
        engine = make_engine(f"sqlite:///{tmp_path / 'a.db'}")
        init_db(engine)
        init_db(engine)  # must not raise
        engine.dispose()

    def test_expected_tables_exist(self, tmp_path: Path) -> None:
        engine = make_engine(f"sqlite:///{tmp_path / 'b.db'}")
        init_db(engine)
        with engine.connect() as connection:
            names = set(engine.dialect.get_table_names(connection))
        assert {
            "targets",
            "runs",
            "app_maps",
            "test_plans",
            "test_cases",
            "executions",
            "verdicts",
            "findings",
            "llm_spans",
            "memory",
        } <= names
        engine.dispose()


class TestForeignKeys:
    def test_foreign_keys_are_enforced(self, factory: sessionmaker[Session]) -> None:
        # SQLite ignores foreign keys unless the pragma is set, so this
        # verifies make_engine actually applied it.
        with pytest.raises(IntegrityError), session_scope(factory) as session:
            session.add(Run(target_id="tgt_does_not_exist"))

    def test_run_links_to_target(self, factory: sessionmaker[Session]) -> None:
        with session_scope(factory) as session:
            target = _seed_target(session)
            _seed_run(session, target)
            target_id = target.id

        with session_scope(factory) as session:
            loaded = session.query(Run).one()
            assert loaded.target_id == target_id
            assert loaded.target.name == "Nimbus Supply"


class TestDesignInvariants:
    def test_classification_defaults_to_unknown_not_bug(
        self, factory: sessionmaker[Session]
    ) -> None:
        # Collapsing "something went wrong" into "the application is broken" is
        # the largest single source of false positives.
        with session_scope(factory) as session:
            target = _seed_target(session)
            run = _seed_run(session, target)
            case = _seed_case(session, run)
            session.add(Execution(case_id=case.id, run_id=run.id))
            session.flush()

        with session_scope(factory) as session:
            loaded = session.query(Execution).one()
            assert loaded.classification is Classification.UNKNOWN
            assert loaded.status is CaseStatus.PENDING

    def test_execution_may_exist_without_a_verdict(
        self, factory: sessionmaker[Session]
    ) -> None:
        # Judgement can lag observation, so an unjudged execution must be a
        # valid state rather than something that looks like a pass.
        with session_scope(factory) as session:
            target = _seed_target(session)
            run = _seed_run(session, target)
            case = _seed_case(session, run)
            session.add(Execution(case_id=case.id, run_id=run.id))

        with session_scope(factory) as session:
            assert session.query(Execution).count() == 1
            assert session.query(Verdict).count() == 0

    def test_finding_records_which_declared_bug_it_matched(
        self, factory: sessionmaker[Session]
    ) -> None:
        # The benchmark's recall and precision measurement depends on pairing a
        # finding with a declared defect.
        with session_scope(factory) as session:
            target = _seed_target(session)
            run = _seed_run(session, target, mutant_selection="all")
            session.add(
                Finding(
                    run_id=run.id,
                    title="Cart total ignores quantity",
                    severity=Severity.HIGH,
                    matched_bug_id="CART_QTY_IGNORED",
                )
            )

        with session_scope(factory) as session:
            finding = session.query(Finding).one()
            assert finding.matched_bug_id == "CART_QTY_IGNORED"
            assert finding.status is FindingStatus.OPEN


class TestConstraints:
    def test_execution_attempt_is_unique_per_case(
        self, factory: sessionmaker[Session]
    ) -> None:
        with pytest.raises(IntegrityError), session_scope(factory) as session:
            target = _seed_target(session)
            run = _seed_run(session, target)
            case = _seed_case(session, run)
            session.add_all(
                [
                    Execution(case_id=case.id, run_id=run.id, attempt=1),
                    Execution(case_id=case.id, run_id=run.id, attempt=1),
                ]
            )

    def test_memory_key_is_unique_per_target_and_kind(
        self, factory: sessionmaker[Session]
    ) -> None:
        with pytest.raises(IntegrityError), session_scope(factory) as session:
            target = _seed_target(session)
            session.add_all(
                [
                    Memory(target_id=target.id, kind="flake", key="case_1", value={"runs": 3}),
                    Memory(target_id=target.id, kind="flake", key="case_1", value={"runs": 4}),
                ]
            )


class TestSessionScope:
    def test_rolls_back_on_exception(self, factory: sessionmaker[Session]) -> None:
        with session_scope(factory) as session:
            _seed_target(session)

        with pytest.raises(RuntimeError), session_scope(factory) as session:
            session.add(Target(name="discarded", base_url="http://localhost:9"))
            raise RuntimeError("boom")

        with session_scope(factory) as session:
            names = {row.name for row in session.query(Target).all()}
            assert names == {"Nimbus Supply"}

    def test_commits_on_success(self, factory: sessionmaker[Session]) -> None:
        with session_scope(factory) as session:
            _seed_target(session)

        with session_scope(factory) as session:
            assert session.query(Target).count() == 1


class TestVerdict:
    def test_stores_signals_and_evidence(self, factory: sessionmaker[Session]) -> None:
        with session_scope(factory) as session:
            target = _seed_target(session)
            run = _seed_run(session, target)
            case = _seed_case(session, run)
            execution = Execution(case_id=case.id, run_id=run.id)
            session.add(execution)
            session.flush()
            session.add(
                Verdict(
                    execution_id=execution.id,
                    decision=VerdictDecision.BUG,
                    confidence=0.82,
                    signals={"arithmetic": True, "judge": True, "visual": False},
                    evidence=["subtotal 19.99 != 59.97"],
                    model="meta/llama-3.3-70b-instruct",
                )
            )

        with session_scope(factory) as session:
            verdict = session.query(Verdict).one()
            # A disagreeing signal must be recorded rather than discarded: it is
            # the only data that calibrates the thresholds.
            assert verdict.signals["visual"] is False
            assert verdict.decision is VerdictDecision.BUG
            assert verdict.confidence == pytest.approx(0.82)
