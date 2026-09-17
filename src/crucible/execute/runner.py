"""Runs checks against a live target and persists executions.

One check invocation becomes one :class:`Execution` row. Verdicts are *not*
written here: observation and judgement are separate layers, so the oracle can
be re-run over stored executions without touching the target again.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from sqlalchemy.orm import Session

from crucible.core.events import EventBus, EventType
from crucible.core.logging import get_logger
from crucible.core.qa import ChatChannel
from crucible.execute.checks import CHECK_REGISTRY, CONTEXT_CHECK_REGISTRY, CheckResult
from crucible.execute.client import AppClient
from crucible.recon.scout import AppMapData
from crucible.store.models import CaseStatus, Classification, Execution, TestCase

logger = get_logger(__name__)


@dataclass(slots=True)
class ExecutionOutcome:
    """What the runner hands back per case, for the CLI and tests."""

    case_id: str
    check_id: str
    result: CheckResult
    execution_id: str


class CheckRunner:
    """Executes checks in registry order and records executions."""

    def __init__(
        self,
        client: AppClient,
        session: Session,
        bus: EventBus,
        run_id: str,
        *,
        app_map: AppMapData | None = None,
        questioner: ChatChannel | None = None,
    ) -> None:
        self._client = client
        self._session = session
        self._bus = bus
        self._run_id = run_id
        #: Recon's findings, handed to the checks that cannot probe honestly
        #: without them (see CONTEXT_CHECK_REGISTRY).
        self._app_map = app_map
        #: Interactive chat channel so checks can ask the human operator
        #: for credentials or decisions when the bot is stuck.
        self._questioner = questioner

    async def run_case(self, case: TestCase) -> ExecutionOutcome:
        """Execute one case's check, persist the execution, return the outcome."""
        check_id = case.requirement_ref or ""
        if check_id not in CHECK_REGISTRY and check_id not in CONTEXT_CHECK_REGISTRY:
            raise ValueError(f"Case {case.id} references unknown check {check_id!r}")

        await self._bus.emit(
            EventType.EXEC_CASE_STARTED, case_id=case.id, check_id=case.requirement_ref
        )

        started = time.perf_counter()
        try:
            if check_id in CONTEXT_CHECK_REGISTRY:
                result: CheckResult = await CONTEXT_CHECK_REGISTRY[check_id](
                    self._client, self._app_map, self._questioner
                )
            else:
                result = await CHECK_REGISTRY[check_id](
                    self._client, self._questioner
                )
        except Exception as exc:
            result = CheckResult(
                check_id=case.requirement_ref or case.id,
                lane="error",
                observation="check raised an exception",
                facts={},
                error=f"{type(exc).__name__}: {exc}",
            )
        duration_ms = (time.perf_counter() - started) * 1000.0

        # A check error is a transport/arrangement failure, not an app fault.
        # Classification is deliberately left UNKNOWN here: the oracle decides
        # it after weighing evidence, and collapsing "we could not run this"
        # into APP_BUG is how automated testers manufacture false positives.
        classification = Classification.UNKNOWN

        execution = Execution(
            case_id=case.id,
            run_id=self._run_id,
            attempt=1,
            status=CaseStatus.FAILED if result.error else CaseStatus.PENDING,
            classification=classification,
            duration_ms=duration_ms,
            artifacts={"facts": result.facts, "observation": result.observation},
        )
        self._session.add(execution)
        self._session.flush()  # assign execution.id before verdicts reference it

        case.status = CaseStatus.ERRORED if result.error else CaseStatus.RUNNING

        await self._bus.emit(
            EventType.EXEC_CASE_FINISHED,
            case_id=case.id,
            check_id=case.requirement_ref,
            duration_ms=round(duration_ms, 1),
            had_error=bool(result.error),
        )
        logger.debug(
            "check_finished check=%s dur=%.0fms error=%s",
            case.requirement_ref,
            duration_ms,
            result.error,
        )
        return ExecutionOutcome(
            case_id=case.id, check_id=case.requirement_ref or "", result=result, execution_id=execution.id
        )

    async def run_all(self, cases: list[TestCase]) -> list[ExecutionOutcome]:
        await self._bus.emit(EventType.EXEC_STARTED, cases=len(cases))
        outcomes = []
        for case in cases:
            outcomes.append(await self.run_case(case))
            self._session.commit()
        await self._bus.emit(EventType.EXEC_FINISHED, executed=len(outcomes))
        return outcomes
