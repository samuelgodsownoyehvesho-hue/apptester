"""Pipeline tests against the behavioural twin: full stack, zero network.

These run the *real* pipeline — recon over a canned fetcher, planning,
execution against the in-memory target, oracle signals, triage, benchmark
scoring — with the only fake parts being the transports. A failure here is a
failure in crucible's own code, not in a fixture mismatch, which is what makes
these tests worth their runtime.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.orm import Session, sessionmaker

from crucible.benchmark.score import Manifest, fetch_manifest, score
from crucible.core.config import Settings
from crucible.core.events import EventBus
from crucible.core.qa import ChatChannel
from crucible.execute.browser import BrowserUnavailable
from crucible.execute.checks import CheckResult
from crucible.execute.client import ApiResponse, AppClient, reset_target
from crucible.pipeline import run_pipeline
from crucible.recon.fetcher import FetchResult, HttpFetcher
from crucible.store.artifacts import ArtifactStore
from crucible.store.db import init_db, make_engine, make_session_factory
from crucible.store.models import Finding, VerdictDecision
from crucible.triage.cluster import cluster
from tests.fake_target import FakeGuineaPig

BUGGED = {
    "CART_QTY_IGNORED",
    "TRUNCATED_ROUNDING",
    "EMPTY_CART_STALE_TOTAL",
    "DISCOUNT_STACKS",
    "LEXICOGRAPHIC_SORT",
    "PAGINATION_OVERLAP",
    "CASE_SENSITIVE_SEARCH",
    "CHECKOUT_ACCEPTS_NEGATIVE_QTY",
    "BROKEN_FOOTER_LINK",
}


class StaticSiteFetcher(HttpFetcher):
    """Serves a tiny HTML site so recon runs without network.

    Subclasses the real fetcher so the type matches the pipeline signature;
    ``fetch`` never touches HTTP.
    """

    def __init__(self) -> None:
        super().__init__()

    async def fetch(self, url: str) -> FetchResult:
        page = (
            "<html><head><title>Shop</title></head><body>"
            '<a href="/catalog">Catalog</a>'
            '<a href="/cart">Cart</a>'
            '<a href="/returns">Returns</a>'
            "</body></html>"
        )
        return FetchResult(url=url, status=200, content_type="text/html", text=page)

    async def aclose(self) -> None:
        return None


class UnlabelledSiteFetcher(HttpFetcher):
    """Serves a page whose form control has no accessible name.

    The defect here lives in markup, not in an API, so it is the fetcher that
    supplies it while the twin's manifest declares it active.
    """

    async def fetch(self, url: str) -> FetchResult:
        page = (
            "<html><head><title>Checkout</title></head><body>"
            "<form action='/order' method='post'>"
            "<input type='email' name='email' placeholder='you@example.com'>"
            "</form></body></html>"
        )
        return FetchResult(url=url, status=200, content_type="text/html", text=page)

    async def aclose(self) -> None:
        return None


class MapOnlyClient(AppClient):
    """Returns 404 for everything; used for recon-only plumbing tests."""

    async def get(self, path: str, *, params: dict[str, str] | None = None) -> ApiResponse:
        return ApiResponse(status=404, text="not found")

    async def post(self, path: str, json_body: Any) -> ApiResponse:
        return ApiResponse(status=404, text="not found")

    async def delete(self, path: str, *, params: dict[str, str] | None = None) -> ApiResponse:
        return ApiResponse(status=404, text="not found")

    async def aclose(self) -> None:
        return None


@pytest.fixture()
def db(tmp_path: Path) -> Iterator[sessionmaker[Session]]:
    engine = make_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    yield make_session_factory(engine)
    engine.dispose()


@pytest.fixture()
def artifacts(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "artifacts")


def _settings() -> Settings:
    return Settings(
        gemini_api_key="",
        nvidia_api_key="",
        enable_gemini=False,
        enable_nvidia=False,
        _env_file=None,
    )


@pytest.mark.asyncio
async def test_buggy_target_is_scored_as_buggy(
    db: sessionmaker[Session], artifacts: ArtifactStore
) -> None:
    """Every simulated defect should be found, and none other than those."""
    fake = FakeGuineaPig(simulate=BUGGED)
    result = await run_pipeline(
        "http://guinea.test",
        _settings(),
        db,
        artifacts,
        fetcher=StaticSiteFetcher(),
        app_client=fake,
    )

    assert result.decision is VerdictDecision.BUG
    assert result.benchmark is not None
    report = result.benchmark

    # These 9 are reachable without a browser. The tenth, an unlabelled form
    # control, needs markup that recon has to observe -- and this canned site
    # serves no forms, so it is covered by its own test instead.
    assert sorted(report.true_positives) == sorted(BUGGED)
    assert report.false_positives == []
    assert report.recall == pytest.approx(1.0)
    assert report.precision == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_clean_target_produces_no_findings(
    db: sessionmaker[Session], artifacts: ArtifactStore
) -> None:
    """A defect-free target must yield zero findings — the precision trap."""
    fake = FakeGuineaPig(simulate=set())
    result = await run_pipeline(
        "http://guinea.test",
        _settings(),
        db,
        artifacts,
        fetcher=StaticSiteFetcher(),
        app_client=fake,
    )

    assert result.decision is VerdictDecision.NOT_A_BUG
    assert result.findings == 0
    assert result.benchmark is not None
    assert result.benchmark.missed == []
    assert result.benchmark.false_positives == []
    assert result.benchmark.recall == pytest.approx(1.0)
    assert result.benchmark.precision == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_single_mutant_is_isolated(
    db: sessionmaker[Session], artifacts: ArtifactStore
) -> None:
    """One active defect: found, and nothing else reported."""
    fake = FakeGuineaPig(simulate={"CART_QTY_IGNORED"})
    result = await run_pipeline(
        "http://guinea.test",
        _settings(),
        db,
        artifacts,
        fetcher=StaticSiteFetcher(),
        app_client=fake,
    )

    assert result.decision is VerdictDecision.BUG
    assert result.benchmark is not None
    assert result.benchmark.true_positives == ["CART_QTY_IGNORED"]
    assert result.benchmark.false_positives == []
    # Recall is measured over the mutants that are *active in this run*, not
    # over the whole catalogue: the other seven are not present, so counting
    # them as misses would understate recall against a target that was asked
    # to exhibit exactly one defect.
    assert result.benchmark.declared_active == 1
    assert result.benchmark.recall == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_artifacts_are_written(
    db: sessionmaker[Session], artifacts: ArtifactStore
) -> None:
    fake = FakeGuineaPig(simulate=set())
    result = await run_pipeline(
        "http://guinea.test",
        _settings(),
        db,
        artifacts,
        fetcher=StaticSiteFetcher(),
        app_client=fake,
    )

    keys = {ref.key for ref in artifacts.list_run(result.run_id)}
    assert any("app_map" in key for key in keys)
    assert any("benchmark" in key for key in keys)


@pytest.mark.asyncio
async def test_findings_persist_with_evidence(
    db: sessionmaker[Session], artifacts: ArtifactStore
) -> None:
    fake = FakeGuineaPig(simulate={"DISCOUNT_STACKS"})
    result = await run_pipeline(
        "http://guinea.test",
        _settings(),
        db,
        artifacts,
        fetcher=StaticSiteFetcher(),
        app_client=fake,
    )

    with db() as session:
        rows = session.query(Finding).filter(Finding.run_id == result.run_id).all()
    assert len(rows) == 1
    finding = rows[0]
    assert finding.matched_bug_id == "DISCOUNT_STACKS"
    signals = finding.evidence["signals"]
    assert any(signal["violated"] is True for signal in signals)


@pytest.mark.asyncio
async def test_recon_derived_signal_reaches_findings(
    db: sessionmaker[Session], artifacts: ArtifactStore
) -> None:
    """A defect recon recorded must reach a finding and be scored.

    Regression: the pipeline rebuilt the oracle's aggregation by hand and
    dropped every recon-derived signal, so an unlabelled control that recon had
    already observed was reported as a miss even though the fact was on disk.
    """
    fake = FakeGuineaPig(simulate={"UNLABELED_CHECKOUT_INPUT"})
    result = await run_pipeline(
        "http://guinea.test",
        _settings(),
        db,
        artifacts,
        fetcher=UnlabelledSiteFetcher(),
        app_client=fake,
    )

    with db() as session:
        rows = session.query(Finding).filter(Finding.run_id == result.run_id).all()
    assert [row.matched_bug_id for row in rows] == ["UNLABELED_CHECKOUT_INPUT"]

    assert result.decision is VerdictDecision.BUG
    assert result.benchmark is not None
    assert result.benchmark.true_positives == ["UNLABELED_CHECKOUT_INPUT"]
    assert result.benchmark.recall == pytest.approx(1.0)
    assert result.benchmark.precision == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_findings_explain_themselves_to_a_non_author(
    db: sessionmaker[Session], artifacts: ArtifactStore
) -> None:
    """Every finding must say what broke and what to change.

    Regression guard: findings used to read "<BUG_ID>: <check name>
    contradicts the declared invariant", which tells the person who has to fix
    it nothing at all.
    """
    fake = FakeGuineaPig(simulate=BUGGED)
    result = await run_pipeline(
        "http://guinea.test",
        _settings(),
        db,
        artifacts,
        fetcher=StaticSiteFetcher(),
        app_client=fake,
    )

    with db() as session:
        rows = session.query(Finding).filter(Finding.run_id == result.run_id).all()

    assert rows
    for row in rows:
        assert row.title, row.matched_bug_id
        assert row.root_cause, f"{row.matched_bug_id} explains no cause"
        assert row.suggested_fix, f"{row.matched_bug_id} suggests no fix"

        # Oracle vocabulary must not reach a human-facing report.
        prose = row.title.lower()
        for banned in ("invariant", "contradicts", "signal:", "check_id"):
            assert banned not in prose, f"jargon {banned!r} in {row.title!r}"

    titles = {row.matched_bug_id: row.title for row in rows}
    assert titles["CART_QTY_IGNORED"] == (
        "The cart charges for one item when you order several"
    )


@pytest.mark.asyncio
async def test_reset_is_optional_for_targets_that_lack_it() -> None:
    """A target with no reset endpoint must not fail the run."""
    client = MapOnlyClient()
    assert await reset_target(client) is False
    await client.aclose()


@pytest.mark.asyncio
async def test_a_run_does_not_inherit_the_previous_runs_state(
    db: sessionmaker[Session], artifacts: ArtifactStore
) -> None:
    """Two runs against the same target must report the same numbers.

    Regression: nothing ever called the target's own reset, so a cart left
    behind by an earlier run changed the figures a later run reported and the
    same code could reach a different verdict purely because of run order.
    """
    fake = FakeGuineaPig(simulate={"CART_QTY_IGNORED", "DISCOUNT_STACKS"})

    # Dirty the target the way a previous run would have left it: a leftover
    # line, and a discount compounded by being applied twice.
    await fake.post(
        "/api/cart",
        {"productId": "leftover", "name": "Leftover", "price": 7.50, "quantity": 4},
    )
    await fake.post("/api/cart/discount", {"code": "SAVE10"})
    await fake.post("/api/cart/discount", {"code": "SAVE10"})
    assert fake.discount_rate > 0.1, "precondition: the target starts dirty"

    result = await run_pipeline(
        "http://guinea.test",
        _settings(),
        db,
        artifacts,
        fetcher=StaticSiteFetcher(),
        app_client=fake,
    )

    arithmetic = next(
        outcome
        for outcome in result.outcomes
        if outcome.check_id == "cart_quantity_arithmetic" and outcome.violated is True
    )
    # 2 x $10.00 with a clean discount rate: the unit-price defect reports the
    # unit price exactly. A leaked discount would have shown less than $10.00.
    assert arithmetic.evidence["expected_total"] == pytest.approx(20.0)
    assert arithmetic.evidence["reported_total"] == pytest.approx(10.0)


@pytest.mark.asyncio
async def test_a_missing_browser_does_not_fail_the_run(
    db: sessionmaker[Session], artifacts: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The browser lane is optional; a missing browser degrades, never breaks.

    A real browser may be impossible to install on a locked-down network. The
    checks have already produced findings worth reporting by the time this lane
    runs, so losing them would be the wrong trade.
    """

    async def unavailable(*_args: object, **_kwargs: object) -> object:
        raise BrowserUnavailable("no usable browser; tried bundled chromium, msedge")

    monkeypatch.setattr("crucible.pipeline.record_walk", unavailable)

    fake = FakeGuineaPig(simulate={"CART_QTY_IGNORED"})
    result = await run_pipeline(
        "http://guinea.test",
        _settings(),
        db,
        artifacts,
        fetcher=StaticSiteFetcher(),
        app_client=fake,
    )

    assert result.browser is None
    assert result.decision is VerdictDecision.BUG
    assert result.benchmark is not None
    assert result.benchmark.true_positives == ["CART_QTY_IGNORED"]


@pytest.mark.asyncio
async def test_browser_lane_can_be_switched_off(
    db: sessionmaker[Session], artifacts: ArtifactStore
) -> None:
    fake = FakeGuineaPig(simulate=set())
    result = await run_pipeline(
        "http://guinea.test",
        _settings(),
        db,
        artifacts,
        fetcher=StaticSiteFetcher(),
        app_client=fake,
        with_browser=False,
    )

    assert result.browser is None
    assert result.findings == 0


@pytest.mark.asyncio
async def test_an_operator_message_is_answered_during_a_run(
    db: sessionmaker[Session], artifacts: ArtifactStore
) -> None:
    """A message typed at a running scan gets a reply, and the log keeps it.

    This is the exact defect this test exists for: the operator's message was
    appended to the conversation and pushed onto the channel's inbox, and
    nothing ever read that inbox -- so their own words appeared on screen and
    the agent was silent forever after. The reply now has to be grounded in what
    the run has actually observed, and the conversation has to reach the disk
    rather than living only in memory.
    """
    fake = FakeGuineaPig(simulate={"CART_QTY_IGNORED"})
    bus = EventBus("live_0001")
    channel = ChatChannel(bus)
    # Posted before the run starts, which is the hardest ordering: the responder
    # does not exist yet, so this has to be queued rather than dropped.
    await channel.post("how far along are you?")

    result = await run_pipeline(
        "http://guinea.test",
        _settings(),
        db,
        artifacts,
        fetcher=StaticSiteFetcher(),
        app_client=fake,
        bus=bus,
        channel=channel,
    )

    assert result.routes > 0
    conversation = result.conversation
    asked_at = next(
        index
        for index, message in enumerate(conversation)
        if message["role"] == "human" and message["text"] == "how far along are you?"
    )
    # The narration the pipeline emits on its own never mentions mapped pages in
    # these words, so this can only be the grounded reply.
    replies = [
        message["text"]
        for message in conversation[asked_at + 1 :]
        if message["role"] == "agent" and "page(s) mapped" in message["text"]
    ]
    assert replies, f"the operator's message was never answered: {conversation}"

    # The conversation has to outlive the process that produced it.
    log_path = artifacts.root / result.run_id / "conversation.json"
    assert log_path.exists()
    saved = json.loads(log_path.read_text(encoding="utf-8"))
    assert len(saved) == len(conversation)


def test_score_flags_label_without_violated_signal() -> None:
    """A finding claiming a bug id without a violated signal is a miss."""
    manifest = Manifest(
        all_ids=frozenset({"X"}), active_ids=frozenset({"X"}), selection="X"
    )
    # Construct the claimed evidence by hand: cluster() only emits drafts for
    # violated signals, so simulate a stale label on a persisted finding.
    outcome = CheckResult(
        check_id="cart_quantity_arithmetic",
        lane="arithmetic",
        observation="",
        facts={"expected_total": 20.0, "reported_total": 20.0},
    )
    from crucible.oracle.signals import SIGNALS_BY_CHECK

    signals = [signal(outcome) for signal in SIGNALS_BY_CHECK["cart_quantity_arithmetic"]]
    finding = Finding(
        run_id="run_x",
        title="claimed",
        matched_bug_id="X",
        evidence={"signals": [signal.as_dict() for signal in signals]},
    )

    report = score(manifest, [finding], signals)
    assert report.true_positives == []
    assert report.missed == ["X"]
    assert report.found == 0


@pytest.mark.asyncio
async def test_fetch_manifest_requires_json() -> None:
    client = MapOnlyClient()
    with pytest.raises(ValueError, match="manifest unavailable"):
        await fetch_manifest(client)
    await client.aclose()


def test_cluster_deduplicates_shared_evidence() -> None:
    """Two violated signals on one defect collapse to one finding."""
    result = CheckResult(
        check_id="cart_quantity_arithmetic",
        lane="arithmetic",
        observation="",
        facts={"expected_total": 20.0, "reported_total": 10.0},
    )
    from crucible.oracle.signals import SIGNALS_BY_CHECK

    signals = [signal(result) for signal in SIGNALS_BY_CHECK["cart_quantity_arithmetic"]]
    drafts = cluster(signals)
    assert len(drafts) == 1
    assert drafts[0].suspected_bug_id == "CART_QTY_IGNORED"
