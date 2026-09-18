"""Tests for the interactive chat channel.

Covers the ask/answer protocol, event emission, conversation recording, the
responder loop that answers messages typed while a run is in progress, and the
plain-language replies the agent composes from what it has observed.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from crucible.core.config import ProviderConfig, Settings
from crucible.core.events import Event, EventBus, EventType
from crucible.core.progress import FindingNote, ProgressTracker, RunProgress
from crucible.core.qa import ChatChannel, ChatMessage, MessageRole
from crucible.llm.chat import ConversationalResponder
from crucible.llm.providers import CompletionResult
from crucible.llm.sensitivity import ProviderName


class TestChatChannel:
    """Unit tests for the channel's core ask/answer flow."""

    @pytest.mark.asyncio
    async def test_ask_returns_answer(self) -> None:
        bus = EventBus("test_run")
        channel = ChatChannel(bus)

        # The ask coroutine should block until we answer.
        async def answer_later() -> None:
            await asyncio.sleep(0.05)
            msgs = channel.messages
            # Find the pending question.
            pending = [m for m in msgs if m.requires_answer]
            assert len(pending) == 1
            await channel.answer(pending[0].id, "skip login")

        task = asyncio.create_task(answer_later())
        result = await channel.ask(
            "I found a login wall",
            options=["Skip", "Provide credentials"],
        )
        await task
        assert result == "skip login"

    @pytest.mark.asyncio
    async def test_ask_emits_events(self) -> None:
        bus = EventBus("test_run")
        channel = ChatChannel(bus)

        events_received: list[str] = []

        def collector(event) -> None:
            events_received.append(event.type.value)

        bus.subscribe(collector)

        async def answer_later() -> None:
            await asyncio.sleep(0.05)
            pending = [m for m in channel.messages if m.requires_answer]
            await channel.answer(pending[0].id, "yes")

        task = asyncio.create_task(answer_later())
        await channel.ask("Need input?", options=["yes", "no"])
        await task

        assert "question.asked" in events_received
        assert "question.answered" in events_received

    @pytest.mark.asyncio
    async def test_answer_returns_false_for_unknown_id(self) -> None:
        bus = EventBus("test_run")
        channel = ChatChannel(bus)
        result = await channel.answer("msg_9999", "hello")
        assert result is False

    @pytest.mark.asyncio
    async def test_inform_does_not_block(self) -> None:
        bus = EventBus("test_run")
        channel = ChatChannel(bus)

        await channel.inform("Just so you know, I found 3 broken links.")
        assert len(channel.messages) == 1
        assert channel.messages[0].role == MessageRole.AGENT
        assert channel.messages[0].requires_answer is False

    @pytest.mark.asyncio
    async def test_conversation_recorded_in_order(self) -> None:
        bus = EventBus("test_run")
        channel = ChatChannel(bus)

        async def multi_turn() -> None:
            await asyncio.sleep(0.02)
            pending = [m for m in channel.messages if m.requires_answer]
            await channel.answer(pending[0].id, "admin secret123")

        task = asyncio.create_task(multi_turn())
        await channel.ask("What's the password?", options=["skip", "provide"])
        await task

        messages = channel.messages
        assert len(messages) == 2  # agent question + human answer
        # The first message is the agent asking.
        assert messages[0].role == MessageRole.AGENT
        assert messages[0].requires_answer is True
        # The second is the human replying.
        assert messages[1].role == MessageRole.HUMAN
        assert messages[1].in_reply_to == messages[0].id

    @pytest.mark.asyncio
    async def test_save_conversation_to_disk(self, tmp_path: Path) -> None:
        bus = EventBus("test_run")
        log_path = tmp_path / "conversation.json"
        channel = ChatChannel(bus, log_path=log_path)

        async def answer_later() -> None:
            await asyncio.sleep(0.02)
            pending = [m for m in channel.messages if m.requires_answer]
            await channel.answer(pending[0].id, "skip it")

        task = asyncio.create_task(answer_later())
        await channel.ask("Login required", options=["skip", "provide"])
        await task

        channel.save()
        assert log_path.exists()

        data = json.loads(log_path.read_text(encoding="utf-8"))
        assert isinstance(data, list)
        assert len(data) >= 2
        assert data[0]["role"] == "agent"
        assert data[0]["requires_answer"] is True
        assert data[1]["role"] == "human"
        assert data[1]["text"] == "skip it"


class TestResponder:
    """A message typed at a running scan must be answered, not just recorded.

    This is the gap the responder closes: messages were appended to the
    conversation and pushed onto an inbox that nothing ever read, so the
    operator saw their own words appear and then silence.
    """

    @pytest.mark.asyncio
    async def test_a_posted_message_gets_an_agent_reply(self) -> None:
        channel = ChatChannel(EventBus("test_run"))
        channel.start_responding(lambda text: f"ack:{text}")

        await channel.post("what have you found?")
        await channel.aclose_responding()

        assert [m.role for m in channel.messages] == [
            MessageRole.HUMAN,
            MessageRole.AGENT,
        ]
        assert channel.messages[-1].text == "ack:what have you found?"
        # A reply is not a question: the operator is not obliged to answer it.
        assert channel.messages[-1].requires_answer is False

    @pytest.mark.asyncio
    async def test_a_message_is_answered_even_if_queued_before_the_responder(self) -> None:
        """The operator can type the instant a scan starts, before any loop exists."""
        channel = ChatChannel(EventBus("test_run"))
        await channel.post("status")

        channel.start_responding(lambda text: f"ack:{text}")
        await channel.aclose_responding()

        assert channel.messages[-1].text == "ack:status"

    @pytest.mark.asyncio
    async def test_responding_flag_tracks_the_loop(self) -> None:
        channel = ChatChannel(EventBus("test_run"))
        assert channel.responding is False

        channel.start_responding(str.upper)
        assert channel.responding is True

        await channel.aclose_responding()
        assert channel.responding is False

    @pytest.mark.asyncio
    async def test_shutdown_is_idempotent(self) -> None:
        channel = ChatChannel(EventBus("test_run"))
        channel.start_responding(str.upper)

        await channel.aclose_responding()
        await channel.aclose_responding()

    @pytest.mark.asyncio
    async def test_starting_twice_keeps_a_single_responder(self) -> None:
        """The dashboard starts answering before the pipeline sees the channel."""
        channel = ChatChannel(EventBus("test_run"))
        channel.start_responding(lambda text: f"first:{text}")
        channel.start_responding(lambda text: f"second:{text}")

        await channel.post("hi")
        await channel.aclose_responding()

        assert channel.messages[-1].text == "first:hi"

    @pytest.mark.asyncio
    async def test_an_async_reply_is_awaited(self) -> None:
        """The model-backed responder is a coroutine, not a plain function."""
        channel = ChatChannel(EventBus("test_run"))

        async def reply(text: str) -> str:
            await asyncio.sleep(0)
            return f"async:{text}"

        channel.start_responding(reply)
        await channel.post("hi")
        await channel.aclose_responding()

        assert channel.messages[-1].text == "async:hi"

    @pytest.mark.asyncio
    async def test_the_responders_resources_are_released_with_the_run(self) -> None:
        """Provider connections must not outlive the run that opened them."""
        channel = ChatChannel(EventBus("test_run"))
        closed: list[bool] = []

        async def aclose() -> None:
            closed.append(True)

        channel.start_responding(str.upper, aclose=aclose)
        await channel.aclose_responding()

        assert closed == [True]

    @pytest.mark.asyncio
    async def test_typing_answers_a_pending_question(self) -> None:
        """An operator who types instead of clicking a button must not be ignored."""
        channel = ChatChannel(EventBus("test_run"))

        async def type_later() -> None:
            await asyncio.sleep(0.02)
            await channel.post("admin hunter2")

        task = asyncio.create_task(type_later())
        answer = await channel.ask("Login required", options=["skip", "provide"])
        await task

        assert answer == "admin hunter2"
        assert [m.role for m in channel.messages] == [
            MessageRole.AGENT,
            MessageRole.HUMAN,
        ]


class TestConversationLog:
    """The conversation must survive the process, even when built before the run id exists."""

    @pytest.mark.asyncio
    async def test_log_path_can_be_attached_after_construction(self, tmp_path: Path) -> None:
        channel = ChatChannel(EventBus("test_run"))
        # Saving with no path configured must be a no-op, not a crash: the
        # channel is usable without a log.
        channel.save()

        log_path = tmp_path / "conversation.json"
        channel.enable_log(log_path)
        await channel.inform("hello")
        channel.save()

        data = json.loads(log_path.read_text(encoding="utf-8"))
        assert [message["role"] for message in data] == ["agent"]


class _FakeModel:
    """A model client that answers however a test needs it to."""

    def __init__(
        self,
        text: str = "A model answer.",
        *,
        raises: Exception | None = None,
        sleeps: float = 0.0,
    ) -> None:
        self.config = ProviderConfig(
            ProviderName.NVIDIA, "http://models.test/v1", "fake-1", "key"
        )
        self.calls: list[list[dict[str, str]]] = []
        self._text = text
        self._raises = raises
        self._sleeps = sleeps
        self.closed = False

    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> CompletionResult:
        self.calls.append(messages)
        if self._sleeps:
            await asyncio.sleep(self._sleeps)
        if self._raises is not None:
            raise self._raises
        return CompletionResult(
            text=self._text,
            model="fake-1",
            prompt_tokens=1,
            completion_tokens=1,
            latency_ms=1.0,
        )

    async def aclose(self) -> None:
        self.closed = True


def _no_provider_settings() -> Settings:
    return Settings(
        gemini_api_key="",
        nvidia_api_key="",
        enable_gemini=False,
        enable_nvidia=False,
        enable_ollama=False,
        _env_file=None,
    )


class TestConversationalResponder:
    """A model is the preferred reply; the grounded one is the floor beneath it.

    Every test here is about the *fallback*, because that is the property that
    decides whether an operator is answered at all when a free-tier endpoint has
    retired the model id, rate limited the key, or been blocked by the network.
    """

    def _progress(self) -> RunProgress:
        progress = RunProgress(target="http://shop.test", stage="crawling the site")
        progress.routes = 22
        return progress

    @pytest.mark.asyncio
    async def test_uses_the_model_answer_when_one_arrives(self) -> None:
        model = _FakeModel("I've mapped 22 pages so far and I'm still crawling.")
        responder = ConversationalResponder(
            self._progress(), settings=_no_provider_settings(), providers=[model]
        )

        assert await responder("how's it going?") == (
            "I've mapped 22 pages so far and I'm still crawling."
        )

    @pytest.mark.asyncio
    async def test_the_model_is_told_what_the_run_has_observed(self) -> None:
        model = _FakeModel()
        progress = self._progress()
        progress.note(FindingNote(title="The cart charges for one item", cause="Quantity ignored."))
        responder = ConversationalResponder(
            progress, settings=_no_provider_settings(), providers=[model]
        )

        await responder("what happened?")

        system = model.calls[0][0]
        assert system["role"] == "system"
        assert "Target: http://shop.test" in system["content"]
        assert "The cart charges for one item" in system["content"]
        assert model.calls[0][1] == {"role": "user", "content": "what happened?"}

    @pytest.mark.asyncio
    async def test_falls_back_when_no_provider_is_configured(self) -> None:
        responder = ConversationalResponder(
            self._progress(), settings=_no_provider_settings()
        )

        reply = await responder("how far along are you?")

        assert "22 page(s) mapped" in reply

    @pytest.mark.asyncio
    async def test_falls_back_when_the_provider_fails(self) -> None:
        """A retired model id answers 410; the operator must not see silence."""
        model = _FakeModel(raises=RuntimeError("410 Gone"))
        responder = ConversationalResponder(
            self._progress(), settings=_no_provider_settings(), providers=[model]
        )

        assert "22 page(s) mapped" in await responder("how far along are you?")

    @pytest.mark.asyncio
    async def test_falls_back_when_the_model_times_out(self) -> None:
        """A scan must not stall because a model endpoint hangs."""
        model = _FakeModel(sleeps=5.0)
        responder = ConversationalResponder(
            self._progress(),
            settings=_no_provider_settings(),
            providers=[model],
            timeout_seconds=0.05,
        )

        assert "22 page(s) mapped" in await responder("how far along are you?")

    @pytest.mark.asyncio
    async def test_falls_back_when_the_model_returns_nothing(self) -> None:
        responder = ConversationalResponder(
            self._progress(), settings=_no_provider_settings(), providers=[_FakeModel("   ")]
        )
        assert "22 page(s) mapped" in await responder("how far along are you?")

    @pytest.mark.asyncio
    async def test_a_failed_provider_is_not_retried_for_every_message(self) -> None:
        """Otherwise each message costs the operator the full timeout, for nothing."""
        model = _FakeModel(raises=RuntimeError("410 Gone"))
        responder = ConversationalResponder(
            self._progress(), settings=_no_provider_settings(), providers=[model]
        )

        await responder("first")
        await responder("second")

        assert len(model.calls) == 1

    @pytest.mark.asyncio
    async def test_a_working_provider_is_still_used_after_a_dead_one(self) -> None:
        dead = _FakeModel(raises=RuntimeError("410 Gone"))
        alive = _FakeModel("The live one answered.")
        responder = ConversationalResponder(
            self._progress(), settings=_no_provider_settings(), providers=[dead, alive]
        )

        assert await responder("hello") == "The live one answered."

    @pytest.mark.asyncio
    async def test_providers_are_closed_at_the_end_of_the_run(self) -> None:
        model = _FakeModel()
        responder = ConversationalResponder(
            self._progress(), settings=_no_provider_settings(), providers=[model]
        )

        await responder("hello")
        await responder.aclose()

        assert model.closed is True


class TestRunProgressReplies:
    """Replies come from observed state; nothing is invented to sound helpful."""

    def test_status_reports_stage_and_counts(self) -> None:
        progress = RunProgress(target="http://shop.test", stage="crawling the site")
        progress.routes = 22

        reply = progress.reply("how far along are you?")

        assert "crawling the site" in reply
        assert "22 page(s) mapped" in reply

    def test_findings_are_listed_by_title(self) -> None:
        progress = RunProgress(target="http://shop.test")
        progress.note(FindingNote(title="The cart charges for one item"))

        reply = progress.reply("any bugs?")

        assert "1 finding(s)" in reply
        assert "charges for one item" in reply

    def test_a_clean_finished_run_says_so(self) -> None:
        progress = RunProgress(target="http://shop.test", routes=12, finished=True)
        assert "No findings" in progress.reply("what have you found")

    def test_explain_gives_the_cause_and_the_fix(self) -> None:
        progress = RunProgress(target="http://shop.test")
        progress.note(
            FindingNote(
                title="The cart charges for one item",
                cause="It adds prices but never multiplies by quantity.",
                fix="Multiply price by quantity per line.",
            )
        )

        reply = progress.reply("why?")

        assert "never multiplies by quantity" in reply
        assert "Multiply price by quantity" in reply

    def test_explain_without_detail_points_at_the_evidence(self) -> None:
        progress = RunProgress(target="http://shop.test")
        progress.note(FindingNote(title="Wrong total"))
        assert "Findings panel" in progress.reply("why")

    def test_a_finding_is_upgraded_rather_than_duplicated(self) -> None:
        """The event stream carries the title; triage later adds cause and fix."""
        progress = RunProgress(target="http://shop.test")
        progress.note(FindingNote(title="Wrong total"))
        progress.note(FindingNote(title="Wrong total", cause="Quantity ignored.", fix="Multiply."))

        assert len(progress.notes) == 1
        assert progress.notes[0].cause == "Quantity ignored."

    def test_an_unrecognised_message_admits_the_limitation(self) -> None:
        progress = RunProgress(target="http://shop.test", stage="crawling the site")

        reply = progress.reply("do you think the design here is intentional?")

        assert "can't hold a conversation" in reply
        assert "design here is intentional" in reply

    def test_score_without_a_manifest_says_there_is_nothing_to_score(self) -> None:
        progress = RunProgress(target="http://shop.test")
        assert "no score to report" in progress.reply("what's your accuracy?")

    def test_score_reports_recall_and_precision(self) -> None:
        progress = RunProgress(target="http://shop.test", recall=1.0, precision=1.0)
        reply = progress.reply("score")
        assert "100%" in reply

    def test_a_long_message_is_not_echoed_whole(self) -> None:
        progress = RunProgress(target="http://shop.test")
        reply = progress.reply("x" * 400)
        assert len(reply) < 700


class TestProgressTracker:
    """The snapshot is derived from the event stream, so it cannot drift."""

    def test_tracks_a_run_from_its_events(self) -> None:
        progress = RunProgress(target="http://shop.test")
        tracker = ProgressTracker(progress)

        def emit(event_type: EventType, **payload: object) -> None:
            tracker(Event(type=event_type, run_id="run_1", payload=dict(payload)))

        emit(EventType.RECON_STARTED)
        assert progress.stage == "crawling the site"

        emit(EventType.RECON_FINISHED, routes=22, forms=2, failed=2)
        assert (progress.routes, progress.forms, progress.broken) == (22, 2, 2)

        emit(EventType.PLAN_CASE_ADDED, check_id="c1", risk=5)
        assert progress.cases_total == 1

        emit(EventType.EXEC_CASE_FINISHED, check_id="c1", duration_ms=41.0)
        assert progress.cases_run == 1

        emit(EventType.BROWSER_FINISHED, pages=8, console_errors=0, video_key="run_1/video/a.webm")
        assert progress.video_key == "run_1/video/a.webm"

        emit(EventType.FINDING_RECORDED, title="Wrong total", severity="high")
        assert [note.title for note in progress.notes] == ["Wrong total"]

        emit(EventType.RUN_FINISHED, decision="BUG", findings=1, benchmark={"recall": 0.9})
        assert progress.finished is True
        assert progress.recall == pytest.approx(0.9)

    def test_a_question_during_the_crawl_reports_work_not_silence(self) -> None:
        """Mid-recon there are no routes yet, but there is visible progress.

        Answering "nothing has come back from the target yet" while the crawl is
        visibly running was worse than useless: it told the operator the agent
        was stuck at the exact moment it was working hardest.
        """
        progress = RunProgress(target="http://shop.test")
        tracker = ProgressTracker(progress)
        tracker(Event(type=EventType.RECON_STARTED, run_id="run_1"))
        for _ in range(5):
            tracker(Event(type=EventType.RECON_PAGE_VISITED, run_id="run_1"))

        reply = progress.reply("how far along are you?")

        assert "crawling the site" in reply
        assert "looked at 5 page(s) already" in reply
        assert "Nothing has come back" not in reply

    def test_malformed_event_payloads_do_not_crash_the_tracker(self) -> None:
        """A payload shape change must degrade the wording, not the run."""
        progress = RunProgress(target="http://shop.test")
        tracker = ProgressTracker(progress)

        tracker(Event(type=EventType.RECON_FINISHED, run_id="run_1", payload={"routes": "lots"}))
        tracker(Event(type=EventType.RUN_FINISHED, run_id="run_1", payload={"benchmark": None}))

        assert progress.routes == 0
        assert progress.recall is None


class TestChatMessageModel:
    """Ensure the serialisation contract is stable."""

    def test_as_dict_minimal(self) -> None:
        msg = ChatMessage(
            id="msg_0001",
            role=MessageRole.AGENT,
            text="Hello",
        )
        d = msg.as_dict()
        assert d["id"] == "msg_0001"
        assert d["role"] == "agent"
        assert d["text"] == "Hello"
        assert "options" not in d
        assert "requires_answer" not in d
        assert "in_reply_to" not in d

    def test_as_dict_full(self) -> None:
        msg = ChatMessage(
            id="msg_0002",
            role=MessageRole.HUMAN,
            text="admin pass",
            options=["skip"],
            requires_answer=True,
            in_reply_to="msg_0001",
        )
        d = msg.as_dict()
        assert d["options"] == ["skip"]
        assert d["requires_answer"] is True
        assert d["in_reply_to"] == "msg_0001"
