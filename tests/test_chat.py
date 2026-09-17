"""Tests for the interactive chat channel.

Covers the ask/answer protocol, event emission, conversation recording,
and the endpoint that the dashboard calls to submit replies.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from crucible.core.events import EventBus, EventType
from crucible.core.qa import ChatChannel, ChatMessage, MessageRole


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
