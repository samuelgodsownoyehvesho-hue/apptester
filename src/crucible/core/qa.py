"""Interactive chat between the agent and the human operator.

During a run the bot may need human input — credentials for a login wall, a
decision about a CAPTCHA, confirmation before a destructive action.  The
:class:`ChatChannel` is the protocol: the bot :meth:`~ChatChannel.ask` a
question and blocks until the human answers via the dashboard (or the CLI).

Design goals:

* **Non-blocking for the rest of the system.**  The channel never holds a
  lock the event loop cares about.  The ``ask`` coroutine yields control
  via an ``asyncio.Event`` that only wakes when the answer arrives.
* **Recorded.**  Every question and answer is appended to a list that the
  API serialises to ``conversation.json`` in the run's artifact directory.
* **Typed.**  Questions carry an id, the prompt text, optional choices, and
  the human-readable context that explains *why* the bot is stuck.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from crucible.core.events import EventBus, EventType
from crucible.core.logging import get_logger

logger = get_logger(__name__)


class MessageRole(StrEnum):
    AGENT = "agent"
    HUMAN = "human"


@dataclass(slots=True)
class ChatMessage:
    """One utterance in the conversation."""

    id: str
    role: MessageRole
    text: str
    ts: datetime = field(default_factory=lambda: datetime.now(UTC))
    #: For agent messages that propose choices (e.g. ["skip", "provide credentials"]).
    options: list[str] = field(default_factory=list)
    #: If true the human *must* answer before the run can continue.
    requires_answer: bool = False
    #: Only set on human replies.
    in_reply_to: str | None = None

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "role": self.role.value,
            "text": self.text,
            "ts": self.ts.isoformat(),
        }
        if self.options:
            d["options"] = self.options
        if self.requires_answer:
            d["requires_answer"] = True
        if self.in_reply_to:
            d["in_reply_to"] = self.in_reply_to
        return d


class ChatChannel:
    """Bidirectional channel between the running pipeline and the human.

    The pipeline side calls :meth:`ask` which returns an answer once the
    human responds.  The API side calls :meth:`answer` with the human's
    reply, which wakes the blocked ``ask``.

    A single :class:`ChatChannel` is shared across the entire run.
    """

    def __init__(self, bus: EventBus, *, log_path: Path | None = None) -> None:
        self._bus = bus
        self._log_path = log_path
        self._counter = 0
        self._pending: dict[str, asyncio.Event] = {}
        self._answers: dict[str, str] = {}
        self._messages: list[ChatMessage] = []

    @property
    def messages(self) -> list[ChatMessage]:
        return list(self._messages)

    def _next_id(self) -> str:
        self._counter += 1
        return f"msg_{self._counter:04d}"

    async def ask(
        self,
        text: str,
        *,
        options: list[str] | None = None,
        context: str = "",
    ) -> str:
        """Send a question to the human and wait for their answer.

        Returns the human's reply text.  If the human picks an option the
        returned value is the option string itself.
        """
        msg_id = self._next_id()
        event = asyncio.Event()
        self._pending[msg_id] = event

        prompt = text
        if context:
            prompt = f"{text}\n\n{context}"

        msg = ChatMessage(
            id=msg_id,
            role=MessageRole.AGENT,
            text=prompt,
            options=options or [],
            requires_answer=True,
        )
        self._messages.append(msg)
        await self._bus.emit(
            EventType.QUESTION_ASKED,
            message_id=msg_id,
            text=prompt,
            options=options or [],
        )

        logger.info("agent_asking id=%s text=%s", msg_id, text[:120])
        await event.wait()
        answer = self._answers.pop(msg_id)
        del self._pending[msg_id]
        return answer

    async def answer(self, message_id: str, text: str) -> bool:
        """Submit a human reply.  Returns True if it matched a pending question."""
        event = self._pending.get(message_id)
        if event is None:
            return False

        reply = ChatMessage(
            id=self._next_id(),
            role=MessageRole.HUMAN,
            text=text,
            in_reply_to=message_id,
        )
        self._messages.append(reply)
        self._answers[message_id] = text
        event.set()

        await self._bus.emit(
            EventType.ANSWER_RECEIVED,
            message_id=message_id,
            text=text,
        )
        logger.info("human_answered id=%s text=%s", message_id, text[:120])
        return True

    async def inform(self, text: str) -> None:
        """Send an informational message to the human (no answer needed)."""
        msg = ChatMessage(
            id=self._next_id(),
            role=MessageRole.AGENT,
            text=text,
        )
        self._messages.append(msg)
        await self._bus.emit(
            EventType.QUESTION_ASKED,
            message_id=msg.id,
            text=text,
            options=[],
        )

    def save(self) -> None:
        """Persist the conversation to disk."""
        if self._log_path is None:
            return
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        data = [msg.as_dict() for msg in self._messages]
        self._log_path.write_text(
            json.dumps(data, indent=2, default=str), encoding="utf-8"
        )
