"""Conversational replies to the operator, with a model where one is reachable.

The deterministic replies in :mod:`crucible.core.progress` answer the questions
an operator actually asks a running scan, but they cannot follow a question that
falls outside that list. This module adds a real answer for those, and keeps the
deterministic one as the floor underneath it.

That ordering is the whole design. A model is the *first* choice, never the only
one: free-tier endpoints retire model ids without notice, get rate limited, and
sit behind networks that block them outright. If any of that happens the operator
still gets a grounded, factual reply instead of silence -- which is what a chat
box that depends on an unreachable API actually delivers.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from crucible.core.config import Settings
from crucible.core.logging import get_logger
from crucible.core.progress import RunProgress
from crucible.llm.providers import ModelClient

logger = get_logger(__name__)

#: Long enough for a two-sentence answer, short enough that an operator watching
#: a scan is not left waiting on the model while the run moves on without them.
REPLY_TIMEOUT_SECONDS = 25.0

#: An answer is a couple of sentences. A large ceiling invites essays.
MAX_REPLY_TOKENS = 400

#: Warm rather than deterministic: this is a conversation, not a classification.
#: Zero temperature produces a clipped, repetitive register that reads like a
#: status endpoint rather than someone answering a question.
REPLY_TEMPERATURE = 0.3

SYSTEM_PROMPT = """\
You are the agent inside Crucible, a tool that automatically tests web \
applications: it crawls a site, exercises it, and reports defects it can prove. \
A human operator is watching one run in a dashboard and can type to you while it \
works. You are answering them.

Rules for your reply:

- One to three sentences. Be direct.
- Plain language. The operator may not be an engineer: avoid words like \
"invariant", "oracle", "signal", "heuristic", and never quote internal ids.
- Ground every statement in the run state below. Never invent a finding, a page, \
a number, or a fix that is not there.
- If the state does not answer the question, say plainly what you do not know yet.
- You cannot stop, pause or redirect a run that is underway, and there is no \
cancel button yet. Say so if asked to stop.
- Do not use markdown, headings, or bullet lists. Write sentences.
- Never claim a defect is fixed. You report; you do not repair.

The current state of the run:

{state}
"""


class ConversationalResponder:
    """Answers an operator's message, preferring a model over canned wording.

    Callable, so it can be handed straight to ``ChatChannel.start_responding``.
    """

    def __init__(
        self,
        progress: RunProgress,
        *,
        settings: Settings,
        providers: Sequence[ModelClient] | None = None,
        timeout_seconds: float = REPLY_TIMEOUT_SECONDS,
    ) -> None:
        self._progress = progress
        self._settings = settings
        self._timeout = timeout_seconds
        #: Injected in tests. When omitted, usable providers are resolved from
        #: settings on first use, so importing this module has no side effects.
        self._injected = list(providers) if providers is not None else None
        self._clients: list[ModelClient] | None = None

    async def __call__(self, text: str) -> str:
        """Return the best available reply: the model's, else the factual one."""
        grounded = self._progress.reply(text)
        for client in self._usable_clients():
            answer = await self._ask(client, text)
            if answer:
                return answer
        return grounded

    async def _ask(self, client: ModelClient, text: str) -> str | None:
        """One attempt against one provider. Never raises."""
        try:
            result = await asyncio.wait_for(
                client.complete(
                    self._messages(text),
                    temperature=REPLY_TEMPERATURE,
                    max_tokens=MAX_REPLY_TOKENS,
                ),
                timeout=self._timeout,
            )
        except Exception as exc:
            # Logged at debug: a blocked or retired endpoint is an expected
            # condition on free tiers, and the operator is still being answered.
            self._retire(client, f"{type(exc).__name__}: {exc}")
            return None

        answer = result.text.strip()
        if not answer:
            self._retire(client, "empty reply")
            return None
        return answer

    def _retire(self, client: ModelClient, reason: str) -> None:
        """Drop a provider that just failed, so the next message is not delayed.

        Retrying a dead endpoint on every message costs the operator the full
        timeout each time, to arrive at an answer that was available immediately.
        """
        logger.debug(
            "chat_reply_provider_retired provider=%s reason=%s",
            client.config.name.value,
            reason,
        )
        if self._clients is not None:
            self._clients = [candidate for candidate in self._clients if candidate is not client]

    def _messages(self, text: str) -> list[dict[str, str]]:
        system = SYSTEM_PROMPT.format(state=self._progress.state_brief())
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ]

    def _usable_clients(self) -> list[ModelClient]:
        """Providers worth trying, in preference order."""
        self._resolve_clients()
        return list(self._clients or [])

    def _resolve_clients(self) -> None:
        if self._clients is not None:
            return
        if self._injected is not None:
            self._clients = self._injected
            return
        clients: list[ModelClient] = []
        for name in self._settings.available_providers():
            config = self._settings.provider_config(name)
            if config is None:
                continue
            clients.append(ModelClient(config, timeout_seconds=self._timeout))
        if not clients:
            logger.info("chat_reply_no_provider")
        self._clients = clients

    async def aclose(self) -> None:
        """Release provider connections. Safe to call when nothing was opened."""
        for client in self._clients or []:
            await client.aclose()
        self._clients = []
