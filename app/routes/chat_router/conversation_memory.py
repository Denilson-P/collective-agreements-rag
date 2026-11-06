"""Bounded, process-local conversation state for the chat endpoint."""

import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, NamedTuple
from threading import RLock
from uuid import UUID, uuid4


class ConversationTurn(NamedTuple):
    """Represent a saved exchange and its evidence-bearing source names."""

    question: str
    answer: str
    documentary: bool
    sources: tuple[str, ...] = ()


@dataclass
class Conversation:
    """Hold the user's explicitly supplied name and recent successful turns."""

    name: str | None = None
    turns: list[ConversationTurn] = field(default_factory=list)


class ConversationMemory:
    """Store isolated sessions with bounded history and thread-safe access.

    Sessions are ephemeral and not shared between processes or restarts.
    """

    def __init__(self, max_sessions: int = 256, max_turns: int = 12) -> None:
        """Configure the maximum session count and turns per session."""
        self.max_sessions = max_sessions
        self.max_turns = max_turns
        self._sessions: OrderedDict[str, Conversation] = OrderedDict()
        self._lock = RLock()

    def get(self, conversation_id: str | None = None) -> tuple[str, Conversation]:
        """Return a snapshot of a UUID-keyed session, creating it if absent.

        Args:
            conversation_id: Previously returned UUID, or None for a new session.

        Returns:
            Tuple containing session UUID and a copy of its saved state.

        Raises:
            ValueError: If the conversation identifier is not a valid UUID.
        """
        key = str(UUID(conversation_id)) if conversation_id is not None else str(uuid4())
        with self._lock:
            session = self._sessions.setdefault(key, Conversation())
            self._sessions.move_to_end(key)
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)
            return key, Conversation(name=session.name, turns=list(session.turns))

    def save(
        self,
        conversation_id: str,
        question: str,
        answer: str,
        documentary: bool,
        sources: list[dict[str, Any]] | None = None,
    ) -> None:
        """Save a successful message and update an explicitly stated name.

        Args:
            conversation_id: UUID returned for the current session.
            question: User's exact message.
            answer: Assistant's response.
            documentary: Whether the turn used documentary retrieval.
            sources: Evidence-bearing sources returned in the chat response.
        """
        with self._lock:
            session = self._sessions.setdefault(conversation_id, Conversation())
            match = re.search(
                r"\b(?:meu nome [ée]|me chamo|pode me chamar de)\s+"
                r"([A-Za-zÀ-ÖØ-öø-ÿ]+(?:[ -][A-Za-zÀ-ÖØ-öø-ÿ]+){0,3})",
                question,
                flags=re.IGNORECASE,
            )
            if match:
                candidate = match.group(1).strip()
                candidate = re.split(r"\s+(?:e|mas|porque|pois)\s+", candidate, maxsplit=1, flags=re.IGNORECASE)[0]
                session.name = candidate.strip().rstrip('.,!?')
            source_names = tuple(dict.fromkeys(
                str(item["source"]) for item in (sources or [])
                if isinstance(item, dict) and isinstance(item.get("source"), str)
                and item["source"].strip()
            ))
            session.turns.append(ConversationTurn(
                question=question,
                answer=answer,
                documentary=documentary,
                sources=source_names,
            ))
            del session.turns[:-self.max_turns]
            self._sessions.move_to_end(conversation_id)
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)


conversation_memory = ConversationMemory()
