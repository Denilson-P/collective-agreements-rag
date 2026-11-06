import re
import unicodedata
from typing import Any

from langchain_core.language_models import BaseChatModel

from app.adapters.qdrant_vector_store import QdrantVectorStoreAdapter
from app.rag.pipeline import run_rag_pipeline
from app.rag.retriever import get_retriever
from app.routes.chat_router.conversation_memory import conversation_memory
from app.routes.chat_router.intent_router import get_direct_answer


_NAME_QUESTION_PATTERNS = (
    re.compile(r"^qual (?:e )?(?:o )?meu nome$"),
    re.compile(r"^voce (?:lembra|sabe) (?:qual (?:e )?(?:o )?meu nome|(?:do )?meu nome)$"),
    re.compile(r"^(?:como (?:eu )?me chamo|voce sabe como (?:eu )?me chamo)$"),
    re.compile(r"^eu (?:te )?falei (?:o )?meu nome$"),
)


def _is_name_question(normalized_question: str) -> bool:
    """Identify common questions about the name supplied in this session.

    Args:
        normalized_question: Accent-free, punctuation-free user message.

    Returns:
        bool: Whether the message asks for the remembered user's name.
    """
    return any(
        pattern.fullmatch(normalized_question)
        for pattern in _NAME_QUESTION_PATTERNS
    )


class ChatService:
    """Answer direct chat messages and session-aware documentary questions."""

    def __init__(
        self,
        vector_store: QdrantVectorStoreAdapter | None = None,
        chat_model: BaseChatModel | None = None,
    ) -> None:
        """Initialize the chat service with the configured retrieval dependencies.

        Args:
            vector_store: Optional Qdrant adapter; resolved lazily when omitted.
            chat_model: Optional injected language model.
        """
        self.vector_store = (
            vector_store if vector_store is not None else get_retriever()
        )
        self.chat_model = chat_model

    def ask(
        self,
        question: str,
        k: int | None = None,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        """Answer a message and retain a bounded conversation history.

        Args:
            question: Validated user message.
            k: Optional document-retrieval limit.
            conversation_id: Previous session UUID, or None to start a session.

        Returns:
            Answer, sources, original question, and reusable conversation UUID.

        Raises:
            ValueError: If the identifier or retrieval parameters are invalid.
            Exception: If the existing retrieval or generation pipeline fails.
        """
        session_id, snapshot = conversation_memory.get(conversation_id)
        normalized = ''.join(
            char for char in unicodedata.normalize('NFKD', question.casefold())
            if not unicodedata.combining(char)
        )
        normalized = ' '.join(re.sub(r'[^\w\s]', ' ', normalized).split())

        if _is_name_question(normalized):
            answer = (
                f'Seu nome é {snapshot.name}.' if snapshot.name
                else 'Você ainda não informou seu nome nesta conversa.'
            )
            result = {'question': question, 'answer': answer, 'sources': []}
            documentary = False
        else:
            direct = get_direct_answer(question)
            introduction = re.search(
                r'\b(?:meu nome [ée]|me chamo|pode me chamar de)\s+',
                question, re.IGNORECASE,
            )
            # Introductions are conversational unless a documentary question
            # is included in the same message.
            documentary_terms = re.search(
                r'\b(?:convenc[aã]o|acordo|cl[aá]usula|documento|seguro|'
                r'aux[ií]lio|f[eé]rias|obriga[cç][aã]o|benef[ií]cio)\b',
                question, re.IGNORECASE,
            )
            if introduction and not documentary_terms:
                name = re.search(
                    r'\b(?:meu nome [ée]|me chamo|pode me chamar de)\s+'
                    r'([A-Za-zÀ-ÖØ-öø-ÿ]+)',
                    question, re.IGNORECASE,
                )
                answer = (f'Prazer, {name.group(1)}! Como posso ajudar?' if name
                          else 'Prazer! Como posso ajudar?')
                result = {'question': question, 'answer': answer, 'sources': []}
                documentary = False
            elif direct is not None:
                result = {'question': question, 'answer': direct, 'sources': []}
                documentary = False
            else:
                effective_question = question
                allowed_sources = None
                # Only contextualize overt follow-ups, never unrelated queries.
                if re.match(r'^\s*(?:e\s+)?(?:quais delas|qual delas|e quanto|e quais|e qual|e sobre|dessas|desses|isso|elas|eles)\b',
                            normalized):
                    previous = next(
                        (turn for turn in reversed(snapshot.turns) if turn[2]),
                        None,
                    )
                    if previous and previous.sources:
                        allowed_sources = list(previous.sources)
                        # When the previous question requested mandatory
                        # provisions, do not carry facultative sources forward.
                        if re.search(r'\b(?:obrigat[oó]ri[oa]s?|exig(?:em|e))\b',
                                     previous.question, re.IGNORECASE):
                            mandatory = [
                                source for source in allowed_sources
                                if re.search(
                                    rf'\*\*{re.escape(source)}\*\*\s*[—–-]\s*Obrigatório',
                                    previous.answer,
                                )
                            ]
                            if '—' in previous.answer and 'Obrigatório' in previous.answer:
                                allowed_sources = mandatory
                        # The new search must focus on the NEW subject.
                        # Previous evidence is not injected into its query.
                        effective_question = question
                    elif previous:
                        # Without verified sources, do not infer a source list.
                        # Preserve the prior behavior for legacy sessions.
                        effective_question = (
                            f'Pergunta anterior: {previous.question[:500]}\n'
                            f'Pergunta de acompanhamento: {question}'
                        )
                pipeline_kwargs = {}
                if allowed_sources is not None:
                    pipeline_kwargs['allowed_sources'] = allowed_sources
                result = run_rag_pipeline(
                    question=effective_question,
                    vector_store=self.vector_store,
                    chat_model=self.chat_model,
                    k=k,
                    **pipeline_kwargs,
                )
                result['question'] = question
                documentary = True

        conversation_memory.save(session_id, question, result['answer'], documentary, sources=result.get('sources', []))
        return {**result, 'conversation_id': session_id}
