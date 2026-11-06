import re
import unicodedata


_GREETING_ANSWERS = {
    "oi": "Olá! Posso ajudar você a consultar e analisar convenções coletivas de trabalho.",
    "ola": "Olá! Posso ajudar você a consultar e analisar convenções coletivas de trabalho.",
    "bom dia": "Bom dia! Como posso ajudar com as convenções coletivas?",
    "boa tarde": "Boa tarde! Como posso ajudar com as convenções coletivas?",
    "boa noite": "Boa noite! Como posso ajudar com as convenções coletivas?",
    "e ai": "Olá! Que informação você gostaria de consultar nas convenções coletivas?",
    "tudo bem": "Tudo bem! Como posso ajudar com as convenções coletivas?",
    "oi tudo bem": "Olá! Tudo bem. Como posso ajudar com as convenções coletivas?",
    "ola tudo bem": "Olá! Tudo bem. Como posso ajudar com as convenções coletivas?",
    "obrigado": "Por nada! Estou à disposição para outras consultas às convenções coletivas.",
    "obrigada": "Por nada! Estou à disposição para outras consultas às convenções coletivas.",
    "muito obrigado": "Por nada! Estou à disposição para outras consultas às convenções coletivas.",
    "muito obrigada": "Por nada! Estou à disposição para outras consultas às convenções coletivas.",
    "valeu": "Por nada! Se precisar consultar outra cláusula, é só perguntar.",
    "ate logo": "Até logo! Quando precisar consultar as convenções, estou por aqui.",
    "tchau": "Até logo! Quando precisar consultar as convenções, estou por aqui.",
}

_CAPABILITY_QUESTIONS = frozenset(
    {
        "o que voce faz",
        "o que voce consegue fazer",
        "como voce funciona",
        "como funciona este chat",
        "como funciona o chat",
        "como usar o chat",
        "como usar essa ferramenta",
        "como posso usar voce",
        "para que serve esse chat",
        "qual a sua funcao",
    }
)

_CAPABILITY_ANSWER = (
    "Posso consultar os documentos indexados, localizar cláusulas de convenções "
    "coletivas, comparar obrigações e benefícios e apresentar trechos das "
    "fontes utilizadas. Faça uma pergunta sobre os documentos para começar."
)

_OFF_TOPIC_ANSWER = (
    "Este chat é voltado à consulta de convenções coletivas de trabalho. "
    "Posso ajudar com cláusulas, obrigações, benefícios e comparações entre "
    "os documentos indexados."
)

# Only unambiguously unrelated requests are short-circuited. Questions that
# could refer to clauses or previous answers must reach the RAG pipeline.
_CLEAR_OFF_TOPIC_PATTERNS = (
    re.compile(r"^(?:me (?:de|ensine|passe) )?uma? receita de .+$"),
    re.compile(r"^(?:qual (?:e|eh) )?a previsao do tempo(?: .*)?$"),
    re.compile(r"^qual (?:e|eh) a capital d[aeo] .+$"),
)


def _normalize_intent_text(message: str) -> str:
    """Normalize conversational text for exact intent matching.

    Args:
        message: Raw user message, potentially containing accents or punctuation.

    Returns:
        str: Case-folded, accent-free text with collapsed whitespace and
        peripheral punctuation removed.
    """
    without_accents = "".join(
        char
        for char in unicodedata.normalize("NFKD", message.casefold())
        if not unicodedata.combining(char)
    )
    cleaned = re.sub(r"[^\w\s]", " ", without_accents)
    return " ".join(cleaned.split())


def get_direct_answer(message: str) -> str | None:
    """Return a direct answer only for clearly non-documentary messages.

    Args:
        message: User input, already validated by the chat request DTO.

    Returns:
        str | None: Fixed answer when the message is an explicit greeting,
        acknowledgement, capability question or clearly off-topic request;
        otherwise None so the existing RAG pipeline handles the request.

    Notes:
        Classification is intentionally conservative: an incidental greeting
        in a substantive question does not suppress document retrieval. No
        embeddings, vector queries or LLM calls are made here.
    """
    normalized = _normalize_intent_text(message)
    if normalized in _GREETING_ANSWERS:
        return _GREETING_ANSWERS[normalized]
    if normalized in _CAPABILITY_QUESTIONS:
        return _CAPABILITY_ANSWER
    if any(pattern.fullmatch(normalized) for pattern in _CLEAR_OFF_TOPIC_PATTERNS):
        return _OFF_TOPIC_ANSWER
    return None
