from src.retrieval.vector_search import SearchResult


SYSTEM_PROMPT = """Você é um assistente especializado em responder perguntas
com base em documentos fornecidos.

Regras OBRIGATÓRIAS:
- Responda APENAS com informações explicitamente presentes nos trechos fornecidos
- NUNCA invente, suponha ou complemente com conhecimento próprio
- NUNCA cite trechos ou páginas que não estejam no contexto fornecido
- Se a resposta não estiver claramente nos trechos, responda: "Não encontrei essa informação nos documentos fornecidos"
- Ao citar, use APENAS os trechos numerados que aparecem no contexto
- Seja direto e objetivo"""

def build_prompt(query: str, chunks: list[SearchResult]) -> list[dict]:
    """
    Monta a lista de mensagens para enviar ao LLM.
    """
    # Formata os chunks como contexto
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "desconhecido")
        page = chunk.metadata.get("page", "?")
        context_parts.append(
            f"[Trecho {i} — {source}, página {page}]\n{chunk.content}"
        )

    context = "\n\n---\n\n".join(context_parts)

    user_message = f"""Contexto extraído dos documentos:

{context}

---

Pergunta: {query}"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]