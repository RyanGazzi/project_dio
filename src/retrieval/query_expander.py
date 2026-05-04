from ollama import Client
from configs.settings import settings

client = Client(host=settings.ollama_host)

EXPANSION_PROMPT = """Gere {n} formas diferentes de fazer a mesma pergunta.
Mantenha as variações CURTAS — máximo 10 palavras cada.
Não responda a pergunta, apenas reformule-a com palavras diferentes.
Retorne APENAS as variações, uma por linha, sem numeração."""


def expand_query(query: str, n: int = 3) -> list[str]:
    """
    Reescreve a query de N formas diferentes.
    Retorna a query original + as variações.
    """
    response = client.chat(
        model="llama3.2",  # ou qwen2.5, mistral — o que você tiver no Ollama
        messages=[
            {"role": "system", "content": EXPANSION_PROMPT.format(n=n)},
            {"role": "user", "content": query},
        ],
    )

    raw = response["message"]["content"].strip()
    variations = [line.strip() for line in raw.splitlines() if line.strip()]
    return [query] + variations[:n]