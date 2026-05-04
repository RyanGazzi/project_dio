from ollama import Client
from configs.settings import settings

client = Client(host=settings.ollama_host)

EXPANSION_PROMPT = """Dada a pergunta abaixo, gere {n} variações curtas.

Regras OBRIGATÓRIAS:
- Mantenha o mesmo significado
- NÃO mude a intenção
- Preserve termos-chave importantes 
- Inclua pelo menos UMA variação corrigindo possíveis erros de digitação
- Inclua pelo menos UMA variação mais "limpa" para busca (sem erros)
- Máximo 10 palavras

Retorne apenas as variações, uma por linha.

Pergunta: {query}
"""


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