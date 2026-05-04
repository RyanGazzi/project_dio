from ollama import Client
from loguru import logger
from configs.settings import settings

client = Client(host=settings.ollama_host)

COMPRESSION_PROMPT = """Dado o trecho de documento abaixo e uma pergunta, 
extraia o conteúdo relevante para responder a pergunta.

Regras:
- Se houver uma definição, explicação completa ou frase principal que responde à pergunta, retorne ela inteira
- Prefira manter contexto suficiente para que a resposta faça sentido sozinha
- Evite cortar partes importantes da explicação
- Só reduza o texto se houver claramente partes irrelevantes
- Considere correspondência semântica, mesmo com erros ou formatação ruim
- Só responda "IRRELEVANTE" se NÃO houver nenhuma informação útil

Retorne apenas o texto extraído, sem explicações.

Pergunta: {query}

Trecho:
{content}"""

def compress_context(query: str, results: list, max_irrelevant: int = 3) -> list:
    compressed = []
    irrelevant_count = 0

    for result in results:
        # Tabelas passam direto — compressão não funciona bem com markdown tabular
        if result.metadata.get("content_type") == "table":
            compressed.append(result)
            continue

        response = client.chat(
            model=settings.ollama_llm_model,
            messages=[{
                "role": "user",
                "content": COMPRESSION_PROMPT.format(
                    query=query,
                    content=result.content,
                )
            }],
        )

        extracted = response["message"]["content"].strip()

        if "irrelevante" in extracted.lower():
            irrelevant_count += 1
            if irrelevant_count >= max_irrelevant:
                break
            continue

        if len(extracted) < 20:
            logger.warning(f"Chunk descartado após compressão: '{extracted}'")
            continue

        result.content = extracted
        compressed.append(result)

    return compressed