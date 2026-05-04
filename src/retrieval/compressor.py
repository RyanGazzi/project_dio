from ollama import Client
from loguru import logger
from configs.settings import settings

client = Client(host=settings.ollama_host)

COMPRESSION_PROMPT = """Dado o trecho de documento abaixo e uma pergunta, 
extraia APENAS as partes do trecho que são diretamente relevantes para responder 
a pergunta. Se nenhuma parte for relevante, responda exatamente: "IRRELEVANTE".
Não adicione explicações, apenas o texto extraído.

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