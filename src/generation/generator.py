from collections.abc import Generator
from ollama import Client
from loguru import logger
from configs.settings import settings

client = Client(host=settings.ollama_host)


def generate_streaming(messages: list[dict]) -> Generator[str, None, None]:
    """
    Chama o LLM com streaming — yield de cada token conforme chega.
    """
    logger.info("Iniciando geração com streaming...")

    stream = client.chat(
        model=settings.ollama_llm_model,
        messages=messages,
        stream=True,
    )

    for chunk in stream:
        token = chunk["message"]["content"]
        if token:
            yield token