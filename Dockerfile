FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-por \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia o pyproject.toml ANTES de instalar
COPY pyproject.toml .

# Instala dependências
RUN pip install --no-cache-dir -e ".[dev]"

# Garante uvicorn instalado explicitamente
RUN pip install uvicorn fastapi

# Copia o código
COPY src/ ./src/
COPY configs/ ./configs/

# Baixa o modelo de reranking
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]