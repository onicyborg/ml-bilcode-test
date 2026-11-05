FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.7 /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev
COPY . .
RUN python -c "import pathlib,json,hashlib; p=pathlib.Path('artifacts'); m=json.loads((p/'model.json').read_text()); assert hashlib.sha256((p/'mobilenetv2.keras').read_bytes()).hexdigest()==m['sha256']; assert hashlib.sha256(pathlib.Path('detector/yolox.onnx').read_bytes()).hexdigest()==m['inference']['detectorSha256']"
RUN useradd --create-home app && chown -R app:app /app
USER app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
