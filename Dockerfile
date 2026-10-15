FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.11.19 /uv /usr/local/bin/uv
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    KEDRO_DISABLE_TELEMETRY=1 \
    MLFLOW_ENABLE_TELEMETRY=false \
    MLFLOW_DISABLE_AGENT_HINT=1
COPY pyproject.toml uv.lock .python-version ./
COPY src ./src
RUN uv sync --locked --no-dev
COPY conf ./conf
COPY notebooks ./notebooks
COPY .telemetry ./
CMD ["demo", "lab", "--host", "0.0.0.0"]
