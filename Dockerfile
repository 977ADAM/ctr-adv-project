FROM node:20-alpine AS web-builder
WORKDIR /web
COPY web/package.json ./
RUN npm install
COPY web .
RUN npm run build

FROM python:3.11-slim
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY web /app/web
COPY artifacts /app/artifacts
COPY --from=web-builder /web/dist /app/web/dist

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir . \
    && pip install --no-cache-dir fastapi "uvicorn[standard]"

EXPOSE 8080

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080"]
