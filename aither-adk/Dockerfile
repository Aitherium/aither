FROM python:3.12-slim AS builder

WORKDIR /build
COPY pyproject.toml README.md ./
COPY adk/ adk/

RUN pip install --no-cache-dir --prefix=/install .

# ---------------------------------------------------------------------------

FROM python:3.12-slim

LABEL org.opencontainers.image.title="aither-adk" \
      org.opencontainers.image.description="AitherOS Agent Development Kit server" \
      org.opencontainers.image.version="0.1.0a1" \
      org.opencontainers.image.vendor="Aitherium" \
      org.opencontainers.image.url="https://aitherium.com" \
      org.opencontainers.image.source="https://github.com/Aitherium/AitherOS-Alpha" \
      org.opencontainers.image.licenses="Apache-2.0"

RUN groupadd --gid 1000 adk && \
    useradd --uid 1000 --gid adk --create-home adk

COPY --from=builder /install /usr/local

WORKDIR /home/adk
USER adk

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8080/health', timeout=4).raise_for_status()"

CMD ["aither-serve", "--host", "0.0.0.0", "--port", "8080"]
