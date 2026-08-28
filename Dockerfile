# IFD Agent Dockerfile
# Handles Initial Flood Determination on the FEMA Map Service Center
# (msc.fema.gov). Includes Playwright for browser automation and PyMuPDF for
# PDF->PNG rendering used by the Bedrock vision extractor.
# Build context: repo root

# Stage 1: platform base
FROM public.ecr.aws/docker/library/python:3.11-slim AS platform-base

# Use bash with pipefail so "curl ... | sh" fails loudly if curl fails.
SHELL ["/bin/bash", "-eo", "pipefail", "-c"]

RUN apt-get update && apt-get install -y \
    git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# safe-chain: supply chain attack protection for subsequent pip installs
RUN curl -fsSL https://github.com/AikidoSec/safe-chain/releases/latest/download/install-safe-chain.sh | sh -s -- --ci
ENV PATH="/root/.safe-chain/shims:/root/.safe-chain/bin:${PATH}"

RUN pip install --no-cache-dir \
    bedrock-agentcore==1.4.7 \
    boto3==1.42.74 \
    "boto3-stubs[s3,secretsmanager]==1.42.74" \
    "aws-opentelemetry-distro>=0.10.1" \
    "opentelemetry-api>=1.20.0" \
    "opentelemetry-sdk>=1.20.0" \
    "opentelemetry-instrumentation-fastapi>=0.42b0" \
    "opentelemetry-instrumentation-boto3sqs>=0.42b0" \
    "opentelemetry-exporter-otlp-proto-http>=1.20.0" \
    "uvicorn>=0.32.0" \
    "itsdangerous>=2.1.0" \
    tenacity==8.3.0 \
    python-dateutil==2.9.0.post0 \
    python-dotenv==1.0.1 \
    "typing_extensions>=4.13.2,<5.0.0" \
    "urllib3>=1.25.4,<1.27"

RUN useradd -m -u 1000 bedrock_agentcore

# Stage 2: playwright base (browser + perms)
FROM platform-base AS playwright-base

RUN apt-get update && apt-get install -y \
    wget gnupg fonts-liberation \
    libasound2 libatk-bridge2.0-0 libatk1.0-0 libcups2 libdbus-1-3 \
    libdrm2 libgbm1 libgtk-3-0 libnspr4 libnss3 libxcomposite1 \
    libxdamage1 libxfixes3 libxkbcommon0 libxrandr2 xdg-utils \
    libu2f-udev libvulkan1 libxshmfence1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir playwright==1.55.0

RUN playwright install-deps chromium

ENV PLAYWRIGHT_BROWSERS_PATH=/opt/playwright-browsers

RUN groupadd -r playwright && \
    usermod -aG playwright root && \
    usermod -aG playwright bedrock_agentcore && \
    mkdir -p /opt/playwright-browsers && \
    playwright install chromium && \
    chown -R root:playwright /opt/playwright-browsers && \
    chmod -R 750 /opt/playwright-browsers

# Stage 3: af-tools on playwright
FROM playwright-base AS playwright-af-tools-base

COPY tools/ tools/
RUN pip install --no-cache-dir -e tools/

# Agent
FROM playwright-af-tools-base

ENV PYTHONUNBUFFERED=1

COPY backend/agents/vci/ifd_agent/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/agents/vci/ifd_agent/ ifd_agent/

RUN chown -R bedrock_agentcore:bedrock_agentcore /app

USER bedrock_agentcore

EXPOSE 8080
EXPOSE 8000

CMD ["opentelemetry-instrument", "python", "-m", "ifd_agent.ifd_agent"]
