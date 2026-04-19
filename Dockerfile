# 🦆 DevDuck - Self-adapting AI agent (dev build from source)
FROM python:3.13-slim

LABEL org.opencontainers.image.title="DevDuck"
LABEL org.opencontainers.image.description="Self-Adapting Agent"
LABEL org.opencontainers.image.source="https://github.com/cagataycali/devduck"
LABEL org.opencontainers.image.licenses="Apache-2.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
      portaudio19-dev \
      libasound2-dev \
      build-essential \
      python3-dev \
      git \
      curl \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 duck
WORKDIR /app

# Fallback version if .git isn't a proper repo (setuptools-scm)
ARG DEVDUCK_VERSION=0.0.0+docker
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_DEVDUCK=${DEVDUCK_VERSION}

COPY --chown=duck:duck . /app

# Trust /app for git (fixes dubious ownership) and install
RUN git config --global --add safe.directory /app && \
    pip install --no-cache-dir .

USER duck
WORKDIR /home/duck
ENV DEVDUCK_AUTO_START_SERVERS=true \
    DEVDUCK_ENABLE_WS=true \
    DEVDUCK_ENABLE_ZENOH=false \
    BYPASS_TOOL_CONSENT=true \
    PYTHONUNBUFFERED=1

EXPOSE 10000 10001 10002 10003
ENTRYPOINT ["devduck"]
