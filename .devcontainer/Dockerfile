# Use the latest PyTorch base image
FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-runtime

# Install system dependencies in a single RUN command to reduce image layers
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    openssh-client \
    less \
    curl \
    wget \
    unzip \
    nano && \
    rm -rf /var/lib/apt/lists/*



RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    jinja2


RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get update \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN npm install -g @openai/codex    