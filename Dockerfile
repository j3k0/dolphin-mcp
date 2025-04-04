FROM python:3.11

# Set working directory
WORKDIR /app

# Install required system dependencies including SQLite and useful shell tools
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    sqlite3 \
    gnupg \
    jq \
    less \
    pandoc \
    vim \
    nano \
    iputils-ping \
    htop \
    git \
    wget \
    procps \
    grep \
    sed \
    gawk \
    tree \
    tmux \
    zip \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 22
RUN mkdir -p /etc/apt/keyrings
RUN curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg
RUN echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_22.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list
RUN apt-get update && apt-get install -y nodejs && rm -rf /var/lib/apt/lists/*
RUN node --version && npm --version

# Install uv and uvenv (formerly uvx) for Python package management
RUN pip install uv uvenv
# Verify uv and uvenv are installed correctly
RUN uv --version && uvenv --version

# Install a Python package using uvenv as a demonstration
# This will install the httpie tool as a standalone package
RUN uvenv install httpie && \
    uvenv list && \
    /root/.local/bin/http --version || echo "httpie not available"

# Create and activate a uv environment for the application
RUN uv venv /app/.venv
ENV PATH="/app/.venv/bin:${PATH}"
ENV VIRTUAL_ENV="/app/.venv"

# Set the PATH to include uvenv bins
ENV PATH="/root/.local/bin:${PATH}"

# Copy requirement files
COPY requirements.txt pyproject.toml ./

# Copy the rest of the application
COPY .env.example README.md dolphin_mcp.py mcp_config.json setup_db.py ./
COPY src ./src

# Install Python dependencies in the virtual environment
RUN uv pip install -e . || pip install -e .
RUN pip install -r requirements.txt

# Set up default environment variables that can be overridden at runtime
ENV OPENAI_API_KEY=""
ENV OPENAI_MODEL="gpt-4o"

# Create directory for dolphin database
RUN mkdir -p ~/.dolphin

# Build the demo database
# RUN python setup_db.py

# Set the entry point to the dolphin-mcp-cli command
ENTRYPOINT ["dolphin-mcp-cli"]

# Default command if none is provided
CMD ["--help"] 
