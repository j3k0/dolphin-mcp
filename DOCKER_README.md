# Docker Setup for Dolphin MCP

This document provides instructions for running Dolphin MCP using Docker.

## Prerequisites

- Docker
- Docker Compose
- OpenAI API key (or other LLM provider API keys as configured)

## Features

- Python 3.10 with all required dependencies
- Node.js 22 for MCP servers that require JavaScript/Node.js
- SQLite database for persistent storage
- Pre-built demo dolphin database

## Setup

1. Create a `.env` file based on the `.env.example`:

```bash
cp .env.example .env
```

2. Edit the `.env` file to add your OpenAI API key:

```
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o
```

## Build and Run

### Building the Docker Image

```bash
docker-compose build
```

### Running with Docker Compose

To get help:

```bash
docker-compose run dolphin-mcp
```

To run a query:

```bash
docker-compose run dolphin-mcp "What dolphin species are endangered?"
```

### Configuration Customization

You can customize the MCP configuration by modifying `mcp_config.json`. The Docker setup will use this configuration file when running the container.

## Custom Models

If you want to use different models (Anthropic, Ollama, etc.), ensure you add the appropriate API keys to your `.env` file and update the `mcp_config.json` accordingly.

## Persistent Storage

The Docker setup uses a named volume `dolphin-data` to persist the dolphin database between container runs. This ensures your database remains available even if you remove and recreate the container.

## MCP Servers with Node.js

The Docker image includes Node.js 22, making it compatible with MCP servers that require JavaScript/Node.js runtime. You can configure such servers in your `mcp_config.json` file.

## Troubleshooting

If you encounter issues:

1. Ensure your API keys are correct in the `.env` file
2. Check that the Docker service is running
3. Verify that the required ports are available if you've configured any port mappings
4. For Ollama models, ensure you have Ollama running and accessible from the container 