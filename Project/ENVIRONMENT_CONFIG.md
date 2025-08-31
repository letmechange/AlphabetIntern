# Environment Variable Configuration Guide

This guide explains how to configure the RAG system using environment variables.

## Quick Setup

1. Create a `.env` file in your project directory
2. Copy the configuration below and fill in your values
3. The system will automatically load these variables

## Configuration Variables

### Core Configuration

| Variable | Description | Default Value |
|----------|-------------|---------------|
| `LLM_PROVIDER` | LLM provider to use | `azure` |
| `LLM_MODEL` | LLM model name | `gpt-4` |
| `EMBEDDING_MODEL_NAME` | Embedding model name | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| `VECTOR_DB_PATH` | Vector database path | `./vector_db` |
| `MANIFEST_PATH` | Document manifest path | `manifest.json` |
| `ENABLE_HELPSTEER` | Enable HelpSteer functionality | `false` |

### Azure OpenAI Configuration

| Variable | Description | Required |
|----------|-------------|----------|
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | Yes |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL | Yes |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | Deployment name | No (defaults to LLM_MODEL) |
| `AZURE_OPENAI_API_VERSION` | API version | No (defaults to 2025-01-01-preview) |

### DeepSeek Configuration

| Variable | Description | Required |
|----------|-------------|----------|
| `DEEPSEEK_API_KEY` | DeepSeek API key | Yes |
| `DEEPSEEK_BASE_URL` | DeepSeek base URL | No (defaults to https://api.deepseek.com) |

### OpenAI Configuration

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key | Yes |
| `OPENAI_BASE_URL` | OpenAI base URL | No (defaults to https://api.openai.com/v1) |

## Example .env File

```bash
# Core Configuration
LLM_PROVIDER=azure
LLM_MODEL=gpt-4
EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
VECTOR_DB_PATH=./vector_db
MANIFEST_PATH=manifest.json
ENABLE_HELPSTEER=false

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=sk-your-actual-api-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/openai/deployments/your-deployment/chat/completions?api-version=2025-01-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
AZURE_OPENAI_API_VERSION=2025-01-01-preview
```

## Setting Environment Variables

### Windows (PowerShell)
```powershell
$env:AZURE_OPENAI_API_KEY="your-api-key-here"
$env:AZURE_OPENAI_ENDPOINT="your-endpoint-here"
```

### Windows (Command Prompt)
```cmd
set AZURE_OPENAI_API_KEY=your-api-key-here
set AZURE_OPENAI_ENDPOINT=your-endpoint-here
```

### Linux/macOS
```bash
export AZURE_OPENAI_API_KEY="your-api-key-here"
export AZURE_OPENAI_ENDPOINT="your-endpoint-here"
```

## Priority Order

The system loads configuration in the following priority order:

1. **Environment Variables** (highest priority)
2. **Default Values** (lowest priority)

This means if you set an environment variable, it will override the default value.

## Validation

The system automatically validates your configuration and will show:
- Configuration summary
- Any missing required variables
- Validation errors

## Security Notes

- Never commit your `.env` file to version control
- Keep your API keys secure
- Use environment variables in production environments
- Consider using a secrets management service for production
