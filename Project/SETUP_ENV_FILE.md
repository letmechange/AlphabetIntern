# Setting Up Configuration with .env Files (Option 2)

This guide shows you how to configure the RAG system using `.env` files instead of setting environment variables manually.

## Prerequisites

1. **Install python-dotenv**:
   ```bash
   pip install python-dotenv
   ```
   
   Or install all requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Setup

### Step 1: Create a .env file

Run the built-in command to create a sample `.env` file:
```bash
python RAGUsage.py create-env
```

This will create a `.env` file in your project directory with all the necessary configuration variables.

### Step 2: Edit the .env file

Open the `.env` file and replace the placeholder values with your actual configuration:

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

### Step 3: Run the system

The system will automatically load the `.env` file when you run:
```bash
python RAGUsage.py example
python RAGUsage.py interactive
python RAGUsage.py batch
```

## Manual .env File Creation

If you prefer to create the `.env` file manually:

1. Create a file named `.env` in your project directory
2. Add your configuration variables (one per line)
3. Use the format: `VARIABLE_NAME=value`
4. No spaces around the `=` sign
5. No quotes needed unless the value contains spaces

Example `.env` file:
```bash
LLM_PROVIDER=azure
LLM_MODEL=gpt-4
AZURE_OPENAI_API_KEY=sk-your-key-here
AZURE_OPENAI_ENDPOINT=https://your-endpoint
```

## Configuration Variables Reference

### Core Configuration
- `LLM_PROVIDER`: Choose `azure`, `openai`, or `deepseek`
- `LLM_MODEL`: Model name (e.g., `gpt-4`, `gpt-3.5-turbo`)
- `EMBEDDING_MODEL_NAME`: Embedding model for document vectorization
- `VECTOR_DB_PATH`: Path to vector database storage
- `MANIFEST_PATH`: Path to document manifest file
- `ENABLE_HELPSTEER`: Enable/disable HelpSteer functionality

### Azure OpenAI
- `AZURE_OPENAI_API_KEY`: Your Azure OpenAI API key
- `AZURE_OPENAI_ENDPOINT`: Your Azure OpenAI endpoint URL
- `AZURE_OPENAI_DEPLOYMENT_NAME`: Deployment name (defaults to LLM_MODEL)
- `AZURE_OPENAI_API_VERSION`: API version

### DeepSeek
- `DEEPSEEK_API_KEY`: Your DeepSeek API key
- `DEEPSEEK_BASE_URL`: DeepSeek base URL (optional)

### OpenAI
- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_BASE_URL`: OpenAI base URL (optional)

## File Structure

Your project should look like this:
```
Project/
├── .env                    # Configuration file (create this)
├── RAGUsage.py            # Main usage script
├── config.py              # Configuration module
├── requirements.txt        # Dependencies
├── Sample/                # Document folder
└── vector_db/             # Vector database
```

## Security Best Practices

1. **Never commit `.env` files** to version control
2. Add `.env` to your `.gitignore` file:
   ```
   # .gitignore
   .env
   .env.local
   .env.*.local
   ```
3. Keep your API keys secure
4. Use different `.env` files for different environments (development, production)

## Troubleshooting

### Common Issues

1. **"python-dotenv not installed"**
   - Solution: `pip install python-dotenv`

2. **Configuration validation failed**
   - Check that all required variables are set in your `.env` file
   - Verify API keys and endpoints are correct

3. **File not found errors**
   - Ensure the `.env` file is in the same directory as your Python scripts
   - Check file permissions

4. **Environment variables not loading**
   - Restart your terminal/IDE after creating the `.env` file
   - Verify the `.env` file syntax (no spaces around `=`)

### Debug Mode

To see what configuration is being loaded, run any command and check the output:
```bash
python RAGUsage.py example
```

You should see:
```
Loaded configuration from .env file
Configuration Summary:
==================================================
LLM Provider: azure
LLM Model: gpt-4
...
```

## Alternative: Hybrid Approach

You can also use a hybrid approach:
- Set some variables in `.env` file
- Set others as environment variables
- Environment variables take priority over `.env` file values

This is useful for:
- Sensitive data (API keys) in environment variables
- Non-sensitive configuration in `.env` files
- Overriding specific values for testing


