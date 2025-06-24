import getpass
import os

# API keys and Endpoints (used for Azure OpenAI) 
if "AZURE_OPENAI_API_KEY" not in os.environ:
    os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass(
        " qhfhcZwRWTGDykfNXzUoINKnPxI8lFiQzBne4vJcbLzvxupagXFHJQQJ99ALACHYHv6XJ3w3AAAAACOGS5fO"
    )
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://sweet-m55d9k6j-eastus2.openai.azure.com/openai/deployments/yu-gpt-4o/chat/completions?api-version=2025-01-01-preview"

# API keys (used for Deepseek)
if not os.getenv("DEEPSEEK_API_KEY"):
    os.environ["DEEPSEEK_API_KEY"] = getpass.getpass("Enter your DeepSeek API key: ")