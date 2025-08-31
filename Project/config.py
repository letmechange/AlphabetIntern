import getpass
import os
import re
"""
User can create the .env file by the following format (replace the placeholder values with your actual values):

LLM_PROVIDER=azure
LLM_MODEL=gpt-4
EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
VECTOR_DB_PATH=./vector_db
MANIFEST_PATH=manifest.json
ENABLE_HELPSTEER=false

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_deepseek_api_key_here
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_DEPLOYMENT_NAME=
AZURE_OPENAI_API_VERSION=

# DeepSeek Configuration (if using DeepSeek)
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_BASE_URL=

"""
# Load .env file if it exists (Option 2 support)

try:
    from dotenv import load_dotenv
    # Load .env file from the current directory
    load_dotenv()
    print("Loaded configuration from .env file")
except ImportError:
    print("python-dotenv not installed. Install with: pip install python-dotenv")
    print("Or set environment variables manually.")
except Exception as e:
    print(f"Error loading .env file: {e}")

def validate_api_key(api_key: str, provider: str) -> bool:
    """
    Validate API key format
    
    Args:
        api_key: API key
        provider: Provider name
        
    Returns:
        bool: Whether the key format is valid
    """
    if not api_key or api_key.strip() == "":
        return False
    
    # Basic format validation
    if provider.lower() == "azure":
        # Azure OpenAI keys usually start with sk-
        return api_key.startswith("sk-") or len(api_key) > 20
    elif provider.lower() == "deepseek":
        # DeepSeek key format validation
        return len(api_key) > 10
    else:
        # Generic validation
        return len(api_key) > 10

def get_api_key(provider: str, env_var: str) -> str:
    """
    Safely get API key
    
    Args:
        provider: Provider name
        env_var: Environment variable name
        
    Returns:
        str: API key
    """
    # First try to get from environment variables
    api_key = os.getenv(env_var)
    
    if api_key and validate_api_key(api_key, provider):
        return api_key
    
    # If not in environment variables or invalid, prompt user to input
    print(f"\n Need to configure {provider} API key")
    print("Please get API key from the following locations:")
    
    if provider.lower() == "azure":
        print("  - Azure OpenAI: https://portal.azure.com/")
        print("  - Or use environment variable: AZURE_OPENAI_API_KEY")
    elif provider.lower() == "deepseek":
        print("  - DeepSeek: https://platform.deepseek.com/")
        print("  - Or use environment variable: DEEPSEEK_API_KEY")
    
    while True:
        try:
            api_key = getpass.getpass(f"Please enter your {provider} API key: ").strip()
            
            if validate_api_key(api_key, provider):
                # Set environment variable
                os.environ[env_var] = api_key
                print(f"{provider} API key configured")
                return api_key
            else:
                print(f"API key format invalid, please re-enter")
                
        except KeyboardInterrupt:
            print("\nUser cancelled operation")
            raise SystemExit(1)
        except Exception as e:
            print(f"Input error: {str(e)}")

def setup_azure_openai():
    """Configure Azure OpenAI"""
    api_key = get_api_key("Azure OpenAI", "AZURE_OPENAI_API_KEY")
    
    # Set Azure OpenAI endpoint
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if not endpoint:
        print("No Azure OpenAI endpoint found in environment variables.")
        print("Please provide your Azure OpenAI endpoint")
        endpoint = input("Enter your Azure OpenAI endpoint: ").strip()
        if not endpoint:
            print("No endpoint provided. Exiting configuration.")
            raise SystemExit(1)
        os.environ["AZURE_OPENAI_ENDPOINT"] = endpoint
        print("Azure OpenAI endpoint configured from user input.")
    
    return api_key, endpoint

def setup_deepseek():
    """Configure DeepSeek"""
    api_key = get_api_key("DeepSeek", "DEEPSEEK_API_KEY")
    return api_key

def validate_config():
    """Validate if configuration is complete"""
    required_vars = {
        "AZURE_OPENAI_API_KEY": "Azure OpenAI API key",
        "AZURE_OPENAI_ENDPOINT": "Azure OpenAI endpoint",
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"{description} ({var})")
    
    if missing_vars:
        print("  Missing the following configuration:")
        for var in missing_vars:
            print(f"  - {var}")
        return False
    
    return True

# Configure Azure OpenAI
try:
    azure_api_key, azure_endpoint = setup_azure_openai()
except SystemExit:
    print("Configuration failed, program exiting")
    exit(1)

# Configure DeepSeek (optional)
try:
    deepseek_api_key = setup_deepseek()
except SystemExit:
    print(" DeepSeek configuration skipped, will only use Azure OpenAI")

# Validate configuration
if not validate_config():
    print("Configuration validation failed")
    exit(1)

print(" Configuration complete")