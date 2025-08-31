#!/usr/bin/env python3
"""
RAG System Usage Examples
Demonstrates how to use the complete RAG system for document queries
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

from RAGSystem import RAGSystem
import os
import time
import re

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def load_config_from_env():
    """
    Load configuration from environment variables with priority
    Returns a dictionary with configuration values
    """
    config = {}
    
    # LLM Provider configuration
    config['llm_provider'] = os.getenv('LLM_PROVIDER', 'azure').lower()
    
    # Model configuration
    config['llm_model'] = os.getenv('LLM_MODEL', 'gpt-4')
    
    # Embedding model configuration
    config['embedding_model_name'] = os.getenv(
        'EMBEDDING_MODEL_NAME', 
        'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    )
    
    # Vector database path
    config['vector_db_path'] = os.getenv('VECTOR_DB_PATH', './vector_db')
    
    # Manifest path
    config['manifest_path'] = os.getenv('MANIFEST_PATH', 'manifest.json')
    
    # Azure OpenAI specific configuration
    if config['llm_provider'] == 'azure':
        config['azure_api_key'] = os.getenv('AZURE_OPENAI_API_KEY')
        config['azure_endpoint'] = os.getenv('AZURE_OPENAI_ENDPOINT')
        config['azure_deployment_name'] = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', config['llm_model'])
        config['azure_api_version'] = os.getenv('AZURE_OPENAI_API_VERSION', '2025-01-01-preview')
    
    # DeepSeek specific configuration
    elif config['llm_provider'] == 'deepseek':
        config['deepseek_api_key'] = os.getenv('DEEPSEEK_API_KEY')
        config['deepseek_base_url'] = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
    
    # HelpSteer configuration
    config['enable_helpsteer'] = os.getenv('ENABLE_HELPSTEER', 'false').lower() == 'true'
    
    return config


def validate_config(config):
    """
    Validate the loaded configuration
    Returns True if valid, False otherwise
    """
    errors = []
    
    # Check required LLM provider
    if config['llm_provider'] not in ['azure', 'openai', 'deepseek']:
        errors.append(f"Unsupported LLM provider: {config['llm_provider']}")
    
    # Check provider-specific required variables
    if config['llm_provider'] == 'azure':
        if not config['azure_api_key']:
            errors.append("AZURE_OPENAI_API_KEY is required for Azure provider")
        if not config['azure_endpoint']:
            errors.append("AZURE_OPENAI_ENDPOINT is required for Azure provider")
    
    elif config['llm_provider'] == 'deepseek':
        if not config['deepseek_api_key']:
            errors.append("DEEPSEEK_API_KEY is required for DeepSeek provider")
    
    elif config['llm_provider'] == 'openai':
        if not config['openai_api_key']:
            errors.append("OPENAI_API_KEY is required for OpenAI provider")
    
    if errors:
        print("Configuration validation errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True


def print_config_summary(config):
    """
    Print a summary of the loaded configuration
    """
    print("Configuration Summary:")
    print("=" * 50)
    print(f"LLM Provider: {config['llm_provider']}")
    print(f"LLM Model: {config['llm_model']}")
    print(f"Embedding Model: {config['embedding_model_name']}")
    print(f"Vector DB Path: {config['vector_db_path']}")
    print(f"Manifest Path: {config['manifest_path']}")
    print(f"HelpSteer Enabled: {config['enable_helpsteer']}")
    
    if config['llm_provider'] == 'azure':
        print(f"Azure Endpoint: {config['azure_endpoint']}")
        print(f"Azure Deployment: {config['azure_deployment_name']}")
        print(f"Azure API Version: {config['azure_api_version']}")
    
    print("=" * 50)


def example_usage():
    """
    Usage examples
    """
    print("RAG System Usage Examples")
    print("=" * 50)
    
    # Load configuration from environment variables
    print("Loading configuration from environment variables...")
    config = load_config_from_env()
    
    # Validate configuration
    if not validate_config(config):
        print("Configuration validation failed. Please check your environment variables.")
        return
    
    # Print configuration summary
    print_config_summary(config)
    
    # 1. Initialize RAG system
    print("\nInitializing RAG system...")
    rag = RAGSystem(
        embedding_model_name=config['embedding_model_name'],
        llm_provider=config['llm_provider'],
        llm_model=config['llm_model'],
        vector_db_path=config['vector_db_path'],
        manifest_path=config['manifest_path'],
        enable_helpsteer=config['enable_helpsteer'],
        enable_query_understanding = False
    )
    
    # 2. Update document library (if Sample folder exists)
    sample_folder = "./Sample"
    if os.path.exists(sample_folder):
        print("\nUpdating document library...")
        rag.update_documents(sample_folder)
    else:
        print(f"\nSample folder does not exist, skipping document update")
    
    # 3. Example queries: need to rewrite, expectation: customer can ask in the terminal at this time
    example_queries = [
        "Can you help me find an example of Low-Power Wide-Area Network technology? And explain it in details.",
        # "你能帮我找到一个低功耗广域网技术的例子吗？并详细解释它。"
        ]
    
    print(f"\nExecuting example queries...")
    for i, query in enumerate(example_queries, 1):
        print(f"\n--- Example {i} ---")
        print(f"Query: {query}")
        
        # Execute query
        result = rag.query(query)
        
        # Display results
        print(f"Answer: {result['answer']}")
        if result['keywords']:
            print(f"Keywords: {result['keywords']}")
        if result['documents']:
            print(f"Relevant documents count: {len(result['documents'])}")
        
        print("-" * 30)


def interactive_mode():
    """
    Interactive mode
    """
    print("Interactive Query Mode")
    print("=" * 50)
    
    # Load configuration from environment variables
    print("Loading configuration from environment variables...")
    config = load_config_from_env()
    
    # Validate configuration
    if not validate_config(config):
        print("Configuration validation failed. Please check your environment variables.")
        return
    
    # Print configuration summary
    print_config_summary(config)
    
    # Initialize system
    print("\nInitializing RAG system...")
    rag = RAGSystem(
        embedding_model_name=config['embedding_model_name'],
        llm_provider=config['llm_provider'],
        llm_model=config['llm_model'],
        vector_db_path=config['vector_db_path'],
        manifest_path=config['manifest_path'],
        enable_helpsteer=config['enable_helpsteer']
    )
    
    # Update documents
    sample_folder = "./Sample"
    if os.path.exists(sample_folder):
        rag.update_documents(sample_folder)
    
    print("\n💬 Start interactive query (enter 'quit' to exit):")
    
    while True:
        try:
            user_input = input("\nPlease enter your question: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Execute query
            result = rag.query(user_input)
            
            # Display results
            print(f"\nAnswer:")
            print(f"{result['answer']}")
            
            if result['keywords']:
                print(f"\nExtracted keywords: {result['keywords']}")
            
            if result['documents']:
                print(f"\nRelevant documents:")
                for i, doc in enumerate(result['documents'], 1):
                    print(f"  {i}. {doc['source']} (Relevance: {doc['score']:.2f})")
                    print(f"     {doc['content']}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")


def batch_query_mode():
    """
    Batch query mode
    """
    print("Batch Query Mode")
    print("=" * 50)
    
    # Load configuration from environment variables
    print("Loading configuration from environment variables...")
    config = load_config_from_env()
    
    # Validate configuration
    if not validate_config(config):
        print("Configuration validation failed. Please check your environment variables.")
        return
    
    # Print configuration summary
    print_config_summary(config)
    
    # Initialize system
    print("\nInitializing RAG system...")
    rag = RAGSystem(
        embedding_model_name=config['embedding_model_name'],
        llm_provider=config['llm_provider'],
        llm_model=config['llm_model'],
        vector_db_path=config['vector_db_path'],
        manifest_path=config['manifest_path'],
        enable_helpsteer=config['enable_helpsteer']
    )
    
    # Update documents
    sample_folder = "./Sample"
    if os.path.exists(sample_folder):
        rag.update_documents(sample_folder)
    
    # Batch query list
    batch_queries = [
        "show me papers related to AI in education.",
        "what are the benefits of physical education and school sport?",
        "any challenges of integrating Education with AI",
        "What is KDI",
        "Can you help me find an example of Low-Power Wide-Area Network technology? And explain it in details.",
        "What are the applications of MCP?",
        "Can you explain LSH attention based on mathematic function?",
        "What is background?",
        "Can you recommend me something related with math education?",
        "Let's talk about transformer.",
        "请展示与教育中人工智能相关的论文。",
        "体育教育和学校运动有什么好处？",
        "将教育与人工智能整合有什么挑战？",
        "什么是KDI？",
        "你能帮我找到一个低功耗广域网技术的例子吗？并详细解释它。",
        "MCP有哪些应用？",
        "你能基于数学函数解释LSH注意力机制吗？",
        "什么是背景？",
        "你能推荐一些与数学教育相关的内容吗？",
        "让我们谈谈transformer。"
    ]
    
    results = []
    
    print(f"\nExecuting batch queries ({len(batch_queries)} questions)...")
    start_time = time.time()
    for i, query in enumerate(batch_queries, 1):
        
        print(f"\nProcessing query {i}/{len(batch_queries)}: {query}")
        
        # Execute query
        result = rag.query(query)
        print("Complete")
        
        # Process keywords
        results.append({
            "query": query,
            "answer": result['answer'],
            "keywords": result['keywords'],
            "status": result['status']
        })
        
    end_time = time.time()
    
    total_time = end_time - start_time
    # Display result summary
    print(f"\nBatch Query Result Summary:")
    print(f"Total queries: {len(results)}")
    print(f"Successful queries: {len([r for r in results if r['status'] == 'success'])}")
    print(f"Failed queries: {len([r for r in results if r['status'] != 'success'])}")
    print(f"Time taken: {total_time}")
    
    # Save results to file
    import json
    with open("batch_query_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to batch_query_results.json")

def update_content():
    """
    Update document library
    """

    print("Batch Query Mode")
    print("=" * 50)
    
    # Load configuration from environment variables
    print("Loading configuration from environment variables...")
    config = load_config_from_env()
    
    # Validate configuration
    if not validate_config(config):
        print("Configuration validation failed. Please check your environment variables.")
        return
    rag = RAGSystem(
        embedding_model_name=config['embedding_model_name'],
        llm_provider=config['llm_provider'],
        llm_model=config['llm_model'],
        vector_db_path=config['vector_db_path'],
        manifest_path=config['manifest_path'],
        enable_helpsteer=config['enable_helpsteer']
    )
    
    rag.update_documents("./Sample")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "example":
            example_usage()
        elif mode == "interactive":
            interactive_mode()
        elif mode == "batch":
            batch_query_mode()
        # elif mode == "create-env":
        #     create_sample_env_file()
        elif mode == "update":
            update_content()
        else:
            print("Invalid mode. Please use: example, interactive, batch, or create-env")
    else:
        print("RAG System Usage Examples")
        print("=" * 50)
        print("Usage:")
        print("  python RAGUsage.py example     - Run example queries")
        print("  python RAGUsage.py interactive  - Interactive query")
        print("  python RAGUsage.py batch        - Batch query")
        # print("  python RAGUsage.py create-env   - Create sample .env file")
        print("\nRunning example queries by default...")
        example_usage() 