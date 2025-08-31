#!/usr/bin/env python3
"""
RAG System Startup Script
Provides a simple command-line interface to start and use the RAG system
"""

import os
import sys
from pathlib import Path

def check_dependencies():
    """Check if necessary dependencies are installed"""
    required_packages = [
        'langchain',
        'langchain-community', 
        'chromadb',
        'sentence-transformers',
        'langchain-openai',
        'langchain-deepseek'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("  Missing the following dependency packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nPlease run the following command to install:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("  All dependency packages are installed")
    return True

def check_config():
    """Check configuration file"""
    config_file = Path("config.py")
    if not config_file.exists():
        print("  Configuration file config.py does not exist")
        return False
    
    # Check environment variables
    required_env_vars = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"]
    missing_vars = []
    
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("  The following environment variables are not set:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these variables in config.py")
        return False
    
    print("  Configuration check passed")
    return True

def show_menu():
    """Display main menu"""
    print("\n" + "="*50)
    print(" RAG System - Retrieval-Augmented Generation")
    print("="*50)
    print("Please select running mode:")
    print("1. Interactive Query (Recommended)")
    print("2. Example Query")
    print("3. Batch Query")
    print("4. System Information")
    print("5. Update Document Library")
    print("0. Exit")
    print("="*50)

def run_interactive():
    """Run interactive query"""
    print("\n Starting interactive query mode...")
    try:
        from Project.RAGUsage import interactive_mode
        interactive_mode()
    except Exception as e:
        print(f" Failed to start interactive mode: {str(e)}")

def run_example():
    """Run example query"""
    print("\n Running example queries...")
    try:
        from Project.RAGUsage import example_usage
        example_usage()
    except Exception as e:
        print(f" Failed to run example queries: {str(e)}")

def run_batch():
    """Run batch query"""
    print("\n Running batch queries...")
    try:
        from Project.RAGUsage import batch_query_mode
        batch_query_mode()
    except Exception as e:
        print(f" Failed to run batch queries: {str(e)}")

def show_system_info():
    """Display system information"""
    print("\n System Information:")
    try:
        from AlphabetIntern.Project.RAGSystem import RAGSystem
        rag = RAGSystem()
        info = rag.get_system_info()
        
        for key, value in info.items():
            if key == "documents":
                print(f"  {key}: {len(value)} documents")
                for doc in value[:3]:  # Only show first 3
                    print(f"    - {Path(doc).name}")
                if len(value) > 3:
                    print(f"    ... and {len(value) - 3} more documents")
            else:
                print(f"  {key}: {value}")
                
    except Exception as e:
        print(f" Failed to get system information: {str(e)}")

def update_documents():
    """Update document library"""
    print("\n Updating document library...")
    try:
        from AlphabetIntern.Project.RAGSystem import RAGSystem
        rag = RAGSystem()
        
        sample_folder = "./Sample"
        if os.path.exists(sample_folder):
            rag.update_documents(sample_folder)
            print(" Document library update complete!")
        else:
            print(f"  Sample folder does not exist: {sample_folder}")
            print("Please put documents in the Sample folder")
            
    except Exception as e:
        print(f" Failed to update document library: {str(e)}")

def main():
    """Main function"""
    print(" Checking system environment...")

    # Check dependencies
    if not check_dependencies():
        return
    
    # Check configuration
    if not check_config():
        print("\n Please configure API keys before running the system")
        return
    
    print(" System environment check complete!")
    
    # Main loop
    while True:
        show_menu()
        
        try:
            choice = input("\nPlease enter your choice (0-5): ").strip()
            
            if choice == "0":
                print(" Goodbye!")
                break
            elif choice == "1":
                run_interactive()
            elif choice == "2":
                run_example()
            elif choice == "3":
                run_batch()
            elif choice == "4":
                show_system_info()
            elif choice == "5":
                update_documents()
            else:
                print(" Invalid choice, please re-enter")
                
        except KeyboardInterrupt:
            print("\n Goodbye!")
            break
        except Exception as e:
            print(f" An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 