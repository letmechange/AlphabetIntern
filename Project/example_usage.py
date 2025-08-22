#!/usr/bin/env python3
"""
RAG System Usage Examples
Demonstrates how to use the complete RAG system for document queries
"""

from RAGSystem import RAGSystem
import os
import time
import re
from googletrans import Translator


# def is_chinese(text):
#     return any('\u4e00' <= ch <= '\u9fff' for ch in text)

# def translate_to_english(text):
#     translator = Translator()
#     return translator.translate(text, src='zh-cn', dest='en').text

def example_usage():
    """
    Usage examples
    """
    print("RAG System Usage Examples")
    print("=" * 50)
    
    # 1. Initialize RAG system
    print("Initializing RAG system...")
    rag = RAGSystem(
        embedding_model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',  # Multilingual model
        llm_provider="azure",
        llm_model="gpt-4",
        vector_db_path="./vector_db",
        manifest_path="manifest.json"
    )
    
    # 2. Update document library (if Sample folder exists)
    sample_folder = "./Sample"
    if os.path.exists(sample_folder):
        print("\n Updating document library...")
        rag.update_documents(sample_folder)
    else:
        print(f"\n Sample folder does not exist, skipping document update")
    
    # 3. Example queries: need to rewrite, expectation: customer can ask in the terminal at this time
    example_queries = [
        "Can you help me find an example of Low-Power Wide-Area Network technology? And explain it in details.",
        "你能帮我找到一个低功耗广域网技术的例子吗？并详细解释它。"
        ]
    
    print(f"\n Executing example queries...")
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
    
    # Initialize system
    rag = RAGSystem(
        embedding_model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',  # Multilingual model
        llm_provider="azure",
        llm_model="gpt-4",
        vector_db_path="./vector_db",
        manifest_path="manifest.json"
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
            print(f"\n Answer:")
            print(f"{result['answer']}")
            
            if result['keywords']:
                print(f"\n Extracted keywords: {result['keywords']}")
            
            if result['documents']:
                print(f"\n Relevant documents:")
                for i, doc in enumerate(result['documents'], 1):
                    print(f"  {i}. {doc['source']} (Relevance: {doc['score']:.2f})")
                    print(f"     {doc['content']}")
            
        except KeyboardInterrupt:
            print("\n Goodbye!")
            break
        except Exception as e:
            print(f" Error: {str(e)}")

def batch_query_mode():
    """
    Batch query mode
    """
    print(" Batch Query Mode")
    print("=" * 50)
    
    # Initialize system
    rag = RAGSystem(
        embedding_model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',  # Multilingual model
        llm_provider="azure",
        llm_model="gpt-4",
        vector_db_path="./vector_db",
        manifest_path="manifest.json"
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
    
    print(f"\n Executing batch queries ({len(batch_queries)} questions)...")
    start_time = time.time()
    for i, query in enumerate(batch_queries, 1):
        
        print(f"\nProcessing query {i}/{len(batch_queries)}: {query}")
        
        # # Determine if it's Chinese
        # if is_chinese(query):
        #     query_for_search = translate_to_english(query)
        #     print(1)
        # else:
        #     query_for_search = query

        # Use English query_for_search for retrieval
        result = rag.query(query)
        print(2)
        # Process keywords
        results.append({
            "query": query,
            "answer": result['answer'],
            "keywords": result['keywords'],
            "status": result['status']
        })
        print(f" Complete")
    end_time = time.time()
    
    total_time = end_time - start_time
    # Display result summary
    print(f"\n Batch Query Result Summary:")
    print(f"Total queries: {len(results)}")
    print(f"Successful queries: {len([r for r in results if r['status'] == 'success'])}")
    print(f"Failed queries: {len([r for r in results if r['status'] != 'success'])}")
    print(f"Time taken: {total_time}")
    # Save results to file
    import json
    with open("batch_query_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n Results saved to batch_query_results.json")

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
        else:
            print(" Invalid mode. Please use: example, interactive, or batch")
    else:
        print(" RAG System Usage Examples")
        print("=" * 50)
        print("Usage:")
        print("  python example_usage.py example    - Run example queries")
        print("  python example_usage.py interactive - Interactive query")
        print("  python example_usage.py batch      - Batch query")
        print("\nRunning example queries by default...")
        example_usage() 