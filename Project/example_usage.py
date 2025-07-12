#!/usr/bin/env python3
"""
RAG系统使用示例
演示如何使用完整的RAG系统进行文档查询
"""

from main import RAGSystem
import os

def example_usage():
    """
    使用示例
    """
    print("🚀 RAG系统使用示例")
    print("=" * 50)
    
    # 1. 初始化RAG系统
    print("1️⃣ 初始化RAG系统...")
    rag = RAGSystem(
        embedding_model_name="all-MiniLM-L6-v2",
        llm_provider="azure",
        llm_model="gpt-4",
        vector_db_path="./vector_db",
        manifest_path="manifest.json"
    )
    
    # 2. 更新文档库（如果有Sample文件夹）
    sample_folder = "./Sample"
    if os.path.exists(sample_folder):
        print("\n2️⃣ 更新文档库...")
        rag.update_documents(sample_folder)
    else:
        print(f"\n⚠️  Sample文件夹不存在，跳过文档更新")
    
    # 3. 示例查询
    example_queries = [
        "什么是机器学习？",
        # "深度学习与传统机器学习有什么区别？",
        # "神经网络的基本原理是什么？",
        # "强化学习在游戏中的应用",
        # "自然语言处理的主要技术"
        "what is machine learning?"
    ]
    
    print(f"\n3️⃣ 执行示例查询...")
    for i, query in enumerate(example_queries, 1):
        print(f"\n--- 示例 {i} ---")
        print(f"查询: {query}")
        
        # 执行查询
        result = rag.query(query)
        
        # 显示结果
        print(f"回答: {result['answer']}")
        if result['keywords']:
            print(f"关键词: {result['keywords']}")
        if result['documents']:
            print(f"相关文档数量: {len(result['documents'])}")
        
        print("-" * 30)

def interactive_mode():
    """
    交互模式
    """
    print("🎯 交互式查询模式")
    print("=" * 50)
    
    # 初始化系统
    rag = RAGSystem(
        embedding_model_name="all-MiniLM-L6-v2",
        llm_provider="azure",
        llm_model="gpt-4",
        vector_db_path="./vector_db",
        manifest_path="manifest.json"
    )
    
    # 更新文档
    sample_folder = "./Sample"
    if os.path.exists(sample_folder):
        rag.update_documents(sample_folder)
    
    print("\n💬 开始交互式查询 (输入 'quit' 退出):")
    
    while True:
        try:
            user_input = input("\n🤔 请输入您的问题: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("👋 再见!")
                break
            
            if not user_input:
                continue
            
            # 执行查询
            result = rag.query(user_input)
            
            # 显示结果
            print(f"\n💡 回答:")
            print(f"{result['answer']}")
            
            if result['keywords']:
                print(f"\n📝 提取的关键词: {result['keywords']}")
            
            if result['documents']:
                print(f"\n📄 相关文档:")
                for i, doc in enumerate(result['documents'], 1):
                    print(f"  {i}. {doc['source']} (相关性: {doc['score']:.2f})")
                    print(f"     {doc['content']}")
            
        except KeyboardInterrupt:
            print("\n👋 再见!")
            break
        except Exception as e:
            print(f"❌ 错误: {str(e)}")

def batch_query_mode():
    """
    批量查询模式
    """
    print("📋 批量查询模式")
    print("=" * 50)
    
    # 初始化系统
    rag = RAGSystem(
        embedding_model_name="all-MiniLM-L6-v2",
        llm_provider="azure",
        llm_model="gpt-4",
        vector_db_path="./vector_db",
        manifest_path="manifest.json"
    )
    
    # 更新文档
    sample_folder = "./Sample"
    if os.path.exists(sample_folder):
        rag.update_documents(sample_folder)
    
    # 批量查询列表
    batch_queries = [
        "人工智能的发展历史",
        "机器学习的应用领域",
        "深度学习的基本概念",
        "神经网络的结构",
        "强化学习的原理"
    ]
    
    results = []
    
    print(f"\n📝 执行批量查询 ({len(batch_queries)} 个问题)...")
    
    for i, query in enumerate(batch_queries, 1):
        print(f"\n处理查询 {i}/{len(batch_queries)}: {query}")
        
        result = rag.query(query)
        results.append({
            "query": query,
            "answer": result['answer'],
            "keywords": result['keywords'],
            "status": result['status']
        })
        
        print(f"✅ 完成")
    
    # 显示结果摘要
    print(f"\n📊 批量查询结果摘要:")
    print(f"总查询数: {len(results)}")
    print(f"成功查询数: {len([r for r in results if r['status'] == 'success'])}")
    print(f"失败查询数: {len([r for r in results if r['status'] != 'success'])}")
    
    # 保存结果到文件
    import json
    with open("batch_query_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 结果已保存到 batch_query_results.json")

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
            print("❌ 无效的模式。请使用: example, interactive, 或 batch")
    else:
        print("🎯 RAG系统使用示例")
        print("=" * 50)
        print("使用方法:")
        print("  python example_usage.py example    - 运行示例查询")
        print("  python example_usage.py interactive - 交互式查询")
        print("  python example_usage.py batch      - 批量查询")
        print("\n默认运行示例查询...")
        example_usage() 