#!/usr/bin/env python3
"""
完整的RAG系统主流程
整合了文档加载、向量化、查询理解、重排序和回答生成
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import json

# 导入项目模块
from config import *
from data_store_loader import update_vectorstore_from_folder, load_manifest
from Embedding import Embedding
from LLMClient import LLM_Client
from QueryUnderstanding import QueryUnderstanding
from Reranker import Reranker
from responser import build_answer

# 导入LangChain组件
from langchain.vectorstores import Chroma
from langchain.schema import Document


class RAGSystem:
    """
    完整的RAG系统类
    整合了文档处理、向量检索、查询理解和回答生成
    """
    
    def __init__(self, 
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 llm_provider: str = "azure",
                 llm_model: str = "gpt-4",
                 vector_db_path: str = "./vector_db",
                 manifest_path: str = "manifest.json",
                 **kwargs):
        """
        初始化RAG系统
        
        Args:
            embedding_model_name: 嵌入模型名称
            llm_provider: LLM提供商 (azure, openai, deepseek)
            llm_model: LLM模型名称
            vector_db_path: 向量数据库路径
            manifest_path: 文档清单文件路径
            **kwargs: 其他参数
        """
        self.vector_db_path = vector_db_path
        self.manifest_path = manifest_path
        
        # 初始化嵌入模型
        print("🔧 初始化嵌入模型...")
        self.embedding_model = Embedding(
            provider="huggingface",
            model_name=embedding_model_name
        ).get_model()
        
        # 初始化LLM客户端
        print("🔧 初始化LLM客户端...")
        if llm_provider == "azure":
            self.llm_client = LLM_Client(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                model_name=llm_model,
                provider=llm_provider,
                endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                deployment_name="yu-gpt-4o",
                api_version="2025-01-01-preview"
            )
        elif llm_provider == "deepseek":
            self.llm_client = LLM_Client(
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                model_name=llm_model,
                provider=llm_provider,
                max_tokens=4000
            )
        else:
            raise ValueError(f"不支持的LLM提供商: {llm_provider}")
        
        # 初始化查询理解组件
        print("🔧 初始化查询理解组件...")
        self.query_understanding = QueryUnderstanding(self.llm_client)
        
        # 初始化重排序组件
        print("🔧 初始化重排序组件...")
        self.reranker = Reranker(self.llm_client, top_k=5)
        
        # 初始化向量数据库
        print("🔧 初始化向量数据库...")
        self.vectorstore = Chroma(
            persist_directory=vector_db_path,
            embedding_function=self.embedding_model
        )
        
        print("✅ RAG系统初始化完成!")
    
    def update_documents(self, folder_path: str) -> None:
        """
        更新文档库
        
        Args:
            folder_path: 文档文件夹路径
        """
        print(f"📚 更新文档库: {folder_path}")
        update_vectorstore_from_folder(
            folder_path=folder_path,
            vectorstore=self.vectorstore,
            embedding_model=self.embedding_model,
            manifest_path=self.manifest_path
        )
        print("✅ 文档库更新完成!")
    
    def search_documents(self, query: str, top_k: int = 10) -> List[Document]:
        """
        搜索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回的文档数量
            
        Returns:
            相关文档列表
        """
        print(f"🔍 搜索文档: {query}")
        
        # 使用查询理解提取关键词
        keywords = self.query_understanding.extract_keywords(query)
        print(f"📝 提取的关键词: {keywords}")
        
        # 使用关键词进行向量搜索
        docs = self.vectorstore.similarity_search(keywords, k=top_k)
        print(f"📄 找到 {len(docs)} 个相关文档")
        
        return docs
    
    def rerank_documents(self, query: str, docs: List[Document]) -> List[tuple]:
        """
        重排序文档
        
        Args:
            query: 原始查询
            docs: 文档列表
            
        Returns:
            重排序后的文档和分数
        """
        print("🔄 重排序文档...")
        
        # 提取文档内容
        doc_contents = [doc.page_content for doc in docs]
        
        # 重排序
        reranked_docs = self.reranker.rerank(query, doc_contents)
        
        print(f"✅ 重排序完成，返回前 {len(reranked_docs)} 个文档")
        return reranked_docs
    
    def generate_answer(self, query: str, top_docs: List[Document]) -> str:
        """
        生成回答
        
        Args:
            query: 用户查询
            top_docs: 相关文档列表
            
        Returns:
            生成的回答
        """
        print("💬 生成回答...")
        
        answer = build_answer(self.llm_client, query, top_docs)
        
        print("✅ 回答生成完成!")
        return answer
    
    def query(self, user_query: str, top_k: int = 10) -> Dict[str, Any]:
        """
        完整的查询流程
        
        Args:
            user_query: 用户查询
            top_k: 检索的文档数量
            
        Returns:
            包含回答和相关信息的字典
        """
        print(f"\n{'='*50}")
        print(f"🤔 用户查询: {user_query}")
        print(f"{'='*50}")
        
        try:
            # 1. 搜索相关文档
            docs = self.search_documents(user_query, top_k)
            
            if not docs:
                return {
                    "answer": "抱歉，没有找到相关的文档信息。",
                    "documents": [],
                    "keywords": "",
                    "status": "no_docs_found"
                }
            
            # 2. 重排序文档
            reranked_results = self.rerank_documents(user_query, docs)
            
            # 3. 获取重排序后的文档
            top_reranked_docs = []
            for doc_content, score in reranked_results:
                # 找到对应的原始文档对象
                for doc in docs:
                    if doc.page_content == doc_content:
                        top_reranked_docs.append(doc)
                        break
            
            # 4. 生成回答
            answer = self.generate_answer(user_query, top_reranked_docs)
            
            # 5. 提取关键词
            keywords = self.query_understanding.extract_keywords(user_query)
            
            # 6. 构建返回结果
            result = {
                "answer": answer,
                "documents": [
                    {
                        "content": doc.page_content[:200] + "...",
                        "source": doc.metadata.get("source", "未知来源"),
                        "score": score
                    }
                    for doc, (_, score) in zip(top_reranked_docs, reranked_results)
                ],
                "keywords": keywords,
                "status": "success"
            }
            
            return result
            
        except Exception as e:
            print(f"❌ 查询过程中出现错误: {str(e)}")
            return {
                "answer": f"抱歉，处理您的查询时出现了错误: {str(e)}",
                "documents": [],
                "keywords": "",
                "status": "error",
                "error": str(e)
            }
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        获取系统信息
        
        Returns:
            系统配置信息
        """
        manifest = load_manifest(self.manifest_path)
        
        return {
            "embedding_model": self.embedding_model.model_name,
            "llm_provider": self.llm_client.provider,
            "llm_model": self.llm_client.model_name,
            "vector_db_path": self.vector_db_path,
            "document_count": len(manifest),
            "documents": list(manifest.keys())
        }


def main():
    """
    主函数 - 演示完整流程
    """
    print("🚀 启动RAG系统...")
    
    # 初始化RAG系统
    rag_system = RAGSystem(
        embedding_model_name="all-MiniLM-L6-v2",
        llm_provider="azure",
        llm_model="gpt-4",
        vector_db_path="./vector_db",
        manifest_path="manifest.json"
    )
    
    # 更新文档库（如果需要）
    sample_folder = "./Sample"
    if os.path.exists(sample_folder):
        print(f"\n📚 更新文档库...")
        rag_system.update_documents(sample_folder)
    
    # 获取系统信息
    print(f"\n📊 系统信息:")
    system_info = rag_system.get_system_info()
    for key, value in system_info.items():
        print(f"  {key}: {value}")
    
    # 交互式查询
    print(f"\n💬 开始交互式查询 (输入 'quit' 退出):")
    
    while True:
        try:
            user_input = input("\n🤔 请输入您的问题: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("👋 再见!")
                break
            
            if not user_input:
                continue
            
            # 执行查询
            result = rag_system.query(user_input)
            
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


if __name__ == "__main__":
    main()