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
from HelpSteer import HelpSteerSystem, EvaluationDimension

# 导入LangChain组件
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


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
                 enable_helpsteer: bool = False,
                 **kwargs):
        """
        初始化RAG系统
        
        Args:
            embedding_model_name: 嵌入模型名称
            llm_provider: LLM提供商 (azure, openai, deepseek)
            llm_model: LLM模型名称
            vector_db_path: 向量数据库路径
            manifest_path: 文档清单文件路径
            enable_helpsteer: 是否启用HelpSteer功能
            **kwargs: 其他参数
        """
        self.vector_db_path = vector_db_path
        self.manifest_path = manifest_path
        self.enable_helpsteer = enable_helpsteer
        
        # 初始化嵌入模型
        print("初始化嵌入模型...")
        self.embedding_model = Embedding(
            provider="huggingface",
            model_name=embedding_model_name
        ).get_model()
        
        # 初始化LLM客户端
        print("初始化LLM客户端...")
        if llm_provider == "azure":
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            self.llm_client = LLM_Client(
                api_key=api_key,
                model_name=llm_model,
                provider=llm_provider,
                endpoint=endpoint,
                deployment_name=llm_model,
                api_version="2025-01-01-preview"
            )
        elif llm_provider == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY")
            self.llm_client = LLM_Client(
                api_key=api_key,
                model_name=llm_model,
                provider=llm_provider
            )
        else:
            raise ValueError(f"不支持的LLM提供商: {llm_provider}")
        
        # 初始化向量数据库
        print("初始化向量数据库...")
        self.vectorstore = Chroma(
            persist_directory=vector_db_path,
            embedding_function=self.embedding_model
        )
        
        # 初始化查询理解模块
        print("初始化查询理解模块...")
        self.query_understanding = QueryUnderstanding(self.llm_client)
        
        # 初始化重排序模块
        print("初始化重排序模块...")
        self.reranker = Reranker(self.llm_client)
        
        # 初始化HelpSteer系统（如果启用）
        if self.enable_helpsteer:
            print("初始化HelpSteer系统...")
            self.helpsteer = HelpSteerSystem(self.llm_client)
        else:
            self.helpsteer = None
        
        print("RAG系统初始化完成!")
    
    def update_documents(self, folder_path: str):
        """更新文档库"""
        print(f"更新文档库: {folder_path}")
        
        if not os.path.exists(folder_path):
            print(f"文件夹不存在: {folder_path}")
            return
        
        update_vectorstore_from_folder(
            folder_path=folder_path,
            vectorstore=self.vectorstore,
            embedding_model=self.embedding_model,
            manifest_path=self.manifest_path
        )
        
        print("文档库更新完成!")
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return {
            "embedding_model": self.embedding_model.__class__.__name__,
            "llm_provider": self.llm_client.provider,
            "llm_model": self.llm_client.model_name,
            "vector_db_path": self.vector_db_path,
            "helpsteer_enabled": self.enable_helpsteer,
            "document_count": len(self.vectorstore.get()["documents"]) if self.vectorstore.get()["documents"] else 0
        }
    
    def search_documents(self, query: str, top_k: int = 10) -> List[Document]:
        """搜索相关文档"""
        print(f"搜索文档: {query}")
        
        # 使用查询理解提取关键词
        keywords = self.query_understanding.extract_keywords(query)
        print(f"提取的关键词: {keywords}")
        
        # 使用关键词进行向量搜索
        docs = self.vectorstore.similarity_search(keywords, k=top_k)
        print(f"找到 {len(docs)} 个相关文档")
        
        return docs
    
    def rerank_documents(self, query: str, docs: List[Document]) -> List[tuple]:
        """重排序文档"""
        print("重排序文档...")
        
        # 提取文档内容
        doc_contents = [doc.page_content for doc in docs]
        
        # 重排序
        reranked_docs = self.reranker.rerank(query, doc_contents)
        
        print(f"重排序完成，返回前 {len(reranked_docs)} 个文档")
        return reranked_docs
    
    def generate_answer(self, query: str, top_docs: List[Document]) -> str:
        """生成回答"""
        print("生成回答...")
        
        answer = build_answer(self.llm_client, query, top_docs)
        
        print("回答生成完成!")
        return answer
    
    def evaluate_response(self, query: str, response: str, context: str) -> Dict[str, Any]:
        """评估响应质量（如果启用HelpSteer）"""
        if not self.enable_helpsteer or not self.helpsteer:
            return {"error": "HelpSteer未启用"}
        
        print("评估响应质量...")
        evaluation = self.helpsteer.evaluator.evaluate_response(query, response, context)
        
        return {
            "overall_score": evaluation.overall_score,
            "dimension_scores": {dim.value: score for dim, score in evaluation.scores.items()},
            "metadata": evaluation.metadata
        }
    
    def improve_response(self, query: str, response: str, context: str, 
                        improvement_targets: List[str] = None) -> Dict[str, Any]:
        """改进响应（如果启用HelpSteer）"""
        if not self.enable_helpsteer or not self.helpsteer:
            return {"error": "HelpSteer未启用"}
        
        print("改进响应...")
        result = self.helpsteer.evaluate_and_improve(
            query, context, response, improvement_targets
        )
        
        return result
    
    def query(self, user_query: str, top_k: int = 10, 
              enable_evaluation: bool = False,
              enable_improvement: bool = False,
              improvement_targets: List[str] = None) -> Dict[str, Any]:
        """完整的查询流程"""
        print(f"\n{'='*50}")
        print(f"用户查询: {user_query}")
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
            
            # 6. 构建上下文（用于评估）
            context = "\n\n".join([doc.page_content for doc in top_reranked_docs])
            
            # 7. 响应评估（如果启用）
            evaluation_result = None
            if enable_evaluation and self.enable_helpsteer:
                evaluation_result = self.evaluate_response(user_query, answer, context)
            
            # 8. 响应改进（如果启用）
            improvement_result = None
            if enable_improvement and self.enable_helpsteer:
                improvement_result = self.improve_response(
                    user_query, answer, context, improvement_targets
                )
                # 如果改进成功，使用改进后的回答
                if improvement_result and improvement_result.get("improved_response"):
                    answer = improvement_result["improved_response"]
            
            # 9. 构建返回结果
            result = {
                "answer": answer,
                "documents": [
                    {
                        "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                        "source": doc.metadata.get("source", "unknown"),
                        "score": next((score for content, score in reranked_results if content == doc.page_content), 0.0)
                    }
                    for doc in top_reranked_docs
                ],
                "keywords": keywords,
                "status": "success"
            }
            
            # 添加评估结果
            if evaluation_result:
                result["evaluation"] = evaluation_result
            
            # 添加改进结果
            if improvement_result:
                result["improvement"] = improvement_result
            
            return result
            
        except Exception as e:
            print(f"查询过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return {
                "answer": f"抱歉，处理您的查询时出现了错误: {str(e)}",
                "documents": [],
                "keywords": "",
                "status": "error",
                "error": str(e)
            }


def main():
    """主函数"""
    print("启动RAG系统...")
    
    # 初始化RAG系统（启用HelpSteer）
    rag_system = RAGSystem(
        embedding_model_name="all-MiniLM-L6-v2",
        llm_provider="azure",
        llm_model="gpt-4",
        vector_db_path="./vector_db",
        manifest_path="manifest.json",
        enable_helpsteer=True  # 启用HelpSteer功能
    )
    
    # 更新文档库（如果需要）
    sample_folder = "./Sample"
    if os.path.exists(sample_folder):
        print(f"\n更新文档库...")
        rag_system.update_documents(sample_folder)
    
    # 获取系统信息
    print(f"\n系统信息:")
    system_info = rag_system.get_system_info()
    for key, value in system_info.items():
        print(f"  {key}: {value}")
    
    # 交互式查询
    print(f"\n开始交互式查询 (输入 'quit' 退出):")
    print("特殊命令:")
    print("  'eval' - 启用响应评估")
    print("  'improve' - 启用响应改进")
    print("  'both' - 同时启用评估和改进")
    
    while True:
        try:
            user_input = input("\n请输入您的问题: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("再见!")
                break
            
            if not user_input:
                continue
            
            # 检查特殊命令
            enable_evaluation = False
            enable_improvement = False
            improvement_targets = None
            
            if user_input.lower() == 'eval':
                enable_evaluation = True
                user_input = input("请输入您的问题: ").strip()
            elif user_input.lower() == 'improve':
                enable_improvement = True
                improvement_targets = ["helpfulness", "clarity", "correctness"]
                user_input = input("请输入您的问题: ").strip()
            elif user_input.lower() == 'both':
                enable_evaluation = True
                enable_improvement = True
                improvement_targets = ["helpfulness", "clarity", "correctness"]
                user_input = input("请输入您的问题: ").strip()
            
            # 执行查询
            result = rag_system.query(
                user_input, 
                enable_evaluation=enable_evaluation,
                enable_improvement=enable_improvement,
                improvement_targets=improvement_targets
            )
            
            # 显示结果
            print(f"\n回答:")
            print(f"{result['answer']}")
            
            if result.get('keywords'):
                print(f"\n提取的关键词: {result['keywords']}")
            
            if result.get('evaluation'):
                print(f"\n响应评估:")
                eval_data = result['evaluation']
                print(f"  综合分数: {eval_data['overall_score']}/10")
                for dim, score in eval_data['dimension_scores'].items():
                    print(f"  {dim}: {score}/10")
            
            if result.get('improvement'):
                print(f"\n响应改进:")
                imp_data = result['improvement']
                if imp_data.get('improvement_analysis'):
                    analysis = imp_data['improvement_analysis']
                    print(f"  分数提升: {analysis['score_improvement']:.2f}")
                    for dim, improvement in analysis['dimension_improvements'].items():
                        print(f"  {dim}: {improvement:+.2f}")
            
            if result.get('documents'):
                print(f"\n相关文档:")
                for i, doc in enumerate(result['documents'], 1):
                    print(f"  {i}. {doc['source']} (相关性: {doc['score']:.2f})")
                    print(f"     {doc['content']}")
            
        except KeyboardInterrupt:
            print("\n再见!")
            break
        except Exception as e:
            print(f"错误: {str(e)}")


if __name__ == "__main__":
    main()