#!/usr/bin/env python3
"""
Complete RAG System Main Process
Integrates document loading, vectorization, query understanding, reranking and answer generation
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import json

# Import project modules
from config import *
from data_store_loader import update_vectorstore_from_folder, load_manifest
from Embedding import Embedding
from LLMClient import LLM_Client
from QueryUnderstanding import QueryUnderstanding
from Reranker import Reranker
from responser import build_answer
from HelpSteer import HelpSteerSystem, EvaluationDimension

# Import LangChain components
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


class RAGSystem:
    """
    Complete RAG System class
    Integrates document processing, vector retrieval, query understanding and answer generation
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
        Initialize RAG system
        
        Args:
            embedding_model_name: Embedding model name
            llm_provider: LLM provider (azure, openai, deepseek)
            llm_model: LLM model name
            vector_db_path: Vector database path
            manifest_path: Document manifest file path
            enable_helpsteer: Whether to enable HelpSteer functionality
            **kwargs: Other parameters
        """
        self.vector_db_path = vector_db_path
        self.manifest_path = manifest_path
        self.enable_helpsteer = enable_helpsteer
        
        # Initialize embedding model
        print("Initializing embedding model...")
        self.embedding_model = Embedding(
            provider="huggingface",
            model_name=embedding_model_name
        ).get_model()
        
        # Initialize LLM client
        print("Initializing LLM client...")
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
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")
        
        # Initialize vector database
        print("Initializing vector database...")
        self.vectorstore = Chroma(
            persist_directory=vector_db_path,
            embedding_function=self.embedding_model
        )
        
        # Initialize query understanding module
        print("Initializing query understanding module...")
        self.query_understanding = QueryUnderstanding(self.llm_client)
        
        # Initialize reranking module
        print("Initializing reranking module...")
        self.reranker = Reranker(self.llm_client)
        
        # Initialize HelpSteer system (if enabled)
        if self.enable_helpsteer:
            print("Initializing HelpSteer system...")
            self.helpsteer = HelpSteerSystem(self.llm_client)
        else:
            self.helpsteer = None
        
        print("RAG system initialization complete!")
    
    def update_documents(self, folder_path: str):
        """Update document library"""
        print(f"Updating document library: {folder_path}")
        
        if not os.path.exists(folder_path):
            print(f"Folder does not exist: {folder_path}")
            return
        
        update_vectorstore_from_folder(
            folder_path=folder_path,
            vectorstore=self.vectorstore,
            embedding_model=self.embedding_model,
            manifest_path=self.manifest_path
        )
        
        print("Document library update complete!")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            "embedding_model": self.embedding_model.__class__.__name__,
            "llm_provider": self.llm_client.provider,
            "llm_model": self.llm_client.model_name,
            "vector_db_path": self.vector_db_path,
            "helpsteer_enabled": self.enable_helpsteer,
            "document_count": len(self.vectorstore.get()["documents"]) if self.vectorstore.get()["documents"] else 0
        }
    
    def search_documents(self, query: str, top_k: int = 10) -> List[Document]:
        """Search for relevant documents"""
        print(f"Searching documents: {query}")
        
        # Use query understanding to extract keywords
        keywords = self.query_understanding.extract_keywords(query)
        print(f"Extracted keywords: {keywords}")
        
        # Use keywords for vector search
        docs = self.vectorstore.similarity_search(keywords, k=top_k)
        print(f"Found {len(docs)} relevant documents")
        
        return docs
    
    def rerank_documents(self, query: str, docs: List[Document]) -> List[tuple]:
        """Rerank documents"""
        print("Reranking documents...")
        
        # Extract document content
        doc_contents = [doc.page_content for doc in docs]
        
        # Rerank
        reranked_docs = self.reranker.rerank(query, doc_contents)
        
        print(f"Reranking complete, returning top {len(reranked_docs)} documents")
        return reranked_docs
    
    def generate_answer(self, query: str, top_docs: List[Document]) -> str:
        """Generate answer"""
        print("Generating answer...")
        
        answer = build_answer(self.llm_client, query, top_docs)
        
        print("Answer generation complete!")
        return answer
    
    def evaluate_response(self, query: str, response: str, context: str) -> Dict[str, Any]:
        """Evaluate response quality (if HelpSteer is enabled)"""
        if not self.enable_helpsteer or not self.helpsteer:
            return {"error": "HelpSteer not enabled"}
        
        print("Evaluating response quality...")
        evaluation = self.helpsteer.evaluator.evaluate_response(query, response, context)
        
        return {
            "overall_score": evaluation.overall_score,
            "dimension_scores": {dim.value: score for dim, score in evaluation.scores.items()},
            "metadata": evaluation.metadata
        }
    
    def improve_response(self, query: str, response: str, context: str, 
                        improvement_targets: List[str] = None) -> Dict[str, Any]:
        """Improve response (if HelpSteer is enabled)"""
        if not self.enable_helpsteer or not self.helpsteer:
            return {"error": "HelpSteer not enabled"}
        
        print("Improving response...")
        result = self.helpsteer.evaluate_and_improve(
            query, context, response, improvement_targets
        )
        
        return result
    
    def query(self, user_query: str, top_k: int = 10, 
              enable_evaluation: bool = False,
              enable_improvement: bool = False,
              improvement_targets: List[str] = None) -> Dict[str, Any]:
        """Complete query process"""
        print(f"\n{'='*50}")
        print(f"User Query: {user_query}")
        print(f"{'='*50}")
        
        try:
            # 1. Search for relevant documents
            docs = self.search_documents(user_query, top_k)
            
            if not docs:
                return {
                    "answer": "Sorry, no relevant document information found.",
                    "documents": [],
                    "keywords": "",
                    "status": "no_docs_found"
                }
            
            # 2. Rerank documents
            reranked_results = self.rerank_documents(user_query, docs)
            
            # 3. Get reranked documents
            top_reranked_docs = []
            for doc_content, score in reranked_results:
                # Find corresponding original document object
                for doc in docs:
                    if doc.page_content == doc_content:
                        top_reranked_docs.append(doc)
                        break
            
            # 4. Generate answer
            answer = self.generate_answer(user_query, top_reranked_docs)
            
            # 5. Extract keywords
            keywords = self.query_understanding.extract_keywords(user_query)
            
            # 6. Build context (for evaluation)
            context = "\n\n".join([doc.page_content for doc in top_reranked_docs])
            
            # 7. Response evaluation (if enabled)
            evaluation_result = None
            if enable_evaluation and self.enable_helpsteer:
                evaluation_result = self.evaluate_response(user_query, answer, context)
            
            # 8. Response improvement (if enabled)
            improvement_result = None
            if enable_improvement and self.enable_helpsteer:
                improvement_result = self.improve_response(
                    user_query, answer, context, improvement_targets
                )
                # If improvement is successful, use improved answer
                if improvement_result and improvement_result.get("improved_response"):
                    answer = improvement_result["improved_response"]
            
            # 9. Build return result
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
            
            # Add evaluation results
            if evaluation_result:
                result["evaluation"] = evaluation_result
            
            # Add improvement results
            if improvement_result:
                result["improvement"] = improvement_result
            
            return result
            
        except Exception as e:
            print(f"Error during query process: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return {
                "answer": f"Sorry, an error occurred while processing your query: {str(e)}",
                "documents": [],
                "keywords": "",
                "status": "error",
                "error": str(e)
            }


def main():
    """Main function"""
    print("Starting RAG system...")
    
    # Initialize RAG system (enable HelpSteer)
    rag_system = RAGSystem(
        embedding_model_name="all-MiniLM-L6-v2",
        llm_provider="azure",
        llm_model="gpt-4",
        vector_db_path="./vector_db",
        manifest_path="manifest.json",
        enable_helpsteer=True  # Enable HelpSteer functionality
    )
    
    # Update document library (if needed)
    sample_folder = "./Sample"
    if os.path.exists(sample_folder):
        print(f"\nUpdating document library...")
        rag_system.update_documents(sample_folder)
    
    # Get system information
    print(f"\nSystem Information:")
    system_info = rag_system.get_system_info()
    for key, value in system_info.items():
        print(f"  {key}: {value}")
    
    # Interactive query
    print(f"\nStart interactive query (enter 'quit' to exit):")
    print("Special commands:")
    print("  'eval' - Enable response evaluation")
    print("  'improve' - Enable response improvement")
    print("  'both' - Enable both evaluation and improvement")
    
    while True:
        try:
            user_input = input("\nPlease enter your question: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Check special commands
            enable_evaluation = False
            enable_improvement = False
            improvement_targets = None
            
            if user_input.lower() == 'eval':
                enable_evaluation = True
                user_input = input("Please enter your question: ").strip()
            elif user_input.lower() == 'improve':
                enable_improvement = True
                improvement_targets = ["helpfulness", "clarity", "correctness"]
                user_input = input("Please enter your question: ").strip()
            elif user_input.lower() == 'both':
                enable_evaluation = True
                enable_improvement = True
                improvement_targets = ["helpfulness", "clarity", "correctness"]
                user_input = input("Please enter your question: ").strip()
            
            # Execute query
            result = rag_system.query(
                user_input, 
                enable_evaluation=enable_evaluation,
                enable_improvement=enable_improvement,
                improvement_targets=improvement_targets
            )
            
            # Display results
            print(f"\nAnswer:")
            print(f"{result['answer']}")
            
            if result.get('keywords'):
                print(f"\nExtracted keywords: {result['keywords']}")
            
            if result.get('evaluation'):
                print(f"\nResponse Evaluation:")
                eval_data = result['evaluation']
                print(f"  Overall Score: {eval_data['overall_score']}/10")
                for dim, score in eval_data['dimension_scores'].items():
                    print(f"  {dim}: {score}/10")
            
            if result.get('improvement'):
                print(f"\nResponse Improvement:")
                imp_data = result['improvement']
                if imp_data.get('improvement_analysis'):
                    analysis = imp_data['improvement_analysis']
                    print(f"  Score Improvement: {analysis['score_improvement']:.2f}")
                    for dim, improvement in analysis['dimension_improvements'].items():
                        print(f"  {dim}: {improvement:+.2f}")
            
            if result.get('documents'):
                print(f"\nRelevant Documents:")
                for i, doc in enumerate(result['documents'], 1):
                    print(f"  {i}. {doc['source']} (Relevance: {doc['score']:.2f})")
                    print(f"     {doc['content']}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()