#!/usr/bin/env python3
"""
HelpSteer Usage Examples
Demonstrates how to use the HelpSteer system for response evaluation and improvement
"""

import os
import sys
from typing import List, Dict, Any

# Import project modules
from config import *
from LLMClient import LLM_Client
from HelpSteer import HelpSteerSystem, EvaluationDimension


def setup_llm_client():
    """Setup LLM client"""
    # Use Azure OpenAI
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    
    if not api_key or not endpoint:
        print("Please configure Azure OpenAI API key and endpoint first")
        return None
    
    return LLM_Client(
        api_key=api_key,
        model_name="gpt-4",
        provider="azure",
        endpoint=endpoint,
        deployment_name="gpt-4",
        api_version="2025-01-01-preview"
    )


def example_evaluation():
    """Example: Evaluate a single response"""
    print("=== HelpSteer Response Evaluation Example ===")
    
    # Setup LLM client
    llm_client = setup_llm_client()
    if not llm_client:
        return
    
    # Initialize HelpSteer system
    helpsteer = HelpSteerSystem(llm_client)
    
    # Example data
    query = "什么是人工智能？"
    context = """
    人工智能(AI)是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。
    AI包括机器学习、自然语言处理、计算机视觉等多个子领域。
    AI技术已经广泛应用于医疗、金融、交通、娱乐等各个行业。
    """
    response = """
    人工智能是计算机科学的一个分支，它试图让计算机模拟人类的智能行为。
    这包括学习、推理、感知和语言理解等能力。
    AI技术已经在许多领域得到应用，如自动驾驶汽车、医疗诊断和推荐系统。
    """
    
    print(f"Query: {query}")
    print(f"Context: {context}")
    print(f"Response: {response}")
    print("\nStarting evaluation...")
    
    # Evaluate response
    evaluation = helpsteer.evaluator.evaluate_response(query, response, context)
    
    # Display evaluation results
    print("\n=== Evaluation Results ===")
    print(f"Overall Score: {evaluation.overall_score}/10")
    print("\nDimension Scores:")
    for dimension, score in evaluation.scores.items():
        print(f"  {dimension.value}: {score}/10")
    
    print(f"\nEvaluation Metadata: {evaluation.metadata}")


def example_improvement():
    """Example: Response improvement"""
    print("\n=== HelpSteer Response Improvement Example ===")
    
    # Setup LLM client
    llm_client = setup_llm_client()
    if not llm_client:
        return
    
    # Initialize HelpSteer system
    helpsteer = HelpSteerSystem(llm_client)
    
    # Example data
    query = "解释机器学习的基本概念"
    context = """
    机器学习是人工智能的一个分支，它使计算机能够在没有明确编程的情况下学习和改进。
    机器学习算法通过分析数据来识别模式，并使用这些模式来做出预测或决策。
    常见的机器学习类型包括监督学习、无监督学习和强化学习。
    """
    current_response = """
    机器学习是一种让电脑学习的方法。
    它可以从数据中学习，然后做出预测。
    有很多种机器学习算法。
    """
    
    print(f"Query: {query}")
    print(f"Current Response: {current_response}")
    
    # Specify improvement targets
    improvement_targets = ["clarity", "helpfulness"]
    
    print(f"\nImprovement Targets: {improvement_targets}")
    print("Starting improvement...")
    
    # Evaluate and improve
    result = helpsteer.evaluate_and_improve(
        query, context, current_response, improvement_targets
    )
    
    # Display results
    print("\n=== Improvement Results ===")
    print(f"Original Response Score: {result['evaluation'].overall_score}/10")
    
    if result["improved_response"]:
        print(f"Improved Response Score: {result['improved_evaluation'].overall_score}/10")
        print(f"Score Improvement: {result['improvement_analysis']['score_improvement']:.2f}")
        print("\nImproved Response:")
        print(result["improved_response"])
        
        print("\nDimension Improvements:")
        for dim, improvement in result["improvement_analysis"]["dimension_improvements"].items():
            print(f"  {dim}: {improvement:+.2f}")


def example_preference_learning():
    """Example: Preference learning"""
    print("\n=== HelpSteer Preference Learning Example ===")
    
    # Setup LLM client
    llm_client = setup_llm_client()
    if not llm_client:
        return
    
    # Initialize HelpSteer system
    helpsteer = HelpSteerSystem(llm_client)
    
    # Example data
    query = "什么是深度学习？"
    context = """
    深度学习是机器学习的一个分支，它使用多层神经网络来模拟人脑的学习过程。
    深度学习模型可以自动学习数据的层次化表示，从简单的特征到复杂的抽象概念。
    常见的深度学习架构包括卷积神经网络(CNN)、循环神经网络(RNN)和Transformer。
    """
    
    response_a = """
    深度学习是机器学习的一个分支，它使用多层神经网络来模拟人脑的学习过程。
    通过多层处理，深度学习模型可以自动学习数据的层次化表示，从简单的特征到复杂的抽象概念。
    这种方法在图像识别、自然语言处理等领域取得了突破性进展。
    """
    
    response_b = """
    深度学习就是用很多层的神经网络。
    每一层都会处理数据，然后传给下一层。
    这样可以学习到更复杂的东西。
    """
    
    print(f"Query: {query}")
    print(f"Response A: {response_a}")
    print(f"Response B: {response_b}")
    
    print("\nGenerating preference data...")
    
    # Generate preference data
    preference_data = helpsteer.trainer.generate_preference_data(
        query, context, response_a, response_b
    )
    
    # Display results
    print("\n=== Preference Analysis Results ===")
    print(f"Better Response Score: {preference_data['better_score']}/10")
    print(f"Worse Response Score: {preference_data['worse_score']}/10")
    print(f"Score Difference: {preference_data['score_difference']:.2f}")
    
    print("\nPreference Analysis:")
    print(preference_data["analysis"])
    
    # Save preference data
    output_file = "helpsteer_preference_data.json"
    helpsteer.trainer.save_preference_data(preference_data, output_file)
    print(f"\nPreference data saved to: {output_file}")


def example_batch_processing():
    """Example: Batch processing"""
    print("\n=== HelpSteer Batch Processing Example ===")
    
    # Setup LLM client
    llm_client = setup_llm_client()
    if not llm_client:
        return
    
    # Initialize HelpSteer system
    helpsteer = HelpSteerSystem(llm_client)
    
    # Example data
    test_cases = [
        {
            "query": "什么是神经网络？",
            "context": "神经网络是一种模仿生物神经系统的计算模型。",
            "response": "神经网络是一种计算模型，它模拟了生物大脑的结构。"
        },
        {
            "query": "解释监督学习",
            "context": "监督学习使用标记的训练数据来学习映射关系。",
            "response": "监督学习需要标记的数据来训练模型。"
        }
    ]
    
    print(f"Starting batch evaluation of {len(test_cases)} test cases...")
    
    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nProcessing test case {i}...")
        
        evaluation = helpsteer.evaluator.evaluate_response(
            test_case["query"], 
            test_case["response"], 
            test_case["context"]
        )
        
        results.append({
            "case": i,
            "score": evaluation.overall_score
        })
    
    # Display summary results
    print("\n=== Batch Evaluation Summary ===")
    total_score = sum(r["score"] for r in results)
    avg_score = total_score / len(results)
    
    print(f"Average Score: {avg_score:.2f}/10")
    print(f"Highest Score: {max(r['score'] for r in results):.2f}/10")
    print(f"Lowest Score: {min(r['score'] for r in results):.2f}/10")
    
    print("\nDetailed Results:")
    for result in results:
        print(f"Case {result['case']}: {result['score']:.2f}/10")


def main():
    """Main function"""
    print("HelpSteer System Usage Examples")
    print("=" * 50)
    
    # Check configuration
    if not os.getenv("AZURE_OPENAI_API_KEY"):
        print("Please configure Azure OpenAI API key first")
        print("You can use environment variables or run config.py")
        return
    
    # Run examples
    try:
        example_evaluation()
        example_improvement()
        example_preference_learning()
        example_batch_processing()
        
        print("\n" + "=" * 50)
        print("All examples completed!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 