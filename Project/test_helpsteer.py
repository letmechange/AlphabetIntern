#!/usr/bin/env python3
"""
HelpSteer Functionality Test Script
Verifies that the HelpSteer system is working properly
"""

import os
import sys
from typing import Dict, Any

# Import project modules
from config import *
from LLMClient import LLM_Client
from HelpSteer import HelpSteerSystem, EvaluationDimension


def test_llm_connection():
    """Test LLM connection"""
    print("=== Testing LLM Connection ===")
    
    try:
        # Check environment variables
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        
        if not api_key or not endpoint:
            print("❌ Missing necessary environment variables")
            print("Please set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT")
            return False
        
        # Initialize LLM client
        llm_client = LLM_Client(
            api_key=api_key,
            model_name="gpt-4",
            provider="azure",
            endpoint=endpoint,
            deployment_name="gpt-4",
            api_version="2025-01-01-preview"
        )
        
        # Test simple call
        test_prompt = "请回答：1+1等于几？"
        response = llm_client.call_llm_api(test_prompt)
        
        if response and len(response.strip()) > 0:
            print("✅ LLM connection test successful")
            print(f"Test response: {response.strip()}")
            return llm_client
        else:
            print("❌ LLM response is empty")
            return False
            
    except Exception as e:
        print(f"❌ LLM connection test failed: {e}")
        return False


def test_helpsteer_initialization(llm_client):
    """Test HelpSteer initialization"""
    print("\n=== Testing HelpSteer Initialization ===")
    
    try:
        helpsteer = HelpSteerSystem(llm_client)
        print("✅ HelpSteer system initialization successful")
        return helpsteer
    except Exception as e:
        print(f"❌ HelpSteer initialization failed: {e}")
        return None


def test_evaluation(helpsteer):
    """Test evaluation functionality"""
    print("\n=== Testing Evaluation Functionality ===")
    
    try:
        # Test data
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
        
        # Execute evaluation
        evaluation = helpsteer.evaluator.evaluate_response(query, response, context)
        
        # Verify results
        if hasattr(evaluation, 'overall_score') and hasattr(evaluation, 'scores'):
            print("✅ Evaluation functionality test successful")
            print(f"Overall Score: {evaluation.overall_score}/10")
            print("Dimension Scores:")
            for dim, score in evaluation.scores.items():
                print(f"  {dim.value}: {score}/10")
            return True
        else:
            print("❌ Evaluation result format incorrect")
            return False
            
    except Exception as e:
        print(f"❌ Evaluation functionality test failed: {e}")
        return False


def test_improvement(helpsteer):
    """Test improvement functionality"""
    print("\n=== Testing Improvement Functionality ===")
    
    try:
        # Test data
        query = "解释机器学习的基本概念"
        context = """
        机器学习是人工智能的一个分支，它使计算机能够在没有明确编程的情况下学习和改进。
        机器学习算法通过分析数据来识别模式，并使用这些模式来做出预测或决策。
        常见的机器学习类型包括监督学习、无监督学习和强化学习。
        """
        response = """
        机器学习是一种让电脑学习的方法。
        它可以从数据中学习，然后做出预测。
        有很多种机器学习算法。
        """
        
        # Specify improvement targets
        improvement_targets = ["clarity", "helpfulness"]
        
        # Execute improvement
        result = helpsteer.evaluate_and_improve(
            query, context, response, improvement_targets
        )
        
        # Verify results
        if result and "improved_response" in result:
            print("✅ Improvement functionality test successful")
            print(f"Original Response Score: {result['evaluation'].overall_score}/10")
            
            if result["improved_response"]:
                print(f"Improved Response Score: {result['improved_evaluation'].overall_score}/10")
                print(f"Score Improvement: {result['improvement_analysis']['score_improvement']:.2f}")
                print("Improved Response:")
                print(result["improved_response"])
            return True
        else:
            print("❌ Improvement result format incorrect")
            return False
            
    except Exception as e:
        print(f"❌ Improvement functionality test failed: {e}")
        return False


def test_preference_learning(helpsteer):
    """Test preference learning functionality"""
    print("\n=== Testing Preference Learning Functionality ===")
    
    try:
        # Test data
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
        
        # Generate preference data
        preference_data = helpsteer.trainer.generate_preference_data(
            query, context, response_a, response_b
        )
        
        # Verify results
        if preference_data and "better_response" in preference_data:
            print("✅ Preference learning functionality test successful")
            print(f"Better Response Score: {preference_data['better_score']}/10")
            print(f"Worse Response Score: {preference_data['worse_score']}/10")
            print(f"Score Difference: {preference_data['score_difference']:.2f}")
            return True
        else:
            print("❌ Preference data format incorrect")
            return False
            
    except Exception as e:
        print(f"❌ Preference learning functionality test failed: {e}")
        return False


def test_batch_processing(helpsteer):
    """Test batch processing functionality"""
    print("\n=== Testing Batch Processing Functionality ===")
    
    try:
        # Test data
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
        
        results = []
        for i, test_case in enumerate(test_cases, 1):
            print(f"Processing test case {i}...")
            
            evaluation = helpsteer.evaluator.evaluate_response(
                test_case["query"], 
                test_case["response"], 
                test_case["context"]
            )
            
            results.append({
                "case": i,
                "score": evaluation.overall_score
            })
        
        # Verify results
        if len(results) == len(test_cases):
            print("✅ Batch processing functionality test successful")
            for result in results:
                print(f"Case {result['case']}: {result['score']:.2f}/10")
            return True
        else:
            print("❌ Batch processing results incomplete")
            return False
            
    except Exception as e:
        print(f"❌ Batch processing functionality test failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("HelpSteer Functionality Testing")
    print("=" * 50)
    
    # Test result statistics
    test_results = {}
    
    # 1. Test LLM connection
    llm_client = test_llm_connection()
    test_results["llm_connection"] = llm_client is not False
    
    if not llm_client:
        print("\n❌ LLM connection failed, cannot continue testing")
        return test_results
    
    # 2. Test HelpSteer initialization
    helpsteer = test_helpsteer_initialization(llm_client)
    test_results["helpsteer_init"] = helpsteer is not None
    
    if not helpsteer:
        print("\n❌ HelpSteer initialization failed, cannot continue testing")
        return test_results
    
    # 3. Test evaluation functionality
    test_results["evaluation"] = test_evaluation(helpsteer)
    
    # 4. Test improvement functionality
    test_results["improvement"] = test_improvement(helpsteer)
    
    # 5. Test preference learning functionality
    test_results["preference_learning"] = test_preference_learning(helpsteer)
    
    # 6. Test batch processing functionality
    test_results["batch_processing"] = test_batch_processing(helpsteer)
    
    # Display test summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    for test_name, result in test_results.items():
        status = "✅ Passed" if result else "❌ Failed"
        print(f"  {test_name}: {status}")
    
    print(f"\nOverall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! HelpSteer system is working properly.")
    else:
        print("⚠️  Some tests failed, please check related functionality.")
    
    return test_results


def main():
    """Main function"""
    try:
        results = run_all_tests()
        
        # Determine exit code based on test results
        if all(results.values()):
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Failure
            
    except KeyboardInterrupt:
        print("\nTesting interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 