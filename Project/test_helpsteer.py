#!/usr/bin/env python3
"""
HelpSteer功能测试脚本
验证HelpSteer系统是否正常工作
"""

import os
import sys
from typing import Dict, Any

# 导入项目模块
from config import *
from LLMClient import LLM_Client
from HelpSteer import HelpSteerSystem, EvaluationDimension


def test_llm_connection():
    """测试LLM连接"""
    print("=== 测试LLM连接 ===")
    
    try:
        # 检查环境变量
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        
        if not api_key or not endpoint:
            print("❌ 缺少必要的环境变量")
            print("请设置 AZURE_OPENAI_API_KEY 和 AZURE_OPENAI_ENDPOINT")
            return False
        
        # 初始化LLM客户端
        llm_client = LLM_Client(
            api_key=api_key,
            model_name="gpt-4",
            provider="azure",
            endpoint=endpoint,
            deployment_name="gpt-4",
            api_version="2025-01-01-preview"
        )
        
        # 测试简单调用
        test_prompt = "请回答：1+1等于几？"
        response = llm_client.call_llm_api(test_prompt)
        
        if response and len(response.strip()) > 0:
            print("✅ LLM连接测试成功")
            print(f"测试响应: {response.strip()}")
            return llm_client
        else:
            print("❌ LLM响应为空")
            return False
            
    except Exception as e:
        print(f"❌ LLM连接测试失败: {e}")
        return False


def test_helpsteer_initialization(llm_client):
    """测试HelpSteer初始化"""
    print("\n=== 测试HelpSteer初始化 ===")
    
    try:
        helpsteer = HelpSteerSystem(llm_client)
        print("✅ HelpSteer系统初始化成功")
        return helpsteer
    except Exception as e:
        print(f"❌ HelpSteer初始化失败: {e}")
        return None


def test_evaluation(helpsteer):
    """测试评估功能"""
    print("\n=== 测试评估功能 ===")
    
    try:
        # 测试数据
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
        
        # 执行评估
        evaluation = helpsteer.evaluator.evaluate_response(query, response, context)
        
        # 验证结果
        if hasattr(evaluation, 'overall_score') and hasattr(evaluation, 'scores'):
            print("✅ 评估功能测试成功")
            print(f"综合分数: {evaluation.overall_score}/10")
            print("各维度分数:")
            for dim, score in evaluation.scores.items():
                print(f"  {dim.value}: {score}/10")
            return True
        else:
            print("❌ 评估结果格式不正确")
            return False
            
    except Exception as e:
        print(f"❌ 评估功能测试失败: {e}")
        return False


def test_improvement(helpsteer):
    """测试改进功能"""
    print("\n=== 测试改进功能 ===")
    
    try:
        # 测试数据
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
        
        # 指定改进目标
        improvement_targets = ["clarity", "helpfulness"]
        
        # 执行改进
        result = helpsteer.evaluate_and_improve(
            query, context, response, improvement_targets
        )
        
        # 验证结果
        if result and "improved_response" in result:
            print("✅ 改进功能测试成功")
            print(f"原始响应分数: {result['evaluation'].overall_score}/10")
            
            if result["improved_response"]:
                print(f"改进后响应分数: {result['improved_evaluation'].overall_score}/10")
                print(f"分数提升: {result['improvement_analysis']['score_improvement']:.2f}")
                print("改进后的响应:")
                print(result["improved_response"])
            return True
        else:
            print("❌ 改进结果格式不正确")
            return False
            
    except Exception as e:
        print(f"❌ 改进功能测试失败: {e}")
        return False


def test_preference_learning(helpsteer):
    """测试偏好学习功能"""
    print("\n=== 测试偏好学习功能 ===")
    
    try:
        # 测试数据
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
        
        # 生成偏好数据
        preference_data = helpsteer.trainer.generate_preference_data(
            query, context, response_a, response_b
        )
        
        # 验证结果
        if preference_data and "better_response" in preference_data:
            print("✅ 偏好学习功能测试成功")
            print(f"更好的响应分数: {preference_data['better_score']}/10")
            print(f"较差的响应分数: {preference_data['worse_score']}/10")
            print(f"分数差异: {preference_data['score_difference']:.2f}")
            return True
        else:
            print("❌ 偏好数据格式不正确")
            return False
            
    except Exception as e:
        print(f"❌ 偏好学习功能测试失败: {e}")
        return False


def test_batch_processing(helpsteer):
    """测试批量处理功能"""
    print("\n=== 测试批量处理功能 ===")
    
    try:
        # 测试数据
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
            print(f"处理测试用例 {i}...")
            
            evaluation = helpsteer.evaluator.evaluate_response(
                test_case["query"], 
                test_case["response"], 
                test_case["context"]
            )
            
            results.append({
                "case": i,
                "score": evaluation.overall_score
            })
        
        # 验证结果
        if len(results) == len(test_cases):
            print("✅ 批量处理功能测试成功")
            for result in results:
                print(f"用例 {result['case']}: {result['score']:.2f}/10")
            return True
        else:
            print("❌ 批量处理结果不完整")
            return False
            
    except Exception as e:
        print(f"❌ 批量处理功能测试失败: {e}")
        return False


def run_all_tests():
    """运行所有测试"""
    print("HelpSteer功能测试")
    print("=" * 50)
    
    # 测试结果统计
    test_results = {}
    
    # 1. 测试LLM连接
    llm_client = test_llm_connection()
    test_results["llm_connection"] = llm_client is not False
    
    if not llm_client:
        print("\n❌ LLM连接失败，无法继续测试")
        return test_results
    
    # 2. 测试HelpSteer初始化
    helpsteer = test_helpsteer_initialization(llm_client)
    test_results["helpsteer_init"] = helpsteer is not None
    
    if not helpsteer:
        print("\n❌ HelpSteer初始化失败，无法继续测试")
        return test_results
    
    # 3. 测试评估功能
    test_results["evaluation"] = test_evaluation(helpsteer)
    
    # 4. 测试改进功能
    test_results["improvement"] = test_improvement(helpsteer)
    
    # 5. 测试偏好学习功能
    test_results["preference_learning"] = test_preference_learning(helpsteer)
    
    # 6. 测试批量处理功能
    test_results["batch_processing"] = test_batch_processing(helpsteer)
    
    # 显示测试总结
    print("\n" + "=" * 50)
    print("测试总结:")
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    for test_name, result in test_results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
    
    print(f"\n总体结果: {passed_tests}/{total_tests} 个测试通过")
    
    if passed_tests == total_tests:
        print("🎉 所有测试通过！HelpSteer系统工作正常。")
    else:
        print("⚠️  部分测试失败，请检查相关功能。")
    
    return test_results


def main():
    """主函数"""
    try:
        results = run_all_tests()
        
        # 根据测试结果决定退出码
        if all(results.values()):
            sys.exit(0)  # 成功
        else:
            sys.exit(1)  # 失败
            
    except KeyboardInterrupt:
        print("\n测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n测试过程中出现未预期的错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 