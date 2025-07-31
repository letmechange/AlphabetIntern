#!/usr/bin/env python3
"""
HelpSteer使用示例
演示如何使用HelpSteer系统进行响应评估和改进
"""

import os
import sys
from typing import List, Dict, Any

# 导入项目模块
from config import *
from LLMClient import LLM_Client
from HelpSteer import HelpSteerSystem, EvaluationDimension


def setup_llm_client():
    """设置LLM客户端"""
    # 使用Azure OpenAI
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    
    if not api_key or not endpoint:
        print("请先配置Azure OpenAI API密钥和端点")
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
    """示例：评估单个响应"""
    print("=== HelpSteer响应评估示例 ===")
    
    # 设置LLM客户端
    llm_client = setup_llm_client()
    if not llm_client:
        return
    
    # 初始化HelpSteer系统
    helpsteer = HelpSteerSystem(llm_client)
    
    # 示例数据
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
    
    print(f"查询: {query}")
    print(f"上下文: {context}")
    print(f"响应: {response}")
    print("\n开始评估...")
    
    # 评估响应
    evaluation = helpsteer.evaluator.evaluate_response(query, response, context)
    
    # 显示评估结果
    print("\n=== 评估结果 ===")
    print(f"综合分数: {evaluation.overall_score}/10")
    print("\n各维度分数:")
    for dimension, score in evaluation.scores.items():
        print(f"  {dimension.value}: {score}/10")
    
    print(f"\n评估元数据: {evaluation.metadata}")


def example_improvement():
    """示例：响应改进"""
    print("\n=== HelpSteer响应改进示例 ===")
    
    # 设置LLM客户端
    llm_client = setup_llm_client()
    if not llm_client:
        return
    
    # 初始化HelpSteer系统
    helpsteer = HelpSteerSystem(llm_client)
    
    # 示例数据
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
    
    print(f"查询: {query}")
    print(f"当前响应: {current_response}")
    
    # 指定改进目标
    improvement_targets = ["clarity", "helpfulness"]
    
    print(f"\n改进目标: {improvement_targets}")
    print("开始改进...")
    
    # 评估并改进
    result = helpsteer.evaluate_and_improve(
        query, context, current_response, improvement_targets
    )
    
    # 显示结果
    print("\n=== 改进结果 ===")
    print(f"原始响应分数: {result['evaluation'].overall_score}/10")
    
    if result["improved_response"]:
        print(f"改进后响应分数: {result['improved_evaluation'].overall_score}/10")
        print(f"分数提升: {result['improvement_analysis']['score_improvement']:.2f}")
        print("\n改进后的响应:")
        print(result["improved_response"])
        
        print("\n各维度改进:")
        for dim, improvement in result["improvement_analysis"]["dimension_improvements"].items():
            print(f"  {dim}: {improvement:+.2f}")


def example_preference_learning():
    """示例：偏好学习"""
    print("\n=== HelpSteer偏好学习示例 ===")
    
    # 设置LLM客户端
    llm_client = setup_llm_client()
    if not llm_client:
        return
    
    # 初始化HelpSteer系统
    helpsteer = HelpSteerSystem(llm_client)
    
    # 示例数据
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
    
    print(f"查询: {query}")
    print(f"响应A: {response_a}")
    print(f"响应B: {response_b}")
    
    print("\n生成偏好数据...")
    
    # 生成偏好数据
    preference_data = helpsteer.trainer.generate_preference_data(
        query, context, response_a, response_b
    )
    
    # 显示结果
    print("\n=== 偏好分析结果 ===")
    print(f"更好的响应分数: {preference_data['better_score']}/10")
    print(f"较差的响应分数: {preference_data['worse_score']}/10")
    print(f"分数差异: {preference_data['score_difference']:.2f}")
    
    print("\n偏好分析:")
    print(preference_data["analysis"])
    
    # 保存偏好数据
    output_file = "helpsteer_preference_data.json"
    helpsteer.trainer.save_preference_data(preference_data, output_file)
    print(f"\n偏好数据已保存到: {output_file}")


def example_batch_processing():
    """示例：批量处理"""
    print("\n=== HelpSteer批量处理示例 ===")
    
    # 设置LLM客户端
    llm_client = setup_llm_client()
    if not llm_client:
        return
    
    # 初始化HelpSteer系统
    helpsteer = HelpSteerSystem(llm_client)
    
    # 示例数据
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
    
    print(f"开始批量评估 {len(test_cases)} 个测试用例...")
    
    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n处理测试用例 {i}...")
        
        evaluation = helpsteer.evaluator.evaluate_response(
            test_case["query"], 
            test_case["response"], 
            test_case["context"]
        )
        
        results.append({
            "case": i,
            "score": evaluation.overall_score
        })
    
    # 显示汇总结果
    print("\n=== 批量评估汇总 ===")
    total_score = sum(r["score"] for r in results)
    avg_score = total_score / len(results)
    
    print(f"平均分数: {avg_score:.2f}/10")
    print(f"最高分数: {max(r['score'] for r in results):.2f}/10")
    print(f"最低分数: {min(r['score'] for r in results):.2f}/10")
    
    print("\n详细结果:")
    for result in results:
        print(f"用例 {result['case']}: {result['score']:.2f}/10")


def main():
    """主函数"""
    print("HelpSteer系统使用示例")
    print("=" * 50)
    
    # 检查配置
    if not os.getenv("AZURE_OPENAI_API_KEY"):
        print("请先配置Azure OpenAI API密钥")
        print("可以使用环境变量或运行 config.py")
        return
    
    # 运行示例
    try:
        example_evaluation()
        example_improvement()
        example_preference_learning()
        example_batch_processing()
        
        print("\n" + "=" * 50)
        print("所有示例运行完成！")
        
    except Exception as e:
        print(f"运行示例时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 