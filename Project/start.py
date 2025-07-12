#!/usr/bin/env python3
"""
RAG系统启动脚本
提供简单的命令行界面来启动和使用RAG系统
"""

import os
import sys
from pathlib import Path

def check_dependencies():
    """检查必要的依赖是否已安装"""
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
        print("❌ 缺少以下依赖包:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n请运行以下命令安装:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ 所有依赖包已安装")
    return True

def check_config():
    """检查配置文件"""
    config_file = Path("config.py")
    if not config_file.exists():
        print("❌ 配置文件 config.py 不存在")
        return False
    
    # 检查环境变量
    required_env_vars = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"]
    missing_vars = []
    
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("⚠️  以下环境变量未设置:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\n请在 config.py 中设置这些变量")
        return False
    
    print("✅ 配置检查通过")
    return True

def show_menu():
    """显示主菜单"""
    print("\n" + "="*50)
    print("🚀 RAG系统 - 检索增强生成")
    print("="*50)
    print("请选择运行模式:")
    print("1. 交互式查询 (推荐)")
    print("2. 示例查询")
    print("3. 批量查询")
    print("4. 系统信息")
    print("5. 更新文档库")
    print("0. 退出")
    print("="*50)

def run_interactive():
    """运行交互式查询"""
    print("\n🎯 启动交互式查询模式...")
    try:
        from example_usage import interactive_mode
        interactive_mode()
    except Exception as e:
        print(f"❌ 启动交互式模式失败: {str(e)}")

def run_example():
    """运行示例查询"""
    print("\n📝 运行示例查询...")
    try:
        from example_usage import example_usage
        example_usage()
    except Exception as e:
        print(f"❌ 运行示例查询失败: {str(e)}")

def run_batch():
    """运行批量查询"""
    print("\n📋 运行批量查询...")
    try:
        from example_usage import batch_query_mode
        batch_query_mode()
    except Exception as e:
        print(f"❌ 运行批量查询失败: {str(e)}")

def show_system_info():
    """显示系统信息"""
    print("\n📊 系统信息:")
    try:
        from main import RAGSystem
        rag = RAGSystem()
        info = rag.get_system_info()
        
        for key, value in info.items():
            if key == "documents":
                print(f"  {key}: {len(value)} 个文档")
                for doc in value[:3]:  # 只显示前3个
                    print(f"    - {Path(doc).name}")
                if len(value) > 3:
                    print(f"    ... 还有 {len(value) - 3} 个文档")
            else:
                print(f"  {key}: {value}")
                
    except Exception as e:
        print(f"❌ 获取系统信息失败: {str(e)}")

def update_documents():
    """更新文档库"""
    print("\n📚 更新文档库...")
    try:
        from main import RAGSystem
        rag = RAGSystem()
        
        sample_folder = "./Sample"
        if os.path.exists(sample_folder):
            rag.update_documents(sample_folder)
            print("✅ 文档库更新完成!")
        else:
            print(f"⚠️  Sample文件夹不存在: {sample_folder}")
            print("请将文档放入 Sample 文件夹中")
            
    except Exception as e:
        print(f"❌ 更新文档库失败: {str(e)}")

def main():
    """主函数"""
    print("🔧 检查系统环境...")
    
    # 检查依赖
    if not check_dependencies():
        return
    
    # 检查配置
    if not check_config():
        print("\n⚠️  请先配置API密钥后再运行系统")
        return
    
    print("✅ 系统环境检查完成!")
    
    # 主循环
    while True:
        show_menu()
        
        try:
            choice = input("\n请输入选择 (0-5): ").strip()
            
            if choice == "0":
                print("👋 再见!")
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
                print("❌ 无效选择，请重新输入")
                
        except KeyboardInterrupt:
            print("\n👋 再见!")
            break
        except Exception as e:
            print(f"❌ 发生错误: {str(e)}")

if __name__ == "__main__":
    main() 