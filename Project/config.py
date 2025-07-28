import getpass
import os
import re

def validate_api_key(api_key: str, provider: str) -> bool:
    """
    验证API密钥格式
    
    Args:
        api_key: API密钥
        provider: 提供商名称
        
    Returns:
        bool: 密钥格式是否有效
    """
    if not api_key or api_key.strip() == "":
        return False
    
    # 基本的格式验证
    if provider.lower() == "azure":
        # Azure OpenAI密钥通常是sk-开头的
        return api_key.startswith("sk-") or len(api_key) > 20
    elif provider.lower() == "deepseek":
        # DeepSeek密钥格式验证
        return len(api_key) > 10
    else:
        # 通用验证
        return len(api_key) > 10

def get_api_key(provider: str, env_var: str) -> str:
    """
    安全地获取API密钥
    
    Args:
        provider: 提供商名称
        env_var: 环境变量名
        
    Returns:
        str: API密钥
    """
    # 首先尝试从环境变量获取
    api_key = os.getenv(env_var)
    
    if api_key and validate_api_key(api_key, provider):
        return api_key
    
    # 如果环境变量中没有或无效，提示用户输入
    print(f"\n 需要配置 {provider} API密钥")
    print("请从以下位置获取API密钥:")
    
    if provider.lower() == "azure":
        print("  - Azure OpenAI: https://portal.azure.com/")
        print("  - 或使用环境变量: AZURE_OPENAI_API_KEY")
    elif provider.lower() == "deepseek":
        print("  - DeepSeek: https://platform.deepseek.com/")
        print("  - 或使用环境变量: DEEPSEEK_API_KEY")
    
    while True:
        try:
            api_key = getpass.getpass(f"请输入您的 {provider} API密钥: ").strip()
            
            if validate_api_key(api_key, provider):
                # 设置环境变量
                os.environ[env_var] = api_key
                print(f"{provider} API密钥已配置")
                return api_key
            else:
                print(f"API密钥格式无效，请重新输入")
                
        except KeyboardInterrupt:
            print("\n用户取消操作")
            raise SystemExit(1)
        except Exception as e:
            print(f"输入错误: {str(e)}")

def setup_azure_openai():
    """配置Azure OpenAI"""
    api_key = get_api_key("Azure OpenAI", "AZURE_OPENAI_API_KEY")
    
    # 设置Azure OpenAI端点
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if not endpoint:
        endpoint = "https://sweet-m55d9k6j-eastus2.openai.azure.com/openai/deployments/yu-gpt-4o/chat/completions?api-version=2025-01-01-preview"
        os.environ["AZURE_OPENAI_ENDPOINT"] = endpoint
        print("使用默认Azure OpenAI端点")
    
    return api_key, endpoint

def setup_deepseek():
    """配置DeepSeek"""
    api_key = get_api_key("DeepSeek", "DEEPSEEK_API_KEY")
    return api_key

def validate_config():
    """验证配置是否完整"""
    required_vars = {
        "AZURE_OPENAI_API_KEY": "Azure OpenAI API密钥",
        "AZURE_OPENAI_ENDPOINT": "Azure OpenAI端点",
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"{description} ({var})")
    
    if missing_vars:
        print("  缺少以下配置:")
        for var in missing_vars:
            print(f"  - {var}")
        return False
    
    return True

# 配置Azure OpenAI
try:
    azure_api_key, azure_endpoint = setup_azure_openai()
except SystemExit:
    print("配置失败，程序退出")
    exit(1)

# 配置DeepSeek (可选)
try:
    deepseek_api_key = setup_deepseek()
except SystemExit:
    print(" DeepSeek配置跳过，将仅使用Azure OpenAI")

# 验证配置
if not validate_config():
    print("配置验证失败")
    exit(1)

print(" 配置完成")