# 🔐 安全配置指南

## ⚠️ 重要安全提醒

**永远不要在代码中硬编码API密钥！** 这样做会带来严重的安全风险。

## 🚨 发现的问题

在原始的 `config.py` 文件中发现了以下安全问题：

### 1. API密钥硬编码
```python
# ❌ 危险做法 - 不要这样做！
os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass(
    " qhfhcZwRWTGDykfNXzUoINKnPxI8lFiQzBne4vJcbLzvxupagXFHJQQJ99ALACHYHv6XJ3w3AAAAACOGS5fO"
)
```

**问题**:
- API密钥直接写在代码中
- 可能被意外提交到版本控制系统
- 任何人都可以看到你的密钥

### 2. 逻辑错误
```python
# ❌ 错误的逻辑
if "AZURE_OPENAI_API_KEY" not in os.environ:
    # 这里应该提示用户输入，而不是使用硬编码密钥
```

### 3. 不一致的处理方式
- Azure OpenAI: 使用硬编码密钥
- DeepSeek: 使用getpass提示用户输入

## ✅ 修复后的安全做法

### 1. 使用环境变量 (推荐)
```bash
# 在系统环境变量中设置
export AZURE_OPENAI_API_KEY="your_actual_api_key"
export AZURE_OPENAI_ENDPOINT="your_endpoint"
export DEEPSEEK_API_KEY="your_deepseek_key"
```

### 2. 使用.env文件
```bash
# 创建.env文件 (不要提交到git)
echo "AZURE_OPENAI_API_KEY=your_actual_api_key" > .env
echo "AZURE_OPENAI_ENDPOINT=your_endpoint" >> .env
echo "DEEPSEEK_API_KEY=your_deepseek_key" >> .env
```

### 3. 交互式输入
```python
# ✅ 安全做法
api_key = getpass.getpass("请输入您的API密钥: ")
if validate_api_key(api_key):
    os.environ["AZURE_OPENAI_API_KEY"] = api_key
```

## 🔧 配置方法

### 方法1: 环境变量 (最安全)
```bash
# Linux/Mac
export AZURE_OPENAI_API_KEY="sk-your-actual-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/..."

# Windows
set AZURE_OPENAI_API_KEY=sk-your-actual-key
set AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/...
```

### 方法2: .env文件
```bash
# 创建.env文件
cat > .env << EOF
AZURE_OPENAI_API_KEY=sk-your-actual-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/...
DEEPSEEK_API_KEY=your-deepseek-key
EOF

# 安装python-dotenv
pip install python-dotenv

# 在代码中加载
from dotenv import load_dotenv
load_dotenv()
```

### 方法3: 配置文件
```json
// config.json (不要提交到git)
{
    "AZURE_OPENAI_API_KEY": "sk-your-actual-key",
    "AZURE_OPENAI_ENDPOINT": "https://your-resource.openai.azure.com/...",
    "DEEPSEEK_API_KEY": "your-deepseek-key"
}
```

## 🛡️ 安全最佳实践

### 1. 永远不要提交密钥到版本控制
```bash
# 在.gitignore中添加
echo ".env" >> .gitignore
echo "config.json" >> .gitignore
echo "*.key" >> .gitignore
```

### 2. 使用密钥验证
```python
def validate_api_key(api_key: str, provider: str) -> bool:
    """验证API密钥格式"""
    if not api_key or api_key.strip() == "":
        return False
    
    if provider.lower() == "azure":
        return api_key.startswith("sk-") or len(api_key) > 20
    elif provider.lower() == "deepseek":
        return len(api_key) > 10
    
    return len(api_key) > 10
```

### 3. 错误处理
```python
try:
    api_key = get_api_key("Azure OpenAI", "AZURE_OPENAI_API_KEY")
except SystemExit:
    print("❌ 配置失败，程序退出")
    exit(1)
```

### 4. 配置验证
```python
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
        print("⚠️  缺少以下配置:")
        for var in missing_vars:
            print(f"  - {var}")
        return False
    
    return True
```

## 🔍 检查清单

在部署前，请确保：

- [ ] 没有API密钥硬编码在代码中
- [ ] .env文件已添加到.gitignore
- [ ] config.json文件已添加到.gitignore
- [ ] 环境变量已正确设置
- [ ] API密钥格式验证通过
- [ ] 错误处理机制完善
- [ ] 配置验证功能正常

## 🚨 紧急处理

如果发现API密钥泄露：

1. **立即撤销密钥**: 在Azure门户中重新生成API密钥
2. **检查代码历史**: 确保没有密钥被提交到版本控制
3. **更新所有环境**: 更新所有使用该密钥的环境
4. **监控使用情况**: 检查是否有异常使用

## 📞 获取API密钥

### Azure OpenAI
1. 访问 [Azure Portal](https://portal.azure.com/)
2. 找到你的Azure OpenAI资源
3. 在"密钥和终结点"中获取API密钥和终结点

### DeepSeek
1. 访问 [DeepSeek Platform](https://platform.deepseek.com/)
2. 注册并登录账户
3. 在API密钥管理页面获取密钥

## 📚 相关资源

- [Azure OpenAI 文档](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [DeepSeek API 文档](https://platform.deepseek.com/docs)
- [Python 环境变量管理](https://docs.python.org/3/library/os.html#os.environ)
- [Git 安全最佳实践](https://git-scm.com/book/en/v2/Git-Tools-Credential-Storage)

---

**记住**: 安全是每个人的责任。请始终遵循安全最佳实践来保护你的API密钥！ 