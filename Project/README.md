# RAG系统 - 检索增强生成

这是一个完整的RAG（Retrieval-Augmented Generation）系统，用于基于文档的智能问答。

## 🏗️ 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   文档加载      │    │   向量化存储    │    │   查询处理      │
│  data_loader    │───▶│   Embedding     │───▶│ QueryUnderstanding│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   文档管理      │    │   向量数据库    │    │   重排序        │
│data_store_loader│    │     Chroma      │    │    Reranker     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   配置管理      │    │   LLM客户端     │    │   回答生成      │
│     config      │    │   LLMClient     │    │   responser     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📁 项目结构

```
Project/
├── main.py                 # 主流程文件
├── example_usage.py        # 使用示例
├── config.py              # 配置管理
├── data_loader.py         # 文档加载器
├── data_store_loader.py   # 文档存储管理
├── Embedding.py           # 嵌入模型
├── LLMClient.py           # LLM客户端
├── QueryUnderstanding.py  # 查询理解
├── Reranker.py            # 重排序
├── responser.py           # 回答生成
├── Sample/                # 示例文档
│   ├── Sample.pdf
│   └── Pronina et al 2022-LLD-FINAL ARTICLE.txt
├── vector_db/             # 向量数据库
└── manifest.json          # 文档清单
```

## 🚀 快速开始

### 1. 环境准备

确保安装了必要的依赖：

```bash
pip install langchain langchain-community chromadb sentence-transformers
pip install langchain-openai langchain-deepseek
```

### 2. 配置API密钥

在 `config.py` 中配置你的API密钥：

```python
# Azure OpenAI
os.environ["AZURE_OPENAI_API_KEY"] = "your_api_key"
os.environ["AZURE_OPENAI_ENDPOINT"] = "your_endpoint"

# DeepSeek
os.environ["DEEPSEEK_API_KEY"] = "your_api_key"
```

### 3. 运行系统

#### 方式一：直接运行主程序
```bash
python main.py
```

#### 方式二：使用示例程序
```bash
# 运行示例查询
python example_usage.py example

# 交互式查询
python example_usage.py interactive

# 批量查询
python example_usage.py batch
```

## 🔧 核心组件

### 1. 文档处理 (data_store_loader.py)
- **功能**: 管理文档的加载、更新和存储
- **特性**: 
  - 自动检测文档变化
  - 增量更新向量数据库
  - 支持PDF和TXT格式

### 2. 嵌入模型 (Embedding.py)
- **功能**: 将文本转换为向量表示
- **模型**: all-MiniLM-L6-v2 (默认)
- **特性**: 支持多种HuggingFace模型

### 3. LLM客户端 (LLMClient.py)
- **功能**: 连接不同的LLM服务
- **支持**: Azure OpenAI, OpenAI, DeepSeek
- **特性**: 统一的API接口

### 4. 查询理解 (QueryUnderstanding.py)
- **功能**: 从用户查询中提取关键词
- **特性**: 针对学术论文优化
- **输出**: 逗号分隔的关键词列表

### 5. 重排序 (Reranker.py)
- **功能**: 对检索结果进行重新排序
- **方法**: 基于LLM的相关性评分
- **输出**: 按相关性排序的文档列表

### 6. 回答生成 (responser.py)
- **功能**: 基于检索到的文档生成回答
- **方法**: 使用LLM进行上下文生成
- **特性**: 支持多文档融合

## 📊 工作流程

### 1. 文档处理阶段
```
文档文件夹 → 文档加载 → 文本分割 → 向量化 → 存储到Chroma
```

### 2. 查询处理阶段
```
用户查询 → 查询理解 → 关键词提取 → 向量检索 → 重排序 → 回答生成
```

### 3. 详细流程
1. **文档加载**: 扫描指定文件夹中的PDF和TXT文件
2. **文本分割**: 使用RecursiveCharacterTextSplitter进行分块
3. **向量化**: 使用sentence-transformers生成嵌入向量
4. **存储**: 将向量存储到Chroma向量数据库
5. **查询理解**: 使用LLM提取查询中的关键词
6. **检索**: 基于关键词进行相似性搜索
7. **重排序**: 使用LLM对检索结果进行相关性评分
8. **生成**: 基于重排序后的文档生成最终回答

## 🎯 使用示例

### 基本使用
```python
from main import RAGSystem

# 初始化系统
rag = RAGSystem(
    embedding_model_name="all-MiniLM-L6-v2",
    llm_provider="azure",
    llm_model="gpt-4"
)

# 更新文档库
rag.update_documents("./Sample")

# 执行查询
result = rag.query("什么是机器学习？")
print(result['answer'])
```

### 批量查询
```python
queries = [
    "深度学习的基本原理",
    "神经网络的结构",
    "强化学习的应用"
]

for query in queries:
    result = rag.query(query)
    print(f"问题: {query}")
    print(f"回答: {result['answer']}")
    print("-" * 50)
```

## ⚙️ 配置选项

### 嵌入模型配置
```python
# 使用不同的嵌入模型
rag = RAGSystem(
    embedding_model_name="all-mpnet-base-v2",  # 更高质量的模型
    llm_provider="azure"
)
```

### LLM提供商配置
```python
# Azure OpenAI
rag = RAGSystem(
    llm_provider="azure",
    llm_model="gpt-4"
)

# DeepSeek
rag = RAGSystem(
    llm_provider="deepseek",
    llm_model="deepseek-chat"
)
```

### 向量数据库配置
```python
# 自定义向量数据库路径
rag = RAGSystem(
    vector_db_path="./custom_vector_db",
    manifest_path="./custom_manifest.json"
)
```

## 📈 性能优化

### 1. 文档分块优化
- **chunk_size**: 控制文档块大小 (默认: 1000)
- **chunk_overlap**: 控制块间重叠 (默认: 50)

### 2. 检索优化
- **top_k**: 控制检索文档数量 (默认: 10)
- **重排序top_k**: 控制重排序后保留的文档数量 (默认: 5)

### 3. 模型选择
- **嵌入模型**: 平衡质量和速度
- **LLM模型**: 根据需求选择不同能力的模型

## 🔍 故障排除

### 常见问题

1. **API密钥错误**
   ```
   解决方案: 检查config.py中的API密钥配置
   ```

2. **文档加载失败**
   ```
   解决方案: 确保文档格式正确，路径存在
   ```

3. **向量数据库错误**
   ```
   解决方案: 删除vector_db文件夹重新初始化
   ```

4. **内存不足**
   ```
   解决方案: 减少chunk_size或使用更小的嵌入模型
   ```

### 调试模式
```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 扩展功能

### 1. 添加新的文档格式
在 `data_store_loader.py` 中添加新的加载器：

```python
# 添加新的文件类型支持
elif path_str.endswith(".docx"):
    from langchain.document_loaders import Docx2txtLoader
    loader = Docx2txtLoader(path_str)
```

### 2. 自定义重排序策略
在 `Reranker.py` 中修改评分方法：

```python
def custom_scoring(self, query, doc):
    # 实现自定义评分逻辑
    pass
```

### 3. 添加缓存机制
```python
# 添加结果缓存
import functools

@functools.lru_cache(maxsize=1000)
def cached_query(self, query):
    return self.query(query)
```

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 邮箱: [your-email@example.com]
- GitHub: [your-github-profile]

---

**注意**: 请确保在使用前正确配置API密钥，并遵守相关服务的使用条款。 