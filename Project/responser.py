from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

def build_answer(llm_client, query, top_docs):
    context = "\n\n".join([doc.page_content for doc in top_docs])
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        
You are an expert academic searcher assistant. Answer the question using ONLY the context below.
For each sentence in your answer, add a reference in square brackets (e.g., [1], [2]) corresponding to the supporting document from the context.
If a sentence is supported by multiple documents, list all relevant references.

你是一名学术搜索助手。请仅根据下方提供的文献内容回答问题。
每一句话后请加上对应文献的引用编号（如 [1], [2]），编号对应 context 中的文献。
如一句话有多个支持文献，可列出所有编号。

At the end of your answer, please provide a reference list showing the document titles or sources that correspond to each reference number used in your answer.

在回答的最后，请提供引用文献列表，显示每个引用编号对应的文献标题或来源。

Context:
{context}

Question:
{question}
""")
    chain = LLMChain(llm=llm_client.llm, prompt=prompt_template)
    return chain.run(context=context, question=query)