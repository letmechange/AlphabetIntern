from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def build_answer(llm_client, query, top_docs):
    context = "\n\n".join([doc.page_content for doc in top_docs])
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are an expert assistant. Answer the question using the context below.

Context:
{context}

Question:
{question}
""")
    chain = LLMChain(llm=llm_client.llm, prompt=prompt_template)
    return chain.run(context=context, question=query)