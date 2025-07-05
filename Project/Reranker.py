class Reranker:
    def __init__(self, llm_client, top_k: int = 5):
        self.llm = llm_client
        self.top_k = top_k
    
    def rerank(self, query: str, docs: list[str]) -> list[tuple[str, float]]:
        scored_docs = []

        for i, doc in enumerate(docs):
            prompt = self.build_prompt(query, doc)
            score_str = self.llm.call_llm_api(prompt).strip()
            try:
                score = float(score_str)
            except:
                score = 0.0
            scored_docs.append((doc, score))
            
        
        # ranking descending way
        sorted_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
        return sorted_docs[:self.top_k]
    
    def build_prompt(self, query:str, doc: str) -> str:
        return f"""
You are an academic assistant helping to rerank retrieved papers.

Given a user query and a paragraphs from a document, rate how relevent the document is to the query on a scale from 1 to 
{self.top_k}, rate how relevant the document is to the query on a scale from 1 to {self.top_k}.
1 = irrelevant, {self.top_k} = extremely relevant.

Respond only with a number.

Query: "{query}"

Document:
\"\"\"
{doc}
\"\"\"

Score (1-{self.top_k}):
"""