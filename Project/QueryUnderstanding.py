class QueryUnderstanding:
    def __init__(self, llm_client):
        self.llm = llm_client

    def extract_keywords(self, query: str):
        """
        Consider we are working on the topic related with acadmeic papers.
        """
        arrow = "\u2192"
        prompt = f"""
You are an academic search assistant.

Your task is to extract the core technical or research-related **keywords** from a user's natural language query, so they can be used to search academic papers.

Ignore filler words like "what is", "can you explain", etc. Only return 2 to 5 relevant **academic keywords** (technical terms, topics, models, methods, etc.) that match the user's intent.

Return teh result as a **comma-separated list** of noun phrases.

Examples:
Query: "Can you explain how convolutional neural networks work in image classification?"
{arrow} Keywords: convolutional neural networks, image classification

Query: "What are the recent advancements in reinforcement learning for robotics?"
{arrow} Keywords: reinforcemnt learning, robotics

Query" "{query}"
{arrow} Keywords: 
"""
        response = self.llm.call_llm_api(prompt)
        return response.strip()
