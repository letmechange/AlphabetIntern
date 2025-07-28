class QueryUnderstanding:
    def __init__(self, llm_client):
        self.llm = llm_client

    def extract_keywords(self, query: str):
        """
        Always use English prompt for keyword extraction.
        """
        arrow = "\u2192"
        prompt = f'''
You are an academic search assistant.

Your task is to extract the core technical or research-related **keywords** from a user's natural language query, so they can be used to search academic papers.

Ignore filler words like "what is", "can you explain", etc. Only return 2 to 5 relevant **academic keywords** (technical terms, topics, models, methods, etc.) that match the user's intent.

Return the result as a **comma-separated list** of noun phrases.

Examples:
Query: "Can you explain how convolutional neural networks work in image classification?"
{arrow} Keywords: convolutional neural networks, image classification

Query: "What are the recent advancements in reinforcement learning for robotics?"
{arrow} Keywords: reinforcement learning, robotics

Query: "How does social media influence political participation?"
{arrow} Keywords: social media, political participation

Query: "Advances in CRISPR gene editing for crop improvement"
{arrow} Keywords: CRISPR, gene editing, crop improvement

Query: "Applications of graph theory in social network analysis"
{arrow} Keywords: graph theory, social network analysis

Query: "Explain the impact of climate change on marine biodiversity"
{arrow} Keywords: climate change, marine biodiversity

Query: "{query}"
{arrow} Keywords: 
'''
        response = self.llm.call_llm_api(prompt)
        return response.strip()
