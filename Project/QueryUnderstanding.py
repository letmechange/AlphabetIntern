"""
Query Understanding Module

This module provides functionality for understanding and processing user queries,
including determining if RAG is needed and extracting relevant keywords.
"""

class QueryUnderstanding:
    """
    A class for understanding and processing user queries.
    
    This class provides methods to:
    1. Determine if a query needs RAG (Retrieval Augmented Generation)
    2. Extract relevant keywords from queries
    
    Attributes:
        llm: Language model client for query processing
    """
    
    def __init__(self, llm_client):
        """
        Initialize QueryUnderstanding with a language model client.
        
        Args:
            llm_client: Language model client for processing queries
        """
        self.llm = llm_client

    def need_rag(self, query: str, conversation_history: list = None) -> bool:
        """
        Determine if a query needs RAG system to answer.
        
        This method analyzes the query and conversation history to decide whether it requires 
        searching through documents (using RAG) or can be answered directly by the language model.
        
        Args:
            query: User's query string
            conversation_history: List of previous conversation turns, each being a dict with 
                                'role' (user/assistant) and 'content' keys
            
        Returns:
            bool: True if the query needs RAG, False if it can be answered directly
        """
        # Format conversation history if provided
        history_text = ""
        if conversation_history:
            history_text = "Previous conversation:\n"
            for turn in conversation_history:
                role = turn['role']
                content = turn['content']
                history_text += f"{role}: {content}\n"
            history_text += "\n"
        
        prompt = f'''
You are a query understanding assistant. Your task is to determine if a query needs to search through documents (using RAG) to be answered properly.

{history_text}Return ONLY "true" or "false":
- Return "true" if the query:
  * Asks about specific facts, data, or information that needs to be looked up
  * Requires domain knowledge or technical details
  * Asks about document content or specific topics
  * Needs concrete examples or references
  * References information or topics from previous conversation that need document lookup
  
- Return "false" if the query:
  * Can be answered with common knowledge
  * Is a general question about concepts
  * Asks for simple explanations
  * Is a greeting or casual conversation
  * Asks about system functionality or commands
  * Is a request for code generation
  * Is a math calculation
  * Only references information already provided in the conversation history

Examples:
Q: "What is machine learning?"
A: false

Q: "What were the key findings in the paper about climate change impact?"
A: true

Q: "Hello, how are you?"
A: false

Q: "What is the recommended approach for handling missing data in the dataset?"
A: true

Q: "Can you write a Python function to sort a list?"
A: false

Q: "What are the main challenges discussed in the research about quantum computing?"
A: true

User Query: "{query}"
Answer: '''
        
        response = self.llm.call_llm_api(prompt).strip().lower()
        return response == "true"

    def extract_keywords(self, query: str, conversation_history: list = None) -> str:
        """
        Extract relevant keywords from a user query and conversation history.
        
        This method processes the query and conversation context to identify and extract 
        key technical or research-related terms that can be used for document search.
        
        Args:
            query: User's query string
            conversation_history: List of previous conversation turns, each being a dict with 
                                'role' (user/assistant) and 'content' keys
            
        Returns:
            str: Comma-separated list of extracted keywords
        """
        # Format conversation history if provided
        history_text = ""
        if conversation_history:
            history_text = "Previous conversation:\n"
            for turn in conversation_history:
                role = turn['role']
                content = turn['content']
                history_text += f"{role}: {content}\n"
            history_text += "\n"

        arrow = "\u2192"
        prompt = f'''
You are an academic search assistant.

{history_text}Your task is to extract the core technical or research-related **keywords** from the user's query and conversation context, so they can be used to search academic papers.

Consider both the current query and any relevant context from the conversation history. Ignore filler words and focus on technical terms, topics, models, methods, etc. that are most relevant to the current information need.

Return 2 to 5 most relevant **academic keywords** as a **comma-separated list** of noun phrases.

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