from langchain_huggingface import HuggingFaceEmbeddings

class Embedding:
    def __init__(self, provider="huggingface", model_name="BAAI/bge-m3", **kwargs):
        """
        Load embedding model from the specified provider.
        
        Args:
            provider (str): One of 'huggingface', etc.
            model_name (str): Model name or ID, depending on the provider.
            **kwargs: Extra arguments for the embedding class (like api_key).
        """
        self.provider = provider.lower()
        self.model_name = model_name

        print(f"[EmbeddingClient] Loading embedding model '{model_name}' from provider: {self.provider}")
        if self.provider == "huggingface":
            if not any(sub in model_name.lower() for sub in ["sentence-transformers", "all-", "bge", "mpnet"]):
                raise ValueError(f"invalid HuggingFace model: {model_name}")
            self.model = HuggingFaceEmbeddings(model_name=model_name, **kwargs)
        
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")
    
    def get_model(self):
        return self.model