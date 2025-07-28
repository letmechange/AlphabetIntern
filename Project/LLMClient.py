from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_openai import AzureChatOpenAI


class LLM_Client:
    def __init__(self, api_key: str, model_name: str = "gpt-4", provider: str = "openai", **kwargs):


        self.provider = provider.lower()
        self.model_name = model_name

        if self.provider == "openai":
            self.llm = ChatOpenAI(openai_api_key=api_key, model_name=model_name)
        
        elif self.provider == "deepseek":
            self.llm = ChatDeepSeek(
                api_key=api_key, 
                model=model_name,
                max_tokens=kwargs.get('max_tokens')
                )
        
        elif self.provider == "azure":
            self.llm = AzureChatOpenAI(
                api_key=api_key,
                azure_endpoint=kwargs.get("endpoint"),
                deployment_name=kwargs.get("deployment_name"),
                api_version=kwargs.get("api_version")
            )
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
        

    def call_llm_api(self, prompt):
        return self.llm.predict(prompt)