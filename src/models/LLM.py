from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAI
from pydantic import BaseModel, root_validator

from src.promptimizer.apikey import OPEN_AI_API_KEY
from src.promptimizer.constants import OPUS, GPT4_NAME, CLAUDE_NAME, GEMINI_NAME, GPT4, GEMINI


class LLM(BaseModel):
    llm_name: str
    temp: float = 0.1
    langchain_model: any

    @root_validator(pre=True)  # `pre=True` means this runs before field validation
    def get_langchain_model(self):
        if self.llm_name == GPT4_NAME:
            return OpenAI(model_name=GPT4, temperature=self.temp, openai_api_key=OPEN_AI_API_KEY)
        elif self.llm_name == CLAUDE_NAME:
            return ChatAnthropic(temperature=0, anthropic_api_key=ANTHROPIC_API_KEY, model_name=OPUS)
        elif self.llm_name == GEMINI_NAME:
            return ChatGoogleGenerativeAI(model=GEMINI)
        else:
            raise ValueError("Invalid LLM name")
