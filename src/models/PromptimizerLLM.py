from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAI
from pydantic import BaseModel, root_validator
from rich import console

from src.promptimizer.apikey import OPEN_AI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY
from src.promptimizer.constants import OPUS, GPT4_NAME, CLAUDE_NAME, GEMINI_NAME, GPT4, GEMINI


class PromptimizerLLM(BaseModel):
    llm_name: str
    temp: float = 0.1
    langchain_model: Any = None

    @root_validator(pre=True, allow_reuse=True)  # `pre=True` means this runs before field validation
    def init_langchain_model(cls, values):
        llm_name = values.get('llm_name')
        temp = values.get('temp', 0.1)  # Default to 0.1 if not provided

        if llm_name == GPT4_NAME:
            values['langchain_model'] = OpenAI(model_name=GPT4,
                                               temperature=temp,
                                               openai_api_key=OPEN_AI_API_KEY)
        elif llm_name == CLAUDE_NAME:
            values['langchain_model'] = ChatAnthropic(
                                                model_name=OPUS,
                                                temperature=temp,
                                                anthropic_api_key=ANTHROPIC_API_KEY)
        elif llm_name == GEMINI_NAME:
            values['langchain_model'] = ChatGoogleGenerativeAI(
                                                model=GEMINI,
                                                google_api_key=GOOGLE_API_KEY)
        else:
            raise ValueError("Invalid LLM name")

        return values
