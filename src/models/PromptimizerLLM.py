from __future__ import annotations # this is important to have at the top
import logging
from typing import Optional
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAI
from pydantic import BaseModel, root_validator

from src.promptimizer.apikey import OPEN_AI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY
from src.promptimizer.constants import OPUS, GPT4_NAME, CLAUDE_NAME, GEMINI_NAME, GPT4, GEMINI


class PromptimizerLLM(BaseModel):
    llm_name: str
    temp: float = 0.1
    api_keys: Optional[dict[str, str]] = None
    langchain_model: BaseChatModel = None

    @root_validator(pre=True, allow_reuse=True)  # `pre=True` means this runs before field validation
    def init_langchain_model(cls, values):
        llm_name = values.get('llm_name')
        temp = values.get('temp', 0.1)  # Default to 0.1 if not provided
        api_keys = values.get('api_keys')
        if llm_name == GPT4_NAME:
            if "chat_gpt" in api_keys.keys() and api_keys["chat_gpt"]:
                values['langchain_model'] = OpenAI(model_name=GPT4,
                                                   temperature=temp,
                                                   openai_api_key=api_keys["chat_gpt"])
            elif OPEN_AI_API_KEY:
                values['langchain_model'] = OpenAI(model_name=GPT4,
                                                   temperature=temp,
                                                   openai_api_key=OPEN_AI_API_KEY)
            else:
                logging.warning("OpenAI API key is missing or invalid.")

        elif llm_name == CLAUDE_NAME:
            if "anthropic" in api_keys.keys() and api_keys["anthropic"]:
                values['langchain_model'] = ChatAnthropic(
                                                    model_name=OPUS,
                                                    temperature=temp,
                                                    anthropic_api_key=api_keys["anthropic"])
            elif ANTHROPIC_API_KEY:
                values['langchain_model'] = ChatAnthropic(
                                                    model_name=OPUS,
                                                    temperature=temp,
                                                    anthropic_api_key=ANTHROPIC_API_KEY)
            else:
                logging.warning("Anthropic API key is missing or invalid.")

        elif llm_name == GEMINI_NAME:
            if "gemini" in api_keys.keys() and api_keys["gemini"]:
                values['langchain_model'] = ChatGoogleGenerativeAI(
                                                    model=GEMINI,
                                                    google_api_key=api_keys["gemini"])
            elif GOOGLE_API_KEY:
                values['langchain_model'] = ChatGoogleGenerativeAI(
                                                    model=GEMINI,
                                                    google_api_key=GOOGLE_API_KEY)
            else:
                logging.warning("Google API key is missing or invalid.")
        else:
            raise ValueError("Invalid LLM name")
        return values


#Ive no idea, but fixed pydantic validation errors *shrug*
#https://stackoverflow.com/questions/63420889/fastapi-pydantic-circular-references-in-separate-files/70384637#70384637
from langchain_core.language_models import BaseChatModel

PromptimizerLLM.update_forward_refs()
