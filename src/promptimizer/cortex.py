from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate

from src.models.PromptimizerLLM import PromptimizerLLM
from src.models.Prompt import Prompt
from enum import Enum, auto


"""
 _______  _______  _______ _________ _______          
(  ____ \(  ___  )(  ____ )\__   __/(  ____ \|\     /|
| (    \/| (   ) || (    )|   ) (   | (    \/( \   / )
| |      | |   | || (____)|   | |   | (__     \ (_) / 
| |      | |   | ||     __)   | |   |  __)     ) _ (  
| |      | |   | || (\ (      | |   | (       / ( ) \ 
| (____/\| (___) || ) \ \__   | |   | (____/\( /   \ )
(_______/(_______)|/   \__/   )_(   (_______/|/     \|
"""


class Cortex:
    """
    Multi-tier system that aims to resolve the issues with llms inability to be creative or plan - pseudo cognitive.

    Attributes:
        representations (dict[str, str]): A dictionary mapping Small compressed strings\binaries\whatever, cortex can use as RAM essentially
        tool_list (dict[str, any]): The synergy can utilize these tools for output generation
        llm_options (dict[str, str]): A dictionary of options/configurations for the llms Cortex can poll
        llms (list[PromptimizerLLM]): List of custom pydantic LLM models to use for prompt generation.
    """

    representations: dict[str, str] = None
    tool_list: dict[str, any] = None
    llm_options: dict[str, str] = None
    llms: list[PromptimizerLLM] = None

    def __init__(self, cortex_input: any):
        """
        Initializes a new instance of the Cortex.

        Args:
            cortex_input (any): The input to the Cortex system.

        Attributes:
            - target_of_action (str): The general action this prompt seeks to achieve
            - cognitive_steps (dict[str, Prompt], optional): prompts for different cognitive steps
            - cognitive_outputs (dict[str, list[str]], optional): A dictionary mapping. Defaults to None.
                - immediate_thoughts: The small models response to the error generation prompt
                - impulsive_thoughts: Adds a little random spice to the mix, just like a real brain ;)
                - long_term_reasoning: Thoughts generated for high complexity inputs by deep models

            - outputs (dict[str, list[str]], optional): A dictionary storing the outputs. Defaults to None.
                - list_of_outputs: the model, based on complexity and thoughts, generates possible outputs
                - final_output: if enabled, picks the output Cortex considers best

            - complexity (float, optional): Complexity of the input as it relates to probable output. Defaults to 0.0.
            - tool_use_likelihood (float, optional):  how encouraged the model will be to use tools. Defaults to 0.0.
            - plan (str, optional): Im thinking this will be a 'running thought' that the model will use to keep a
              modifiable plan. Defaults to None.
        """

        self.cortex_input = cortex_input

        self.target_of_action = None
        self.cognitive_steps: dict[str, Prompt] = None
        self.cognitive_outputs: dict[str, list[str]] = None
        self.outputs: dict[str, list[str]] = None

        self.complexity: float = 0.0
        self.tool_use_likelihood: float = 0.0
        self.plan: str = None

    def take_in_input(self):
        pass


    #TODO:
    # -refactor docstring to use Spynx notation
    """
        :param <name>: Describes a parameter by name.
        :type <name>: Specifies the type of the parameter.
        :returns: Provides a brief description of what the method returns.
        :rtype: Specifies the return type of the method.
        :raises <exception>: Describes any exceptions that the method can raise.
    """