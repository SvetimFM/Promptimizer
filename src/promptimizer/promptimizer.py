from src.models.PromptimizerLLM import PromptimizerLLM
from src.models.Prompt import Prompt


# given an initial prompt (seed), and a target of action (error function to minimize), generate a universe of improved
# prompts
def _expand_prompt(prompt_candidate: str, target_of_action: str, expansion_factor: int, monte_carlo: bool = False) -> \
        list[str]:
    """
    :param prompt_candidate: initial prompt to expand upon.
    :param target_of_action: semantic error function to minimize.
    :param expansion_factor: number of expansions to perform.
    :param monte_carlo: whether 50% of expansions are semantically identical, but worded differently.
    :return: list of expanded prompts.
    """


# given a prompt, use LLM to score it against a target of action (error function to minimize)
def _critique_prompt(prompt, target_of_action: str, test_data: dict[any, any] = None) -> str:
    """
    :param prompt: prompt to score.
    :param target_of_action: semantic error function to minimize.
    :param test_data: optional dictionary of example data (data is used for prompt scoring as test data)
    :return: semantic error of the prompt against the target action, optionally tested against provided data.
    """
    pass


def _error_correct_prompt(prompt: str, semantic_error: str, target_of_action: str) -> str:
    """
    :param prompt: prompt to correct.
    :param semantic_error: semantic error of the prompt.
    :param target_of_action: semantic error function to minimize.
    :return: prompt with the semantic error corrected.
    """
    pass


# TODO: implement Successive Rejects
# scores a universe of prompts against a target of action (error function to minimize)
def _select_best_prompts(prompts: list[str], target_of_action: str, winner_count) -> list[str]:
    """
    :param prompts: list of prompts to score.
    :param target_of_action: semantic error function to minimize.
    :param winner_count: number of prompts to return.
    :return: list of prompts with the best scores, up to 'winner_count'
    """
    pass


# TODO: Implement training data and subsequent testing
def run_optimizer(llm: PromptimizerLLM, prompt: Prompt, expansion_factor: int = 10, steps_factor: int = 1, winner_count: int = 1, target_of_action: str = None, example_data: dict[any, any] = None) -> list[str]:
    """
    :param llm: LLM to use for prompt generation.
    :param prompt: initial prompt to expand upon.
    :param expansion_factor: number of expansions to perform.
    :param steps_factor: count of evolution steps to perform -> !!this will be deprecated in favor of semantic based completion (llm decides it cannot improve the prompt further)!!
    :param winner_count: number of prompts to return.
    :param target_of_action: semantic error function to minimize.
    :param example_data: dictionary of example data (data is used for prompt scoring)
    :return: named tuple of items, returning optimized prompt, score, list of improvements made from the original prompt
    """
    return [llm, prompt]



#TODO:
# -Summary:
# consider implementation of a target_of_action optimizer, which, based on the user provided generalized action,
# selects one of the following well-understood categories and optimizes for them (reducing the field of possibilities
# over which the prompt has to be optimized)
# -Categories:
# needle_in_haystack
# code_gen
# code_gen_troubleshoot
# text_gen_qa
# text_gen_summarization
# text_gen_translation
# text_gen_writing
# text_gen_chat
# categorization
# fact_checking
# sentiment_analysis

# TODO: Implement optimization for prompts for image generation and code output explicitly

# TODO: Implement 'is_monte_carlo: bool' and its logic
