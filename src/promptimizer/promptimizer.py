from src.models.PromptimizerLLM import PromptimizerLLM
from src.models.Prompt import Prompt


class Promptimizer:
    def __init__(self,
                 llm: PromptimizerLLM,
                 seed_prompt: Prompt,
                 winner_count: int = 1,
                 target_of_action: str = None,
                 example_data: dict[any, any] = None):
        """
        :param llm: LLM custom pydantic model to use for prompt generation.
        :param seed_prompt: initial prompt custom pydantic model to expand upon.
        :param winner_count: number of prompts to return.
        :param target_of_action: semantic error function to minimize.
        :param example_data: dictionary of example data (data is used for prompt scoring)
        :return: named tuple of items, returning optimized prompt, score, list of improvements made from the original prompt
        """
        self.llm = llm

        self.seed_prompt = seed_prompt
        self.seed_prompt.score = self._score_prompt(seed_prompt, example_data)
        self.toa = target_of_action
        self.test_data = example_data

        self.winner_count = winner_count
        self.optimizedPrompts = None

    # exposed method to train the prompt
    def promptimize(self, expansion_factor: int = 10, steps_factor: int = 1, ):
        """
        :param expansion_factor: number of expansions to perform.
        :param steps_factor: count of evolution steps to perform -> !!this will be deprecated in favor of semantic based completion (llm decides it cannot improve the prompt further)!!
        """
        return self._promptimize(self.seed_prompt, steps_factor, expansion_factor)

    def _promptimize(self, prompt: Prompt, gen_limit: int, expansion_limit: int):
        prompt_generation_count = int(prompt.id.split("_")[0])+1

        toa = self.toa

        if prompt_generation_count == gen_limit:
            return prompt

        # STEP 1: ASSESS CURRENT PROMPT, CREATE GRADIENT
        # score the prompt
        self._score_prompt(prompt, self.test_data)
        prompt.score = self._critique_prompt(prompt)

        # STEP 2: EXPAND AND IMPROVE PROMPT BASED ON GRADIENT STEPS TIMES
        # expand the prompt based on the critique steps_limit times -> error correction step
        prompt_candidates = self._expand_prompt(prompt_generation_count, expansion_limit, monte_carlo=False)

        # STEP 3: SCORE AND SELECT BEST PROMPTS BASED ON SCORE
        candidates_scored = (self._score_prompt(candidate, self.test_data) for candidate in prompt_candidates)
        # TODO: implement sorting for Prompt objects on score
        candidates_scored = sorted(candidates_scored)[:self.winner_count]

        if candidates_scored[0] < prompt.score:
            return prompt

        # STEP 4: RECURSE ON TOP PROMPTS
        for next_gen_prompt in candidates_scored:
            self._promptimize(next_gen_prompt, gen_limit, expansion_limit)

    # given an initial prompt (seed), its target of action (error function to minimize), its semantic gradient,
    # generate expansion_factor improved prompts
    def _expand_prompt(self, prompt_candidate: Prompt, expansion_factor: int, monte_carlo: bool = False,) -> \
            list[Prompt]:
        """
        :param prompt_candidate: initial prompt to expand upon.
        :param expansion_factor: number of expansions to perform.
        :param monte_carlo: whether 50% of expansions are semantically identical, but worded differently.
        :return: list of expanded prompts.
        """

    # TODO: CRITICAL -> PASS CRITIQUE HISTORY OPTIONALLY
    # This would be akin to passing the gradient history to the LLM, which would allow it to learn from the gradient history.
    # given a prompt, its score, training data, and semantic critique
    # use LLM to score it against a target of action (error function to minimize)
    def _critique_prompt(self, prompt: Prompt, target_of_action: str = None) -> str:
        """
        :param prompt: prompt to score.
        :return: semantic error of the prompt against the target action, optionally tested against provided data.
        """
        pass

    # TODO: implement Successive Rejects
    # # sorts and returns winner_count prompts from all prompts, based on their scores
    # def _select_best_prompts(self, prompts: list[Prompt], winner_count: int) -> list[str]:
    #     """
    #     :param prompts: list of prompts to score.
    #     :param winner_count: number of prompts to return.
    #     :return: list of prompts with the best scores, up to 'winner_count'
    #     """
    #     pass

    # score a prompt against a target of action (error function to minimize) and optional test data
    def _score_prompt(self, prompt: Prompt, data: dict[any, any] = None, semantic: bool = True) -> float:
        """
        :param prompt: prompt to score.
        :param data: optional dictionary of example data (data is used for prompt scoring)
        :param semantic: whether to score the prompt against the target of action (semantic error function)
        :return: score of the prompt against the target of action (semantic error function)
        """

        pass

    #TODO:
    # -Summary:
    # consider implementation of a target_of_action optimizer, which, based on the user provided generalized action,
    # selects one of the following well-understood categories and optimizes for them (reducing the field of possibilities
    # over which the prompt has to be optimized)
    # -Categories:
    # needle_in_haystack
    # code_gen !!!
    # image_gen !!
    # code_gen_troubleshoot
    # text_gen_qa
    # text_gen_summarization
    # text_gen_translation
    # text_gen_writing
    # text_gen_chat
    # categorization
    # fact_checking
    # sentiment_analysis

    # TODO: Implement 'is_monte_carlo: bool' and its logic
