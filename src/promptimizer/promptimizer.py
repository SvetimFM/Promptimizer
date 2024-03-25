import logging

from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate

from src.models.PromptimizerLLM import PromptimizerLLM
from src.models.Prompt import Prompt
from enum import Enum, auto


class Promptimizer:
    def __init__(self,
                 llm: PromptimizerLLM,
                 seed_prompt: Prompt,
                 winner_count: int = 1,
                 compress: bool = False,
                 image_gen: bool = False,
                 synthetic_examples: bool = False,
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
        self.test_data = example_data
        self.winner_count = winner_count
        self.optimizedPrompts = None
        self.compress = compress
        self.synthetic_examples = synthetic_examples
        if image_gen:
            self.toa_list = ImageTaskType.enum_to_comma_separated_string()
        else:
            self.toa_list = TaskType.enum_to_comma_separated_string()

        # dictionary of all optimization prompts
        self.optimization_prompts = {}
        with open('src/tuning_prompts/toa_selection.txt', 'r') as file:
            data = file.read().replace('\n', '')
            self.optimization_prompts["toa_selection"] = data
        with open('src/tuning_prompts/semantic_error_backprop.txt', 'r') as file:
            data = file.read().replace('\n', '')
            self.optimization_prompts["semantic_error_generation"] = data
        with open('src/tuning_prompts/error_correction.txt', 'r') as file:
            data = file.read().replace('\n', '')
            self.optimization_prompts["error_correction"] = data
        with open('src/tuning_prompts/float_score_generator.txt', 'r') as file:
            data = file.read().replace('\n', '')
            self.optimization_prompts["float_score_generator"] = data
        with open('src/tuning_prompts/compression_prompt.txt', 'r') as file:
            data = file.read().replace('\n', '')
            self.optimization_prompts["compression_prompt"] = data

        # select target of action for a given seed prompt
        self._select_toa(self.seed_prompt, self.llm.llm_name)

    # exposed method to train the prompt
    def promptimize(self, expansion_factor: int = 10, steps_factor: int = 1):
        """
        :param expansion_factor: number of expansions to perform.
        :param steps_factor: count of evolution steps to perform -> !!this will be deprecated in favor of
        semantic based completion (llm decides it cannot improve the prompt further)!!
        """
        final_prompt = self._promptimize(self.seed_prompt, steps_factor, expansion_factor)

        print(f"final prompt val: {final_prompt.val}")

        compression_template = PromptTemplate(template=self.optimization_prompts["compression_prompt"],
                                              input_variables=["prompt"])

        compression_chain = LLMChain(llm=self.llm.langchain_model,
                                     prompt=compression_template,
                                     verbose=True)

        if self.compress:
            final_prompt.val = compression_chain.run(prompt=final_prompt.val)
            self._score_prompt(final_prompt)
            print(f"final compressed prompt val: {final_prompt.val}")

        self.optimizedPrompts = final_prompt

        return_object = {"original_prompt": self.seed_prompt.val,
                         "original_prompt_score": self.seed_prompt.score,
                         "optimized_prompt": self.optimizedPrompts.val,
                         "optimized_prompt_score": self.optimizedPrompts.score}

        return return_object

    def _promptimize(self, prompt: Prompt, gen_limit: int, expansion_limit: int) -> Prompt:
        print("promptimize started")
        print(f"prompt id: {prompt.id}")
        print(f"prompt val: {prompt.val}")
        print(f"prompt score: {prompt.score}")

        prompt_generation_count = int(prompt.id.split("_")[0])+1

        # STEP 1: ASSESS CURRENT PROMPT, CREATE GRADIENT (for initial)
        if prompt.optimization_vector is None:
            self._critique_prompt(prompt)
        if prompt.score is None:
            self._score_prompt(prompt, self.test_data)

        # STEP 2: EXPAND AND IMPROVE PROMPT BASED ON GRADIENT STEPS TIMES
        # expand the prompt based on the critique steps_limit times -> error correction step
        prompt_candidates = self._expand_prompt(prompt, prompt_generation_count, expansion_limit, monte_carlo=False)
        print(f"prompt candidate ids: {[candidate.id for candidate in prompt_candidates]}")

        # STEP 3: SCORE AND SELECT BEST PROMPTS BASED ON SCORE
        # TODO: implement sorting for Prompt objects on score
        candidates_scored = sorted(prompt_candidates)
        candidates_scored.reverse()
        candidates_scored = candidates_scored[:self.winner_count]

        if candidates_scored[0].score < prompt.score:
            return candidates_scored[0]

        if prompt_generation_count == gen_limit:
            return candidates_scored[0]

        # STEP 4: RECURSE ON TOP PROMPTS
        for next_gen_prompt in candidates_scored:
            return self._promptimize(next_gen_prompt, gen_limit, expansion_limit)

    # given an initial prompt (seed), its target of action (error function to minimize), its semantic gradient,
    # generate expansion_factor improved prompts
    def _expand_prompt(self,
                       prompt_candidate: Prompt,
                       prompt_generation: int,
                       expansion_factor: int,
                       monte_carlo: bool = False) -> list[Prompt]:
        """
        :param prompt_candidate: initial prompt to expand upon.
        :param expansion_factor: number of expansions to perform.
        :param monte_carlo: whether 50% of expansions are semantically identical, but worded differently.
        :return: list of expanded prompts.
        """
        print("expansion started")
        print(f"prompt id: {prompt_candidate.id}")
        print(f"prompt val: {prompt_candidate.val}")
        print(f"prompt score: {prompt_candidate.score}")

        expansions: list[Prompt] = []

        error_correction = PromptTemplate(template=self.optimization_prompts["error_correction"],
                                          input_variables=["prompt", "toa", "llm_name", "semantic_error", "synthetic_examples"])

        expansion_chain = LLMChain(llm=self.llm.langchain_model,
                                   prompt=error_correction,
                                   verbose=True)

        for _ in range(expansion_factor):
            if self.synthetic_examples:
                new_prompt_string = expansion_chain.run(prompt=prompt_candidate.val,
                                                        toa=prompt_candidate.toa,
                                                        llm_name=self.llm.llm_name,
                                                        semantic_error=prompt_candidate.optimization_vector,
                                                        synthetic_examples=f"As part of output, generate synthetic examples to guide {self.llm.llm_name}")

            else:
                new_prompt_string = expansion_chain.run(prompt=prompt_candidate.val,
                                                        toa=prompt_candidate.toa,
                                                        llm_name=self.llm.llm_name,
                                                        semantic_error=prompt_candidate.optimization_vector,
                                                        synthetic_examples="")

            new_prompt = Prompt(gen=prompt_generation,
                                id_in_gen=_,
                                val=new_prompt_string,
                                toa=prompt_candidate.toa)

            self._score_prompt(new_prompt, self.test_data)
            expansions.append(new_prompt)

        print(f"prompt expansion count: {len(expansions)}")
        print(f"prompt expansion ids: {[expansion.id for expansion in expansions]}")
        print(f"prompt expansion scores: {[expansion.score for expansion in expansions]}")

        return expansions

    # TODO: CRITICAL -> PASS CRITIQUE HISTORY OPTIONALLY
    # This would be akin to passing the gradient history to the LLM,
    # which would allow it to learn from the gradient history.
    # given a prompt, its score, training data, and semantic critique
    # use LLM to score it against a target of action (error function to minimize)
    def _critique_prompt(self, prompt: Prompt) -> str:
        """
        :param prompt: prompt to score.
        :return: semantic error of the prompt against the target action, optionally tested against provided data.
        """
        # generate semantic error
        semantic_error_generation = PromptTemplate(template=self.optimization_prompts["semantic_error_generation"],
                                                   input_variables=["prompt", "toa"])

        error_generator_chain = LLMChain(llm=self.llm.langchain_model,
                                         prompt=semantic_error_generation,
                                         verbose=True)

        prompt.optimization_vector = error_generator_chain.run(prompt=prompt.val,
                                                               toa=prompt.toa)

        print(f"critique for {prompt.id}: {prompt.optimization_vector}")

    # TODO: implement Successive Rejects
    # sorts and returns winner_count prompts from all prompts, based on their scores
    # score a prompt against a target of action (error function to minimize) and optional test data
    def _score_prompt(self, prompt: Prompt, data: dict[any, any] = None, semantic: bool = True) -> float:
        """
        :param prompt: prompt to score.
        :param data: optional dictionary of example data (data is used for prompt scoring)
        :param semantic: whether to score the prompt against the target of action (semantic error function)
        :return: score of the prompt against the target of action (semantic error function)
        """
        # TODO: add error based on test data

        # score given prompt against semantic error
        scoring_prompt = PromptTemplate(template=self.optimization_prompts["float_score_generator"],
                                        input_variables=["prompt", "toa", "semantic_error"],
                                        verbose=True)

        scoring_prompt_chain = LLMChain(llm=self.llm.langchain_model,
                                         prompt=scoring_prompt,
                                         verbose=True)

        prompt.score = scoring_prompt_chain.run(prompt=prompt.val,
                                                toa=prompt.toa,
                                                semantic_error=prompt.optimization_vector)

        print(f"score for {prompt.id}: {prompt.score}")

    # set the target of action for a given seed prompt
    def _select_toa(self, prompt: Prompt, llm_name: str):
        toa_selection = PromptTemplate(template=self.optimization_prompts["toa_selection"],
                                       input_variables=["prompt", "toa", "llm"])

        toa_chain = LLMChain(llm=self.llm.langchain_model,
                             prompt=toa_selection,
                             verbose=True)

        prompt.toa = toa_chain.run(prompt=prompt.val,
                                   toa=self.toa_list,
                                   llm=llm_name)

        print(f"toa for seed prompt: {prompt.toa}")


class TaskType(Enum):
    NEEDLE_IN_HAYSTACK = 'needle in haystack'
    CODE_GEN = 'code generation'
    CODE_EVALUATION = 'code evaluation'
    IMAGE_GEN = 'image generation'
    IMAGE_EVAL = 'image evaluation'
    CODE_GEN_TROUBLESHOOT = 'code troubleshooting'
    TEXT_GEN_QA = 'generative question answering'
    TEXT_GEN_SUMMARIZATION = 'text summarization'
    TEXT_GEN_TRANSLATION = 'text translation'
    TEXT_GEN_WRITING = 'text generation'
    TEXT_GEN_CHAT = 'text_gen_chat'
    CATEGORIZATION = 'categorization'
    FACT_CHECKING = 'fact_checking'
    SENTIMENT_ANALYSIS = 'sentiment_analysis'

    @classmethod
    def enum_to_comma_separated_string(cls):
        # Use a list comprehension to extract the values (using .value) of each enum member
        # Then, join these values into a comma-separated string, converting them to strings if necessary
        return ', '.join(str(member.value) for member in TaskType)


class ImageTaskType(Enum):
    GEN_AI_ART_DALL_E = 'generative visual generation prompt for dall-e model'
    GEN_AI_ART_MIDJOURNEY = 'generative visual generation prompt for midjourney model'
    GEN_AI_ART_SD = 'generative visual generation prompt for stability ai models'
    GEN_AI_ART_OTHER = 'generative visual generation prompt'

    @classmethod
    def enum_to_comma_separated_string(cls):
        # Use a list comprehension to extract the values (using .value) of each enum member
        # Then, join these values into a comma-separated string, converting them to strings if necessary
        return ', '.join(str(member.value) for member in TaskType)