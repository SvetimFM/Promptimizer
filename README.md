# Promptimizer: Automatic Prompt Optimization Framework
#### Based on [Automatic Prompt Optimization with “Gradient Descent" and Beam Search](https://arxiv.org/pdf/2305.03495.pdf) paper.
#### Promptimizer is a framework designed to optimize textual prompts for interactions with Large Language Models (LLMs). 


### Features:
    Synthetic Example generation: optimization will create synthetic examples against which the seed prompt is modified

    Optional Monte Carlo Expansion: Generates a diverse set of prompt expansions, including semantically identical but differently worded variations, to explore a broader space of potential improvements.

    Semantic Error Critique: Evaluates prompts based on their semantic alignment with a target action or objective, identifying specific areas for improvement.

    Error Correction: Offers direct methods for correcting identified semantic errors in prompts, ensuring targeted refinement towards the optimization goal.

    Coming: Domain-Specific Optimization: Categorizes the optimization task into well-understood domains (e.g., text generation, code generation, sentiment analysis) and applies domain-specific heuristics or strategies for more effective optimization.


### Usage

To use Promptimizer, users should define their initial prompt, target action, and, optionally (to be implemented), a set of example data to guide the optimization process. The framework then iterates through expansion, critique, correction, and selection phases to refine the initial prompt into an optimized version.


### Example usage
 
````
promptimizer = Promptimizer(llm=llm,
                            seed_prompt=input_prompt,
                            winner_count=count_of_winners,
                            example_data=None,
                            compress=compress_final_prompt,
                            image_gen=image_prompt_optimization,
                            synthetic_examples=generate_synthetic_examples)
                            
response = promptimizer.promptimize(expansion_factor=count_of_versions,
                                    steps_factor=count_of_generations)
````


### Installation

Promptimizer is designed to be integrated with existing LLM frameworks and APIs. 

```
git clone https://github.com/SvetimFM/Promptimizer


```


### Contributing

Contributions to Promptimizer are welcome! Whether it's through submitting bug reports, proposing improvements, or adding new features, we value community input. Please refer to our contribution guidelines for more information.


### License

Promptimizer is open source and available under LICENSE.


### Acknowledgements

This project builds upon the foundational work in automatic prompt optimization, including concepts like textual gradients and beam search optimizations. We acknowledge the contributions of researchers and developers in the fields of machine learning and natural language processing that have paved the way for tools like Promptimizer.