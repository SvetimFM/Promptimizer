import asyncio
import threading

import streamlit as st
import time

from src.models.PromptimizerLLM import PromptimizerLLM
from src.models.Prompt import Prompt
from src.promptimizer.constants import GPT4_NAME, CLAUDE_NAME, GEMINI_NAME
from src.promptimizer.promptimizer import Promptimizer

response_list: list[dict] = []

# TODO: IMPLEMENT TOA
def webpage():
    init_page()

    st.title("P R O M P T I M I Z E R")
    st.subheader("Best Prompt. One Click.")

    st.sidebar.title("Model Configuration")
    llm_selection = st.sidebar.radio('Select LLM: (Gemini available for free)', [GPT4_NAME, CLAUDE_NAME, GEMINI_NAME], index=2)
    temp_selection = st.sidebar.slider('Temperature:', 0.0, 1.0,
                                       0.1)  # remove this for release - find best temp (likely either 0.0 or 0.6, might be 0.1)

    st.sidebar.title("Promptimizer Configuration")
    count_of_generations = st.sidebar.slider('Number of steps:', 2, 7, 3)
    count_of_versions = st.sidebar.slider('Number of expansions per step:', 2, 10, 4)
    count_of_winners = st.sidebar.slider('Number of winners in each step:', 1, 5, 2)

    main_tab, key_config = st.tabs(["Home", "API Keys"])
    with st.container(border=True):
        topic = main_tab.text_area('Enter your prompt', st.session_state.history["content"])
        image_prompt_optimization = main_tab.toggle("Image Generation Prompt")
        compress_final_prompt = main_tab.toggle("Shorten Output Prompt")
        generate_synthetic_examples = main_tab.toggle("Generate Synthetic Examples")
        submit_to_promptimizer = main_tab.button('Optimize!')

    with st.container(border=True):
        key_config_chat_gpt = key_config.text_input('ChatGPT API Key', '')
        key_config_anthropic = key_config.text_input('Anthropic API Key', '')
        key_config_gemini = key_config.text_input('Gemini API Key', '')

    progress = 0

    if submit_to_promptimizer:
        # create an object for validation purposes
        form_dict = {
            "llm_config": {
                "llm_name": llm_selection,
                "temp": temp_selection,
                "api_keys": {"chat_gpt": key_config_chat_gpt,
                             "anthropic": key_config_anthropic,
                             "gemini": key_config_gemini}
            },
            "prompt_config": {
                "val": topic,
                "gen_count": count_of_generations,
                "ver_count": count_of_versions,
                "gen": 0,
                "id_in_gen": 0,
            }
        }

        # save session
        st.session_state['history'] = {
            "role": "user",
            "content": form_dict["prompt_config"]["val"]
        }

        form_errors = validate_form(form_dict)
        if not form_errors:
            start_time = time.time()

            llm_config = form_dict["llm_config"]
            prompt_config = form_dict["prompt_config"]

            llm = PromptimizerLLM(**llm_config)

            # TODO: add manual custom target_of_action
            input_prompt = Prompt(**prompt_config)

            promptimizer = Promptimizer(llm=llm,
                                        seed_prompt=input_prompt,
                                        # TODO: implement frontend for these
                                        winner_count=count_of_winners,
                                        example_data=None,
                                        compress=compress_final_prompt,
                                        image_gen=image_prompt_optimization,
                                        synthetic_examples=generate_synthetic_examples)
            try:
                with st.spinner("Optimizing prompt..."):
                    response = promptimizer.promptimize(expansion_factor=count_of_versions,
                                                        steps_factor=count_of_generations)

                    st.session_state['outputs'].append(response)

            except Exception as e:
                st.error(f"Uh oh! {e}")
                return
            st.success("Optimization complete!")

            # logging
            end_time = time.time()
            elapsed_time = end_time - start_time
            st.sidebar.write(f"Request took: {elapsed_time:.2f} seconds")

        else:
            st.error(f"{form_errors}")

    for output in st.session_state.get('outputs', []):
        cont = st.container(height=600, border=True)
        left, right = cont.columns(2, gap='large')
        left.write(f"Original Prompt Score: {output['original_prompt_score']}")
        left.write(f"Original Prompt: \n\n {output['original_prompt']}")
        right.write(f"Optimized Prompt Score: {output['optimized_prompt_score']}")
        right.write(f"\n {output['optimized_prompt']}")


# returns a list of errors in the form
def validate_form(form_dict: dict[dict]) -> list[dict]:
    # fields exist check
    missing_fields = []
    for key, form_sub_item in form_dict.items():
        for test, value in form_sub_item.items():
            if value is None:
                missing_fields.append([test, value])

    if missing_fields:
        return {"required fields missing": missing_fields}
    else:
        return {}


# creating a cookie to store user form history between reloads
def init_page():
    st.set_page_config(page_title="Promptimizer")
    if 'outputs' not in st.session_state:
        st.session_state['outputs'] = []

    if 'history' not in st.session_state:
        st.session_state['history'] = {"role": "user",
                                       "content": "Prompt Goes Here"
                                       }


def main():
    webpage()


if __name__ == '__main__':
    main()
