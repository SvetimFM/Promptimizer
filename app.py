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

    title_left, title_center, title_right = st.columns(3)
    title_left.markdown("<h1 style='text-align: left; color: white; padding-top: 10px;'>P R O M P T I M I Z E R</h1>",
                        unsafe_allow_html=True)
    title_right.markdown("""
    <div style='display: flex; justify-content: flex-end; align-items: center; height: 100%;'>
        <h2 text-align: right; style='color: white;'>Best Prompt. One Click</h2>
    </div>
    """, unsafe_allow_html=True)


    st.sidebar.title("Model Configuration")
    llm_selection = st.sidebar.radio('Select LLM: (Gemini available for free)', [GPT4_NAME, CLAUDE_NAME, GEMINI_NAME], index=2)
    temp_selection = st.sidebar.slider('Temperature:', 0.0, 1.0, 0.1)  # remove this for release - find best temp (likely either 0.0 or 0.6, might be 0.1)
    st.sidebar.info("Set above 0.6 for image prompts")

    st.sidebar.title("Promptimizer Configuration")
    count_of_generations = st.sidebar.slider('Number of steps:', 2, 7, 3)
    count_of_versions = st.sidebar.slider('Number of expansions per step:', 2, 10, 4)
    count_of_winners = st.sidebar.slider('Number of winners selected in each step:', 1, 5, 2)
    st.sidebar.markdown("---")

    st.sidebar.title("Description")
    st.sidebar.markdown("- **Number of steps** - count of evolutions over which the prompt will be improved")
    st.sidebar.markdown("- **Number of expansions** - how many candidate prompts are generated in each step")
    st.sidebar.markdown("- **Number of winners** - number of prompts that get expanded in the following step")
    st.sidebar.info("For more information, see [README](https://github.com/SvetimFM/Promptimizer/blob/main/README.md)")

    main_tab, key_config = st.tabs(["Home", "API Keys"])
    with st.container(border=True):
        main_tab.markdown("")
        topic = main_tab.text_area('Enter your prompt', st.session_state.history["content"], height=250)

        main_tab.markdown("---")

        image_prompt_optimization = main_tab.toggle("Image Generation Prompt")
        compress_final_prompt = main_tab.toggle("Shorten Output Prompt")
        generate_synthetic_examples = main_tab.toggle("Generate Synthetic Examples")

        main_tab.markdown("---")

        submit_to_promptimizer = main_tab.button('Optimize!', use_container_width=True)
        main_tab.markdown("")

    with st.container(border=True):
        key_config.markdown("")
        key_config_chat_gpt = key_config.text_input('ChatGPT API Key', '')
        key_config_anthropic = key_config.text_input('Anthropic API Key', '')
        key_config_gemini = key_config.text_input('Gemini API Key', '')

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

            input_prompt = Prompt(**prompt_config)

            promptimizer = Promptimizer(llm=llm,
                                        seed_prompt=input_prompt,
                                        winner_count=count_of_winners,
                                        example_data=None,
                                        compress=compress_final_prompt,
                                        image_gen=image_prompt_optimization,
                                        synthetic_examples=generate_synthetic_examples)
            try:
                with st.spinner("Optimizing prompt... (takes ~2 minutes)"):
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
        cont = st.container(border=False)

        left, right = cont.columns(2, gap='small')

        left.markdown("**Original Prompt:**")
        left.info(f"Original Prompt Score: **{output['original_prompt_score']}**")
        right.markdown("**Optimized Prompt:**")
        right.info(f"Optimized Prompt Score: **{output['optimized_prompt_score']}**")

        left_container = left.container(height=400, border=True)
        left_container.write(f"{output['original_prompt']}")

        right_container = right.container(height=400, border=True)
        right_container.write(f"{output['optimized_prompt']}")


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
    st.set_page_config(layout="wide",
                       page_title="Promptimizer",
                       page_icon="✨")

    st.markdown(
        """
       <style>
       [data-testid="stSidebar"][aria-expanded="true"]{
           min-width: 350px;
       }
       """,
        unsafe_allow_html=True,
    )

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
