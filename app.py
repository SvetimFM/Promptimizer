import streamlit as st
import time

from src.models.PromptimizerLLM import PromptimizerLLM
from src.models.Prompt import Prompt
from src.promptimizer.constants import GPT4_NAME, CLAUDE_NAME, GEMINI_NAME
from src.promptimizer.promptimizer import Promptimizer


# TODO: IMPLEMENT TOA
def webpage():
    init_page()

    st.title("P R O M P T I M I Z E R")
    st.subheader("Best Version of your Prompt, in One Click.")

    st.sidebar.title("Model Configuration")
    llm_selection = st.sidebar.radio('Select LLM:', [GPT4_NAME, CLAUDE_NAME, GEMINI_NAME])
    temp_selection = st.sidebar.slider('Temperature:', 0.0, 1.0, 0.1)  # remove this for release - find best temp (likely either 0.0 or 0.6, might be 0.1)

    st.sidebar.title("Promptimizer Configuration")
    st.sidebar.subheader("(Recommended to leave on default)")
    topic = st.sidebar.text_area('Enter the base prompt to be optimized', st.session_state.history["content"])
    st.sidebar.write("Count of steps in which prompt will be optimized. Greatly increases latency, has diminishing returns")
    count_of_generations = st.sidebar.slider('Number of steps:', 2, 7, 3)
    st.sidebar.write("Count of prompt versions in each step. Greatly increases cost")
    count_of_versions = st.sidebar.slider('Number of versions per step:', 2, 10, 6)

    submit_to_promptimizer = st.sidebar.button('Submit Optimization')

    if submit_to_promptimizer:
        # create an object for validation purposes
        form_dict = {
            "llm_config": {
                "llm_name": llm_selection,
                "temp": temp_selection,
            },
            "prompt_config": {
                "val":  topic,
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

            #try:
            llm_config = form_dict["llm_config"]
            prompt_config = form_dict["prompt_config"]

            llm = PromptimizerLLM(**llm_config)

            # TODO: add manual custom target_of_action
            input_prompt = Prompt(**prompt_config)

            promptimizer = Promptimizer(llm=llm,
                                        seed_prompt=input_prompt,
                                        # TODO: implement frontend for these
                                        winner_count=1,
                                        example_data=None)

            response = promptimizer.promptimize(expansion_factor=count_of_versions,
                                     steps_factor=count_of_generations)

            st.write(response)

            # logging
            end_time = time.time()
            elapsed_time = end_time - start_time
            st.write(f"Request took: {elapsed_time:.2f} seconds")

            #except Exception as e:
            #    st.error(f"Error: {e}")
        else:
            st.error(f"{form_errors}")


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
    if 'history' not in st.session_state:
        st.session_state['history'] = {"role": "user",
                                       "content": "Prompt Goes Here"
                                       }


def main():
    webpage()


if __name__ == '__main__':
    main()
