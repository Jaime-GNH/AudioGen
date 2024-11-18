import streamlit as st
from st_session_management import set_default_session_state_prompts
from utils import get_env_vars


def start_page():
    st.set_page_config(page_title="AI Audio Gen", page_icon=':loudspeaker:',
                       layout="wide", initial_sidebar_state="collapsed")
    if not st.session_state.get("positive_prompt"):
        set_default_session_state_prompts()

    if st.session_state.get('first_run', True):
        get_env_vars()
        st.session_state['first_run'] = False
