from typing import Optional
import streamlit as st


def set_default_session_state_prompts():
    st.session_state['positive_prompt'] = None
    st.session_state['negative_prompt'] = None
    st.session_state['pipeline'] = None


def set_session_state_key(key: str, value: Optional = None):
    st.session_state[key] = value
