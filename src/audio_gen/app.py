import streamlit as st
from st_config import start_page
from st_sidebar import set_sidebar
import st_components as stc

# STARTUP
start_page()
set_sidebar()

# INTRO
stc.welcome_intro()
st.divider()

# QUERIES
st.subheader("Prompt")
stc.set_prompt()
stc.print_current_prompts()

st.divider()

# GENERATION
st.subheader("Generate Samples")
with st.spinner('Getting pipeline...'):
    stc.get_pipeline()
with st.spinner('Generating audios...'):
    finished = stc.run_pipeline()
    if finished:
        st.balloons()
