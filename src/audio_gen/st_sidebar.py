import os
import streamlit as st
from st_session_management import set_default_session_state_prompts


def set_sidebar():
    st.sidebar.subheader("""
    Additional Configuration
    """)
    st.session_state['reporting'] = st.sidebar.text_input('Reporting directory:',
                                                          os.path.abspath(os.environ.get('REPORTING_DIR',
                                                                                         '../../reporting')))
    st.sidebar.markdown("##\n\n##")
    st.sidebar.divider()
    st.sidebar.button("Reset prompt", on_click=set_default_session_state_prompts)
