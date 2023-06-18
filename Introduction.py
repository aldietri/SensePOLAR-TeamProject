import streamlit as st

st.set_page_config(
    page_title="Welcome Page",
    page_icon="👋",
)

st.write("# Welcome to SensePOLAR! 🌊")

st.markdown(
    """
    SensePOLAR is the first (semi-) supervised framework for augmenting interpretability into contextual word embeddings (BERT). Interpretability is added by rating words on scales that encode user-selected senses, like correctness or left-right-direction.
"""
)