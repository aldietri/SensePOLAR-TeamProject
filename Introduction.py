import streamlit as st
from sensepolar.embed.bertEmbed import BERTWordEmbeddings
from sensepolar.oracle.dictionaryapi import Dictionary

st.set_page_config(
    page_title="Welcome Page",
    page_icon="👋",
)

# Changes sidebar color
sidebar_color = """
<style>
[data-testid=stSidebar]{
    background-color: #7895CB;
}
</style>
"""
st.markdown(sidebar_color, unsafe_allow_html=True) 

st.write("# Welcome to SensePOLAR! 🌊")

st.markdown(
    """
    The first (semi-) supervised framework for augmenting interpretability into contextual word embeddings (BERT).
Interpretability is added by rating words on scales that encode user-selected senses, like correctness or left-right-direction.
"""
)

st.write("## Introduction")

st.markdown(
    """
    Adding interpretability to word embeddings represents an area of active research in text representation. Recent work has explored the potential of embedding words via so-called polar dimensions (e.g. good vs. bad, correct vs.wrong). Examples of such recent approaches include SemAxis, POLAR, FrameAxis, and BiImp. Although these approaches provide interpretable dimensions for words, they have not been designed to deal with polysemy, i.e. they can not easily distinguish between different senses of words. To address this limitation, we present SensePOLAR, an extension of the original POLAR framework that enables word-sense aware interpretability for pre-trained contextual word embeddings. The resulting interpretable word embeddings achieve a level of performance that is comparable to original contextual word embeddings across a variety of natural language processing tasks including the GLUE and SQuAD benchmarks. Our work removes a fundamental limitation of existing approaches by offering users sense aware interpretations for contextual word embeddings.
    """
)

st.markdown(
    """
    SensePOLAR overview. Pre-trained contextual word embeddings are transformed into an interpretable space where the word’s semantics are rated on scales individually encoded by opposite senses such as “good”↔“bad”. The scores across the dimensions are representative of the strength of relationship (between word and dimension) which allows us to rank the dimensions and thereby identify the most discriminative dimensions for a word. In this example, the word “wave” is used in two senses: hand waving and ocean wave. SensePOLAR not only generates dimensions that are representative of individual contextual meanings, the alignment to the respective sense spaces also aligns well with human judgement. SensePOLAR generates neutral scores for dimensions not related to the word in the given context (e.g., “idle”↔“work”, “social”↔“unsocial”). We follow the WordNet convention to represent a particular sense of a word. For example, “Tide.v.01” represents the word “tide” in the sense of surge (rise or move forward).
    """
)

st.write("## Beginner Mode")

st.markdown(
    """
    Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet.
    """
)


st.write("## Expert Mode")

st.markdown(
    """
    Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet.
    """
)

@st.cache_resource
def load_bert_model():
    """
    # Load Bert model.
    """
    return BERTWordEmbeddings()

@st.cache_data
def create_dictionary(selected_dict, api_key):
    """
    Returns a dictionary object for the given selected dictionary and api_key.
    
    Parameters:
    -----------
    selected_dict: string
        A string containing the selected dictionary
    api_key: string
        A string containing the selected API key.

    Returns:
    --------
    Dictionary
        Dictionary object fot he selected dictionary and api_key.
    """
    return Dictionary(selected_dict, api_key)

@st.cache_data
def get_dict_definitions(word, selected_dict, api_key):
    """
    Return top 5 most common definitions from the selected dictionary for a given word.

    Parameters:
    -----------
    word : string
        A string containing a word.
    selected_dict: string
        A string containing the selected dictionary
    api_key: string
        A string containing the selected API key.

    Returns:
    --------
    definitions : list
        The top 5 most common word net definitions for the given word.
    """

    # Create dicitonary
    oracle = create_dictionary(selected_dict, api_key)
    
    # Fetch definitions
    definitions = oracle.get_definitions(word)

    # Create subsample (top 5 definitions)
    sample_definitions = [','.join(row) for row in definitions[:5]]

    # Return sample 
    return sample_definitions

# Load bert model into cache so beginner and expert page are more performant
load_bert_model()

# Load dictionary into cache so beinner page is more performant
get_dict_definitions("test", "wordnet", "")

