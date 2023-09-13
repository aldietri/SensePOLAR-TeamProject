import streamlit as st
import uuid

from lookup import LookupCreator
from sensepolar.polarity import WordPolarity
from sensepolar.embed.bertEmbed import BERTWordEmbeddings
from sensepolar.polarDim import PolarDimensions
from sensepolar.oracle.dictionaryapi import Dictionary
from sensepolar.plotter import PolarityPlotter
import streamlit.components.v1 as components

import pandas as pd
import numpy as np

st.set_page_config(layout="centered", page_title="SensePOLAR", page_icon="🌊")
st.elements.utils._shown_default_value_warning=True

# Removes menu and footer
# hide_streamlit_style = """
# <style>
# #MainMenu {visibility: hidden;}
# footer {visibility: hidden;}

# st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# TODO: When creating running file force theme?
# https://discuss.huggingface.co/t/can-i-set-the-dark-theme-in-streamlit/17843/3
# Another option would be to pass --theme.base dark in the command line when you run your app.

# Changes block color and other attributes
block_color = """
<style>
[data-testid="stHorizontalBlock"]:nth-of-type(1){
    background-color: #7895CB;
    padding: 4px;
    border-radius: 10px;
    color: #FFFFFF
}
</style>
"""
st.markdown(block_color, unsafe_allow_html=True) 

# TODO: May need to be changed or removed, dependent on future adjustments of the site
# Changes color of text input labels
label_color = """
<style>
.stTextInput > label{
    color: #FFFFFF
}
</style>
"""
st.markdown(label_color, unsafe_allow_html=True) 

# Changes sidebar color
sidebar_color = """
<style>
[data-testid=stSidebar]{
    background-color: #7895CB;
}
</style>
"""
st.markdown(sidebar_color, unsafe_allow_html=True) 

# TODO: Can possibly be removed
# Changes button allignment to righ (min and del button)
# btn_alignment = """
# <style>
# [data-testid="stHorizontalBlock"]:nth-of-type(1){
#     text-align: right;
# }
# </style>
# """
# st.markdown(btn_alignment, unsafe_allow_html=True) 

# Changes alignment of text - centers headerContainer text to buttons
text_alignment = """
<style>
[data-testid="stText"] {
    padding-top: 11px
}
</style>
"""
st.markdown(text_alignment, unsafe_allow_html=True) 

# Changes the color of help tooltip to white
tooltip_hover_target = """
<style>
.css-900wwq svg {
    stroke: #FFFFFF
}
</style>
"""
st.markdown(tooltip_hover_target, unsafe_allow_html=True) 

#
colour_widget_display = """
<style>
[data-stale="false"] {
    display: None
}
</style>
"""
# st.markdown(colour_widget_display, unsafe_allow_html=True) 

# def ChangeWidgetFontSize(wgt_txt, wch_font_size = '12px'):
#     htmlstr = """<script>var elements = window.parent.document.querySelectorAll('*'), i;
#                     for (i = 0; i < elements.length; ++i) { if (elements[i].innerText == |wgt_txt|) 
#                         { elements[i].style.fontSize='""" + wch_font_size + """';} } </script>  """

#     htmlstr = htmlstr.replace('|wgt_txt|', "'" + wgt_txt + "'")
#     components.html(f"{htmlstr}", height=0, width=0)

# Changes color of widget text with certain text
def ColourWidgetText(wgt_txt, wch_colour = '#000000'):
    htmlstr = """<script>var elements = window.parent.document.querySelectorAll('*'), i;
                    for (i = 0; i < elements.length; ++i) { if (elements[i].innerText == |wgt_txt|) 
                        elements[i].style.color = ' """ + wch_colour + """ '; } </script>  """

    htmlstr = htmlstr.replace('|wgt_txt|', "'" + wgt_txt + "'")
    components.html(f"{htmlstr}", height=0, width=0)

# Headline
st.title("🐣 Beginner", anchor=False)

#Subheadline with short explaination
with st.expander("Intro", expanded=True):
    st.write("""
        Welcome to our beginner page! 👋 Here, we introduce you to the SensePOLAR Framework, 
        a user-friendly and accessible tool for harnessing advanced technologies. 
        Designed for beginners and those with limited technical knowledge, this page simplifies the use of the SensePOLAR Framework 
        through an intuitive frontend and a clear step-by-step process. Embrace the SensePOLAR Framework to unlock new possibilities
        in NLP ✨. 
    """)

st.header("Antonyms")

# Creates entry in session state for rows
if "rows_antonyms" not in st.session_state:
    st.session_state["rows_antonyms"] = []

# Creates entry in session state for antonym pairs
if "antonyms" not in st.session_state:
    st.session_state["antonyms"] = {}

# Creates entry in session state for definitions for antonym pairs
if "definitions" not in st.session_state:
    st.session_state["definitions"] = {}

# Creates entry in session state to track indices of selected definitions for antonym pairs
if "def_indices" not in st.session_state:
    st.session_state["def_indices"] = {}

# entry in session state to check whether this is the first load up of the page
if "initial_page_load_example" not in st.session_state:
    st.session_state["initial_page_load_example"] = True

# entry in session state to check whether this is the first load up of the page
# This needs to be in here to fix some issues between multipages and the loadup
if "initial_page_load_row" not in st.session_state:
    st.session_state["initial_page_load_row"] = True

# Memory to repopulate api_key text field in case the selected dictionary is switched 
if "mem_api_key" not in st.session_state:
    st.session_state["mem_api_key"] = ""

# Sidebar

with st.sidebar:
    # Dictionary selection
    st.markdown("# Dictionary")
    selected_dict = st.selectbox("Please select a dictionary", ["wordnet", "wordnik [Experimental]", "dictionaryapi [Experimental]"]).split(" ")[0]
    api_key = ""
    if selected_dict == "wordnik":
        api_key = st.text_input("Please insert your API KEY", key="api_key", value=st.session_state["mem_api_key"]).strip()
        # Save api key to memory
        st.session_state["mem_api_key"] = api_key
        # Changes Color of select box widget in row that can't be changed by streamlit itself
        ColourWidgetText("Please insert your API KEY", "#31333F")

    # Select method (projection or base-change)
    st.markdown("# Visualization")
    method = st.selectbox("Please select a transformation method for the antonym space", ["base-change", "projection"])
    if len(st.session_state["antonyms"]) < 2:
        given_options = ["Standard", "Polar", "Most discriminative"]
    else:
        given_options = ["Standard", "2D", "Polar", "Most discriminative"]                 
                           
    # Multiselect to select plots that will be displayed
    selected_options = st.multiselect("Please select some of the given visualization options", given_options, key="visualization_options")

    selected_ordering = ""
    if "Standard" in selected_options or "Most discriminative" in selected_options:
        st.markdown("## General")
        # Ascending or Descending ordering of Most descriminative antonym pairs
        selected_ordering = st.selectbox("Please select the ordering of the antonym pairs", options=["Ascending", "Descending"])
        
    # Axes choice for 2d plot and polar plot
    x_axis_index = 0
    y_axis_index = 0
    axes_values = list(st.session_state["antonyms"].values())
    axes_values = [[ant.split("_")[0] for ant in axis] for axis in axes_values]
    if "2D" in selected_options:
        st.markdown("## 2D")
        axes_column = st.columns(2)
        x_axis = axes_column[0].selectbox("x-axis", axes_values, format_func=lambda x: ", ".join(x))
        x_axis_index = axes_values.index(x_axis)
        y_axis = axes_column[1].selectbox("y-axis", axes_values, format_func=lambda x: ", ".join(x))
        y_axis_index = axes_values.index(y_axis)

    # Polar Plot settings
    polar_axes = []
    polar_display = ""
    if "Polar" in selected_options:
        st.markdown("## Polar")
        polar_display = st.selectbox("Please select the way the polar axes should be displayed in", ["solo", "grouped"])

        polar_axes = st.multiselect("Please select the axis values that are to be displayed in the polar plot", axes_values, default=axes_values[0:len(axes_values)], format_func=lambda x: ", ".join(x))

    # Number Input for Most discriminative plot
    k = 3
    if "Most discriminative" in selected_options:
        st.write("## Most Discriminative")
        max_k_value = max(1, len(st.session_state["antonyms"]))
        k = st.number_input("Please select the amount of most discriminative antonym pairs to consider ", min_value=1, max_value=max_k_value, value=max_k_value)
    
# ROW LOGIC

def add_row():
    """
    Adds a row containing antonym pairs, their definitions and other related entities to the streamlit session state.
    """

    # Construct unique row ID
    row_id = str(uuid.uuid4())

    # Row
    st.session_state["rows_antonyms"].append(row_id)

    # Antonyms
    st.session_state[f"ant1_{row_id}"] = ""
    st.session_state[f"ant2_{row_id}"] = ""
    st.session_state[f"mem_ant1_{row_id}"] = ""
    st.session_state[f"mem_ant2_{row_id}"] = ""

    # Minimize/Maximize Button
    st.session_state[f"min_{row_id}"] = True

    # Antonym Definitions and their Index
    st.session_state[f"select1_{row_id}"] = ""
    st.session_state[f"select2_{row_id}"] = ""
    st.session_state[f"mem_select1_{row_id}"] = ""
    st.session_state[f"mem_select2_{row_id}"] = ""
    st.session_state[f"definitions1_{row_id}"] = []
    st.session_state[f"definitions2_{row_id}"] = []

def remove_row(row_id):
    """
    Removes a row containing antonym pairs, their definitions and other related entities from the streamlit session state.

    Parameters:
    -----------
    row_id : string
        A string containing the unique row ID.
    """
    
    # Row
    st.session_state["rows_antonyms"].remove(row_id)

    # Antonyms
    del st.session_state["antonyms"][row_id]

    # Definitions
    del st.session_state["definitions"][row_id]

    # Everything else related to the given row ID
    for entry in st.session_state:
        if row_id in entry:
            del st.session_state[entry]

def toggle_row_min_button(row_id):
    """
    Minimizes/Maximizes the row and saves/restores relevant information contained in the minimized container.

    Parameters:
    -----------
    row_id : string
        A string containing the unique row ID.
    """

    # Update state of minimize/maximize button in session state
    st.session_state[f"min_{row_id}"] = not st.session_state[f"min_{row_id}"]

    # When row is minimized save antonyms and defintions into memory else load from memory into relevant session state
    # This is a workaround that needed to be implemented since minimizing the text input fields caused them to be loaded out of the session state in certain scenarios
    if not st.session_state[f"min_{row_id}"]:
        st.session_state[f"mem_ant1_{row_id}"] = st.session_state[f"ant1_{row_id}"]
        st.session_state[f"mem_ant2_{row_id}"] = st.session_state[f"ant2_{row_id}"]
        st.session_state[f"mem_select1_{row_id}"] = st.session_state[f"select1_{row_id}"]
        st.session_state[f"mem_select2_{row_id}"] = st.session_state[f"select2_{row_id}"]
    else:
        st.session_state[f"ant1_{row_id}"] = st.session_state[f"mem_ant1_{row_id}"]
        st.session_state[f"ant2_{row_id}"] = st.session_state[f"mem_ant2_{row_id}"]
        st.session_state[f"select1_{row_id}"] = st.session_state[f"mem_select1_{row_id}"]
        st.session_state[f"select2_{row_id}"] = st.session_state[f"mem_select2_{row_id}"]

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

def get_dict_example(word, selected_dict, api_key, index):
    if word == "":
        return 0, ""
    
    # Create dicitonary
    oracle = create_dictionary(selected_dict, api_key)

    # Fetch all examples
    try:
        examples = oracle.get_examples(word)
    except:
        st.warning("An error with the dictionary has occured please reload the page or reselect the dictionary", icon="⚠️")
        return 0, ""

    # Loop through examples until an example can be fetched
    example = ""
    for _ in range(len(examples)):
        if len(examples[index]) > 0:
           example = examples[index][0]
           break 
        else:
            index += 1

    return index, example

def generate_row(row_id, selected_dict, api_key):
    """
    # Generates streamlit elements for a given row ID and saves antonym pairs and their definitions into the session state.

    Parameters:
    -----------
    row_id : string
        A string containing the unique row ID.
    selected_dict: string
        A string containing the selected dictionary
    api_key: string
        A string containing the selected API key.
    """

    # List containing antonym data
    antonym_pair = []

    # List containing antonym data
    definition_pair = []

    # Main Container
    mainContainer = st.container()

    # Header
    headerContainer = mainContainer.container()
    textColumn, buttonColumns = headerContainer.columns([4.7, 1])

    # Preserve antonym values when minimzing entry
    # if else initializiation is to preserve value and prevent error when switching between multipages
    if st.session_state[f"min_{row_id}"]:
        ant1 = st.session_state[f"ant1_{row_id}"] if f"ant1_{row_id}" in st.session_state else st.session_state[f"mem_ant1_{row_id}"]
        ant2 = st.session_state[f"ant2_{row_id}"] if f"ant2_{row_id}" in st.session_state else st.session_state[f"mem_ant2_{row_id}"]
        textColumn.text(f"Polar: {ant1} - {ant2}")

        def1 = st.session_state[f"select1_{row_id}"] if f"select1_{row_id}" in st.session_state else st.session_state[f"mem_select1_{row_id}"]
        def2 = st.session_state[f"select2_{row_id}"] if f"select2_{row_id}" in st.session_state else st.session_state[f"mem_select2_{row_id}"]
    else:
        ant1 = st.session_state[f"mem_ant1_{row_id}"]
        ant2 = st.session_state[f"mem_ant2_{row_id}"]
        textColumn.text(f"Polar: {ant1} - {ant2}")

        def1 = st.session_state[f"mem_select1_{row_id}"]
        def2 = st.session_state[f"mem_select2_{row_id}"]

    minIcon = ":heavy_minus_sign:"
    delIcon = ":x:"

    # Minimze and Delete buttons
    minCol, delCol = buttonColumns.columns(2)
    minCol.button(minIcon, key=f"minbtn_{row_id}", on_click=toggle_row_min_button, args=[row_id])
    delCol.button(delIcon, key=f"del_{row_id}", on_click=remove_row, args=[row_id])
    
    # Load defintions to populate selectboxes
    definitions1, definitions2 = st.session_state[f"definitions1_{row_id}"], st.session_state[f"definitions2_{row_id}"]

    # Form
    if st.session_state[f"min_{row_id}"]:
        # Container
        formContainer = mainContainer.container()

        # Antonym and Definition Columns
        antCol, defCol = formContainer.columns(2)

        # Get indices of definitions
        def1_index = definitions1.index(def1) if definitions1 and def1 else 0
        def2_index = definitions2.index(def2) if definitions2 and def2 else 0

        # Fetch examples from defintion
        def1_index, example1 = get_dict_example(ant1, selected_dict, api_key, def1_index)
        def2_index, example2 = get_dict_example(ant2, selected_dict, api_key, def2_index)

        # Track index pair of selected definitions and save to session state
        index_pair = [def1_index, def2_index]
        st.session_state["def_indices"][row_id] = index_pair

        # Antonym text inputs
        ant1 = antCol.text_input("Antonym", help=example1, key=f"ant1_{row_id}", value=ant1).strip()
        ant2 = antCol.text_input("Antonym", help=example2, key=f"ant2_{row_id}", value=ant2).strip()

        # Fetch wordnet definitions and save to sessions state
        if ant1:
            definitions1 = get_dict_definitions(ant1, selected_dict, api_key)
            st.session_state[f"definitions1_{row_id}"] = definitions1

        if ant2:
            definitions2 = get_dict_definitions(ant2, selected_dict, api_key)
            st.session_state[f"definitions2_{row_id}"] = definitions2

        # Checks if antonyms are present and if not resets definitions selectboxes
        if ant1 == "":
            definitions1 = []
        if ant2 == "":
            definitions2 = []

        # Definition Selectboxes
        def1 = defCol.selectbox("Definition", definitions1, index=def1_index, key=f"select1_{row_id}")
        def2 = defCol.selectbox("Definition", definitions2, index=def2_index, key=f"select2_{row_id}", label_visibility="hidden")
        
        # Change color of defintion text
        # ColourWidgetText("Definition", "#FFFFFF")

    # Get idx to have a boundary of which examples are to be checked, i.e., only previous subjects are to be checked and not all to achieve consistent numbering
    idx = st.session_state["rows_antonyms"].index(row_id)

    # Count number of occurences of same antonym in antonym pairs
    try:
        no_occurences_1 = sum([(ant1 in ant_pair[0] or ant1 in ant_pair[1]) for ant_pair in st.session_state["antonyms"].values()][:idx])
        no_occurences_2 = sum([(ant2 in ant_pair[0] or ant2 in ant_pair[1]) for ant_pair in st.session_state["antonyms"].values()][:idx])
    except:
        no_occurences_1 = 0
        no_occurences_2 = 0

    # Append count as ID if antonym does not equal empty string
    if ant1:
        ant1 = ant1 + f"_{no_occurences_1}"
    if ant2:
        ant2 = ant2 + f"_{no_occurences_2}"

    # Add antonym pair to designated list
    if ant1 or ant2:
        antonym_pair = [ant1, ant2]

    # Add definitions of the antonym pair to designated list
    if def1 or def2:
        definition_pair = [def1, def2]


    # Safe antonym pair and definition data to session state
    st.session_state["antonyms"][row_id] = antonym_pair
    st.session_state["definitions"][row_id] = definition_pair


# Adds one row at the first page load
if st.session_state["initial_page_load_row"]:
    add_row()

# Necessary to add and delete rows
# Recreates rows for every row contained in the session state
for idx, row in enumerate(st.session_state["rows_antonyms"], start=1):  
    # Generate elements for antonym pairs
    generate_row(row, selected_dict, api_key)

# Create button to add row to session state with unqiue ID
st.button("Add Polar", on_click=add_row)


st.header("Subjects")

# SUBJECT LOGIC

# Creates entry in session state for example input
if "rows_examples" not in st.session_state:
    st.session_state["rows_examples"] = []

# Creates entry in session state for word and context examples
if "examples" not in st.session_state:
    st.session_state["examples"] = {}

def add_example():
    """
    Adds an example containing words, their context and other related entities to the streamlit session state.
    """

    # Construct unique ID for example 
    example_id = str(uuid.uuid4())

    # Example
    st.session_state["rows_examples"].append(example_id)

    # Word
    st.session_state[f"word_{example_id}"] = ""
    st.session_state[f"mem_word_{example_id}"] = ""

    # Minimizie/Maximize Button
    st.session_state[f"min_{example_id}"] = True

    # Context
    st.session_state[f"context_{example_id}"] = ""
    st.session_state[f"mem_context_{example_id}"] = ""

def remove_example(example_id):
    """
    Removes an example containing words, their context and other related entities from the streamlit session state.

    Parameters:
    -----------
    example_id : string
        A string containing the unique example ID.
    """
    
    # Example
    st.session_state["rows_examples"].remove(example_id)

    # Word and Context
    del st.session_state["examples"][example_id]

    # Everything else related to the given example ID.
    for entry in st.session_state:
        if example_id in entry:
            del st.session_state[entry]

def toggle_example_min_button(example_id):
    """
    Minimizes/Maximizes the example and saves/restores relevant information contained in the minimized container.

    Parameters:
    -----------
    row_id : string
        A string containing the unique example ID.
    """

    # Update state of minimize/maximize button in session state
    st.session_state[f"min_{example_id}"] = not st.session_state[f"min_{example_id}"]

    # When example is minimized save word and context into memory else load from memory into relevant session state
    # This is a workaround that needed to be implemented since minimizing the text input fields caused them to be loaded out of the session state in certain scenarios
    if not st.session_state[f"min_{example_id}"]:
        st.session_state[f"mem_word_{example_id}"] = st.session_state[f"word_{example_id}"]
        st.session_state[f"mem_context_{example_id}"] = st.session_state[f"context_{example_id}"]
    else:
        st.session_state[f"word_{example_id}"] = st.session_state[f"mem_word_{example_id}"]
        st.session_state[f"context_{example_id}"] = st.session_state[f"mem_context_{example_id}"]


def generate_example(example_id):
    """
    # Generates streamlit elements for a given example ID and saves words and their contexts into the session state.

    Parameters:
    -----------
    row_id : string
        A string containing the unique example ID.
    """

    # Main Container
    mainContainer = st.container()

    # Header
    headerContainer = mainContainer.container()
    textColumn, buttonColumns = headerContainer.columns([4.7, 1])

    # Preserve word and context values when minimzing entry
    # if else initializiation is to preserve value and prevent error when switching between multipages
    if st.session_state[f"min_{example_id}"]:
        word = st.session_state[f"word_{example_id}"] if f"word_{example_id}" in st.session_state else st.session_state[f"mem_word_{example_id}"]
        context = st.session_state[f"context_{example_id}"] if f"context_{example_id}" in st.session_state else st.session_state[f"mem_context_{example_id}"]
        textColumn.text(f"Subject: {word} - {context}")
    else:
        word = st.session_state[f"mem_word_{example_id}"]
        context = st.session_state[f"mem_context_{example_id}"]
        textColumn.text(f"Subject: {word} - {context}")

    # Minimze and Delete buttons
    minCol, delCol = buttonColumns.columns(2)
    minCol.button(":heavy_minus_sign:", key=f"minbtn_{example_id}", on_click=toggle_example_min_button, args=[example_id])
    delCol.button(":x:", key=f"del_{example_id}", on_click=remove_example, args=[example_id])

    # Form
    if st.session_state[f"min_{example_id}"]:
        # Container
        formContainer = mainContainer.container()
        wordCol, contextCol = formContainer.columns(2)

        # Antonym text inputs
        word = wordCol.text_input("Word", key=f"word_{example_id}", value=word).strip()
        context = contextCol.text_input("Context", key=f"context_{example_id}", value=context).strip()

    # Get idx to have a boundary of which examples are to be checked, i.e., only previous subjects are to be checked and not all to achieve consistent numbering
    idx = st.session_state["rows_examples"].index(example_id)
    # Count number of occurences of same word in subjects
    no_occurences = sum([word in example[0] for example in st.session_state["examples"].values()][:idx])
    # Append count as ID
    if no_occurences > 0:
        word = word + f"_{no_occurences}"

    # Load word and context into session state
    st.session_state["examples"][example_id] = [word, context]

# Adds one example at the first page load
if st.session_state["initial_page_load_example"]:
    add_example()

# Necessary to add and delete example
# Recreates example for every example contained in the session state
for idx, example in enumerate(st.session_state["rows_examples"], start=1):
    # Generate elements word context pairs
    generate_example(example)

# Create button to add example to session state with unqiue ID
st.button("Add Subject", on_click=add_example)

# SensePOLAR Logic

@st.cache_resource
def load_bert_model(model_name='bert-base-uncased'):
    """
    # Load Bert model.
    """
    return BERTWordEmbeddings(model_name=model_name)

@st.cache_data
def create_sense_polar(_model_, antonyms, examples, indices, method, selected_dict, api_key):
    """
    # Generate word embeddings based on the SensePOLAR Framework implementation.

    Parameters:
    -----------
    model: BertWordEmbeddings
        A bert model
    antonyms : dict
        A dict containing antonym pairs.
    examples : dict
        A dict containing word and context pairs.
    indices: dict
        A dict containing the indices of the selected definitions of an antonym
    method : string
        A string containg the transformation method for the antonym space.
    selected_dict: string
        A string containing the selected dictionary
    api_key: string
        A string containing the selected API key.

    Returns:
    -----------
    words : list
        A list containing the words that were analyzed.
    polar_dimensions: list
        A list containing the polar dimensions of the analyzed words.
    """

    # Convert to list 
    antonyms = list(antonyms.values())
    examples = list(examples.values())
    indices = list(indices.values())

    # Define paths
    out_path = "./antonyms/"
    antonym_path = out_path + "polar_dimensions.pkl"

    # Initiliaze wordnet dictionary and create lookup files
    dictionary = create_dictionary(selected_dict, api_key)
    lookupSpace = LookupCreator(dictionary=dictionary, out_path=out_path, antonym_pairs=antonyms)
    lookupSpace.create_lookup_files(indices)

    # Create polar Dimensions
    pdc = PolarDimensions(_model_, antonym_path=out_path + "antonym_wordnet_example_sentences_readable_extended.txt")
    pdc.create_polar_dimensions(out_path)

    # Calculate word polarity
    wp = WordPolarity(_model_, antonym_path=antonym_path, lookup_path=out_path, method=method)

    words = []
    contexts = []
    polar_dimensions = []

    for word, context in examples:
        words.append(word)
        contexts.append(context)
        polar_dimensions.append(wp.analyze_word(word, context))  

    # Create result dataframe

    ordered_polar_dimensions = []
    for dim in polar_dimensions:
            order_dim = sorted(dim, key=lambda x: (x[0][0][0], x[0][1][0]))
            ordered_polar_dimensions.append(order_dim)

    # Value sorting of the respective columns
    antonym_1 = [dim[0][0] for dim in ordered_polar_dimensions[0]] * len(words)
    definition_1 = [dim[0][1] for dim in ordered_polar_dimensions[0]] * len(words)
    antonym_2 = [dim[1][0] for dim in ordered_polar_dimensions[0]] * len(words)
    definition_2 = [dim[1][1] for dim in ordered_polar_dimensions[0]] * len(words)
    polar_values = [dim[2] for subdim in ordered_polar_dimensions for dim in subdim]

    polar_words = np.repeat(words, len(antonym_1)/len(words))
    polar_contexts = np.repeat(contexts, len(antonym_1)/len(contexts))

    # Create dataframe
    polar_data = pd.DataFrame({"word": polar_words,
                               "context": polar_contexts,  
                               "antonym_1": antonym_1,
                               "definition_1": definition_1,
                               "antonym_2": antonym_2, 
                               "definition_2": definition_2,
                               "value": polar_values})

    # Replace simicolons with other placeholder so that conversion to csv is not messed up
    polar_data["definition_1"] = polar_data["definition_1"].str.replace(";", "|")
    polar_data["definition_2"] = polar_data["definition_2"].str.replace(";", "|")

    # Convert dataframe to csv
    df = convert_df_to_csv(polar_data)

    return df, words, polar_dimensions

@st.cache_data
def create_visualisations(options, antonyms, definitions, words, contexts, polar_dimensions, k, x_axis, y_axis, ordering, polar_absolute, polar_axes):
    """
    # Creates visualizations for the word embeddings based on the SensePOLAR Framework implementation.

    Parameters:
    -----------
    options : list
        A list containing the specified plots that are to be returned.
    antonyms : dict
        A dict containing antonym pairs.
    definitions: dict
        A dict containg the antonym definitions.
    words : list
        A list containing the analyzed words.
    contexts: list
        A list containing the context of the analyzed words.
    polar_dimensions: list
        A list containing the polar dimensions of the analyzed words.
    k: int
        An integer to indicate how many antonym pairs to consider when selecting the most discriminative antonym pairs.
    x_axis: int
        A number containing the index of the selected x-axis value.
    y_axis: int
        A number containing the index of the selected y-axis value.
    ordering: string
        A string indicating the ordering of the polar values.
    polar_absolute: string
        A string indicating the desired display of the polar axes in the plot.
    polar_axes: list
        A list containing the axes that are to be displayed in the polar plot.
    """

    ordering = "asec" if ordering == "Ascending" else "desc"
    plotter = PolarityPlotter(sort_by="descriptive", order_by=ordering)

    tabs = st.tabs(options)

    if "Standard" in options:
        fig = plotter.plot_word_polarity(words, contexts, polar_dimensions)
        tabs[options.index("Standard")].plotly_chart(fig, use_container_width=True)

    if "2D" in options:
        antonyms = list(antonyms.values())
        definitions = list(definitions.values())

        # Iterate through both lists and create antonym definition pairs
        ant_def_pair = []
        for sublist1, sublist2 in zip(antonyms, definitions):
            pair = []
            for ant, definition in zip(sublist1, sublist2):
                pair.append([ant, definition])
            ant_def_pair.append(pair)

        x_axis_value = ant_def_pair[x_axis]
        y_axis_value = ant_def_pair[y_axis]

        fig = plotter.plot_word_polarity_2d(words, contexts, polar_dimensions, x_axis=x_axis, y_axis=y_axis)
        # fig = plotter.plot_word_polarity_2d(words, contexts, polar_dimensions, x_axis=x_axis_value, y_axis=y_axis_value)
        tabs[options.index("2D")].plotly_chart(fig, use_container_width=True)

    if "Polar" in options:
        if polar_absolute == "grouped":
            fig = plotter.plot_word_polarity_polar_absolute(words, contexts, polar_dimensions, polar_axes)
        else:
            fig = plotter.plot_word_polarity_polar(words, contexts, polar_dimensions, polar_axes)
        tabs[options.index("Polar")].plotly_chart(fig, use_container_width=True)

    if "Most discriminative" in options:
        fig = plotter.plot_descriptive_antonym_pairs(words, contexts, polar_dimensions, words, k)
        tabs[options.index("Most discriminative")].plotly_chart(fig, use_container_width=True)

@st.cache_data
def convert_df_to_csv(df):
    """
    # Convert pandas Dataframe to csv file.

    Parameters:
    -----------
    df : pandas.Dataframe
        A pandas dataframe.

    Returns:
    -----------
    A csv file.
    """

    return df.to_csv(index=False).encode('utf-8')


@st.cache_data
def convert_df(antonyms, definitions, indices):
    """
    # Converts a the relevant session state data to a downloadable excel file.

    Parameters:
    -----------
    antonyms : dict
        A dict containing antonym pairs.
    definitions: dict
        A dict containg the antonym definitions.
    indices: dict
        A dict containing the indices of the selected antonym definitions.

    Returns:
    -----------
    preprocessed_data : bytes
        A bytes object (excel file) generated from a pandas dataframe.
    """

    # Convert to list 
    antonyms = list(antonyms.values())
    definitions = list(definitions.values())
    def_indices = list(indices.values())

    # Standard layout
    data = {
        "antonym_1": [],
        "antonym_2": [],
        "example_antonym_1": [],
        "example_antonym_2": [],
        "def1": [],
        "def2": [],
    }   
    
    # Update data dict dependent on the state of the filed text inputs
    if antonyms:
        for idx, antonym_pair in zip(def_indices, antonyms):
            # If antonym pair is initialized update data else populate with empty strings
            if len(antonym_pair) > 0:
                ant1 = antonym_pair[0].split("_")[0]
                ant2 = antonym_pair[1].split("_")[0]

                data["antonym_1"].append(ant1)
                data["antonym_2"].append(ant2)

                 # Fetch examples from defintion
                _, example1 = get_dict_example(ant1, selected_dict, api_key, idx[0]) if ant1 else ("", "")
                _, example2 = get_dict_example(ant2, selected_dict, api_key, idx[1]) if ant2 else ("", "")

                # Save to dict
                data["example_antonym_1"].append(example1)
                data["example_antonym_2"].append(example2)

            else:
                data["antonym_1"].append("")
                data["antonym_2"].append("")
                data["example_antonym_1"].append("")
                data["example_antonym_2"].append("")

    # Update definitions dependent on the state of the field
    if definitions:
            data["def1"] = [definition_pair[0] if len(definition_pair) > 0 else "" for definition_pair in definitions]
            data["def2"] = [definition_pair[1] if len(definition_pair) > 0 else "" for definition_pair in definitions]


    # Transform data dict to pandas dataframe to excel file
    # Orient index and transpose are necessary to fix some bugs that were caused by misalligned array sizes
    df = convert_df_to_csv(pd.DataFrame.from_dict(data, orient="index").transpose())

    return df

# Create two columns for download and execute button, array of floats declares size in relation to the other columns
downloadCol, executeCol, _ = st.columns([0.2, 0.3, 0.8])

# Get excel file from inputs
df = convert_df(st.session_state["antonyms"], st.session_state["definitions"], st.session_state["def_indices"])

# Download button
download_button = downloadCol.download_button(label="Download", data=df, file_name="SensePolar.csv")

def check_input(input):
    """
    # Checks whether an input is null or contains null values.

    Parameters:
    -----------
    input : list
        A nested list containing values.

    Returns:
    -----------
    boolean
        A boolean indicating whether the input is null or contains null values.
    """

    # If input is null return false
    if not input:
        return False
    
    # Loop through nested list
    for pair in input:
        # If inner list is null return false
        if not pair:
            return False
        # Else loop through inner entries
        for entry in pair:
            # If inner entry is null return false
            if not entry:
                return False
    
    # If everything is okay return true
    return True

def check_inputs(antonyms, definitions, examples):
    """
    # Checks whether all relevant input fields are populated in a proper manner.

    Parameters:
    -----------
    antonyms : dict
        A dict containing antonym pairs.
    definitions: dict
        A dict containg the antonym definitions.
    examples : dict
        A dict containing word and context pairs.

    Returns:
    -----------
    boolean
        A boolean indicating whether all relevant input fields are populated in a proper manner.
    """

    # Convert to list 
    antonyms = list(antonyms.values())
    definitions = list(definitions.values())
    examples = list(examples.values())

    # Loop through all inputs
    for pair in [antonyms, definitions, examples]:
        # Get evaluation
        eval = check_input(pair)
        # If a flaw has been found display warning and return false
        if not eval:
            st.warning("Please check whether all necessary input fields have been populated before executing", icon="⚠️")
            return False
    
    # If everything is okay return true
    return True

# Load Bert model
model = load_bert_model()

# Initialize
polar_results = []

# Session state to check whether the result download button has been clicked
# This allows the visualization to automatically reload
if "result_download" not in st.session_state:
    st.session_state["result_download"] = False

# Execute button - Execute SensePOLAR calculation and visualization
if executeCol.button("Execute") or st.session_state["result_download"]:
    # Checks whether all relevant input fields are populated in a proper manner and then execute calculation and visualization
    if check_inputs(st.session_state["antonyms"], st.session_state["definitions"], st.session_state["examples"]):
        # Check whether visualization options have been selected
        if not selected_options:
            st.warning("Please select some visualization options", icon="⚠️")
        else:
            try:
                polar_results, words, polar_dimensions = create_sense_polar(model, st.session_state["antonyms"], st.session_state["examples"], st.session_state["def_indices"], method, selected_dict, api_key)
                # Check if polar dimensions calculation was possible for all words otherwise the context didn't contain the subject word
                if None in polar_dimensions:
                    st.warning("The context must contain your example word", icon="⚠️")
                else:
                    contexts = [elem[1] for elem in st.session_state["examples"].values()]
                    create_visualisations(selected_options, st.session_state["antonyms"], st.session_state["definitions"], words, contexts, polar_dimensions, k, x_axis_index, y_axis_index, selected_ordering, polar_display, polar_axes)
            except:
                st.warning("An error has occured. Please check your selected antonyms or API KEY", icon="⚠️")

# If results were calculated show download button for it
if polar_results:
        result_download = st.download_button(label="Download Results", data=polar_results, file_name="SensePolar_Results.csv", key="result_download")
 
# Change color of defintion text
ColourWidgetText("Definition", "#FFFFFF")

# Signifies that first page load is over
if st.session_state["initial_page_load_example"]:
    st.session_state["initial_page_load_example"] = False

# Signifies that first page load is over
if st.session_state["initial_page_load_row"]:
    st.session_state["initial_page_load_row"] = False