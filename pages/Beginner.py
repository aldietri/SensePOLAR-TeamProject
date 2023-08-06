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

# Creates entry in session state for antonym pairs
if "definitions" not in st.session_state:
    st.session_state["definitions"] = {}

# entry in session state to check whether this is the first load up of the page
if "initial_page_load_example" not in st.session_state:
    st.session_state["initial_page_load_example"] = True

# entry in session state to check whether this is the first load up of the page
# This needs to be in here to fix some issues between multipages and the loadup
if "initial_page_load_row" not in st.session_state:
    st.session_state["initial_page_load_row"] = True

# Memory to repopulate api_key text field in case the selected dictionary is switched 
if "mem_api_key" not in st.session_state:
    st.session_state["mem_api_key"] =""

# Sidebar

with st.sidebar:
    # Dictionary selection
    st.markdown("# Dictionary")
    selected_dict = st.selectbox("Please select a dictionary", ["wordnet", "wordnik", "dictionaryapi"]) 
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

    # Axes choice for 2D plot
    x_axis = []
    y_axis = []
    if "2D" in selected_options:
        st.markdown("## 2D")
        axes_column = st.columns(2)
        x_axis = axes_column[0].selectbox("x-axis", st.session_state["antonyms"].values(), format_func=lambda x: ", ".join(x))
        y_axis = axes_column[1].selectbox("y-axis", st.session_state["antonyms"].values(), format_func=lambda x: ", ".join(x))

    # Number Input for Most discriminative plot
    k = 3
    if "Most discriminative" in selected_options:
        st.write("## Most Discriminative")
        k = st.number_input("Please select the amount of most discriminative antonym pairs to consider ", min_value=1, max_value=len(st.session_state["antonyms"]))
    
        # Ascending or Descending ordering of Most descriminative antonym pairs
        selected_ordering = st.selectbox("Please select the ordering of the most descriminative antonym pairs", options=["Ascending", "Descending"])

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
    st.session_state[f"def1_{row_id}"] = ""
    st.session_state[f"def2_{row_id}"] = ""
    st.session_state[f"def1_index_{row_id}"] = 0
    st.session_state[f"def2_index_{row_id}"] = 0

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

    # When row is minimized save antonyms into memory else load from memory into relevant session state
    # This is a workaround that needed to be implemented since minimizing the text input fields caused them to be loaded out of the session state in certain scenarios
    if not st.session_state[f"min_{row_id}"]:
        st.session_state[f"mem_ant1_{row_id}"] = st.session_state[f"ant1_{row_id}"]
        st.session_state[f"mem_ant2_{row_id}"] = st.session_state[f"ant2_{row_id}"]
    else:
        st.session_state[f"ant1_{row_id}"] = st.session_state[f"mem_ant1_{row_id}"]
        st.session_state[f"ant2_{row_id}"] = st.session_state[f"mem_ant2_{row_id}"]

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
    else:
        ant1 = st.session_state[f"mem_ant1_{row_id}"]
        ant2 = st.session_state[f"mem_ant2_{row_id}"]
        textColumn.text(f"Polar: {ant1} - {ant2}")

    # TODO: Icon of minimize button dependent on state
    minIcon = ":heavy_minus_sign:" #"🗕" if st.session_state[f"min_{row_id}"] else "🗖"
    delIcon = ":x:" #✖

    # Minimze and Delete buttons
    minCol, delCol = buttonColumns.columns(2)
    minCol.button(minIcon, key=f"minbtn_{row_id}", on_click=toggle_row_min_button, args=[row_id])
    delCol.button(delIcon, key=f"del_{row_id}", on_click=remove_row, args=[row_id])
    
    # Load defintions to populate selectboxes (mainly to reload input after minimizing/maximizing formContainer)
    def1, def2 = st.session_state[f"def1_{row_id}"], st.session_state[f"def2_{row_id}"]

    # Form
    if st.session_state[f"min_{row_id}"]:
        # Container
        formContainer = mainContainer.container()

        # Antonym and Definition Columns
        antCol, defCol = formContainer.columns(2)

        # Antonym text inputs
        ant1 = antCol.text_input("Antonym", key=f"ant1_{row_id}", value=st.session_state[f"mem_ant1_{row_id}"]).strip()
        ant2 = antCol.text_input("Antonym", key=f"ant2_{row_id}", value=st.session_state[f"mem_ant2_{row_id}"], label_visibility="hidden").strip()

        # Fetch wordnet definitions
        definitions1, definitions2 = [], []
        
        if ant1:
            definitions1 = get_dict_definitions(ant1, selected_dict, api_key)
        if ant2:
            definitions2 = get_dict_definitions(ant2, selected_dict, api_key)
        
        # Definition Selectboxes
        def1 = defCol.selectbox("Definition", definitions1, index=st.session_state[f"def1_index_{row_id}"], key=f"select1_{row_id}")
        def2 = defCol.selectbox("Definition", definitions2, index=st.session_state[f"def2_index_{row_id}"], key=f"select2_{row_id}", label_visibility="hidden")

        # Preserve selected defintion values when minimizing entry
        if st.session_state[f"min_{row_id}"] and def1:
            st.session_state[f"def1_{row_id}"] = def1
            def1_index = definitions1.index(def1)
            st.session_state[f"def1_index_{row_id}"] = def1_index
            
        if st.session_state[f"min_{row_id}"] and def2:
            st.session_state[f"def2_{row_id}"] = def2
            def2_index = definitions2.index(def2)
            st.session_state[f"def2_index_{row_id}"] = def2_index

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
    # Count number of occurences of same word in subjjects
    no_occurences = sum([word in example[0] for example in st.session_state["examples"].values()][:idx])
    # Append count as ID
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

# Debugging/Visualization
# st.write(st.session_state)


# SensePOLAR Logic

@st.cache_resource
def load_bert_model():
    """
    # Load Bert model.
    """
    return BERTWordEmbeddings()

@st.cache_data
def create_sense_polar(_model_, antonyms, examples, method, selected_dict, api_key):
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

    # Define paths
    out_path = "./antonyms/"
    antonym_path = out_path + "polar_dimensions.pkl"

    # Initiliaze wordnet dictionary and create lookup files
    dictionary = create_dictionary(selected_dict, api_key)
    lookupSpace = LookupCreator(dictionary=dictionary, out_path=out_path, antonym_pairs=antonyms)
    lookupSpace.create_lookup_files()

    # Create polar Dimensions
    pdc = PolarDimensions(_model_, antonym_path=out_path + "antonym_wordnet_example_sentences_readable_extended.txt")
    pdc.create_polar_dimensions(out_path)

    # Calculate word polarity
    wp = WordPolarity(_model_, antonym_path=antonym_path, lookup_path=out_path, method=method, number_polar=len(antonyms))
    
    words = []
    polar_dimensions = []
    for word, context in examples:
        words.append(word)
        polar_dimensions.append(wp.analyze_word(word, context))  

    # Create result dataframe

    # Value sorting of the respective columns
    antonym_1 = [dim[0] for dim in polar_dimensions[0]] * len(words)
    antonym_2 = [dim[1] for dim in polar_dimensions[0]] * len(words)
    polar_values = [dim[2] for subdim in polar_dimensions for dim in subdim]

    polar_words = np.repeat(words, len(antonym_1)/len(words))

    # Create dataframe
    polar_data = pd.DataFrame({"word": polar_words, 
                               "antonym_1":antonym_1,
                               "antonym_2": antonym_2, 
                               "value": polar_values})

    # Convert dataframe to csv
    df = convert_df_to_csv(polar_data)

    return df, words, polar_dimensions


@st.cache_data
def create_visualisations(options, words, polar_dimensions, k, x_axis, y_axis):
    """
    # Creates visualizations for the word embeddings based on the SensePOLAR Framework implementation.

    Parameters:
    -----------
    options : list
        A list containing the specified plots that are to be returned.
    words : list
        A list containing the analyzed words.
    polar_dimensions: list
        A list containing the polar dimensions of the analyzed words.
    k: int
        An integer to indicate how many antonym pairs to consider when selecting the most discriminative antonym pairs
    x_axis: list
        A list containing the x_axis values that are to be displayed for a 2D plot
    y_axis: list
        A list containing the y_axis values that are to be displayed for a 2D plot
    """

    plotter = PolarityPlotter()

    tabs = st.tabs(options)

    if "Standard" in options:
        fig = plotter.plot_word_polarity(words, polar_dimensions)
        tabs[options.index("Standard")].plotly_chart(fig, use_container_width=True)

    if "2D" in options:
        fig = plotter.plot_word_polarity_2d_interactive(words, polar_dimensions, x_antonym_pair=tuple(x_axis), y_antonym_pair=tuple(y_axis))
        tabs[options.index("2D")].plotly_chart(fig, use_container_width=True)

    if "Polar" in options:
        fig = plotter.plot_word_polarity_polar_fig(words, polar_dimensions)
        tabs[options.index("Polar")].plotly_chart(fig, use_container_width=True)

    if "Most discriminative" in options:
        # TODO: Use selected ordering
        fig = plotter.plot_descriptive_antonym_pairs(words, polar_dimensions, words, k)
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
def convert_df(antonyms, definitions):
    """
    # Converts a the relevant session state data to a downloadable excel file.

    Parameters:
    -----------
    antonyms : dict
        A dict containing antonym pairs.
    definitions: dict
        A dict containg the antonym definitions.

    Returns:
    -----------
    preprocessed_data : bytes
        A bytes object (excel file) generated from a pandas dataframe.
    """

    # Convert to list 
    antonyms = list(antonyms.values())
    definitions = list(definitions.values())

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
        # Create dict to fetch examples for given word
        dictionary = create_dictionary("wordnet", "")
        for idx, antonym_pair in enumerate(antonyms):
            # If antonym pair is initialized update data else populate with empty strings
            if len(antonym_pair) > 0:
                data["antonym_1"].append(antonym_pair[0])
                data["antonym_2"].append(antonym_pair[1])

                # If antonym is populated fetch examples
                try:
                    if antonym_pair[0] != "" and definitions[idx][0]:
                        example = dictionary.get_examples(antonym_pair[0])[0][0] if dictionary.get_examples(antonym_pair[0])[0] else antonym_pair[0]
                        data["example_antonym_1"].append(example)
                except:
                     data["example_antonym_1"].append("")

                try:
                    if antonym_pair[1] != "" and definitions[idx][1]:
                        example = dictionary.get_examples(antonym_pair[1])[0][0] if dictionary.get_examples(antonym_pair[1])[0] else antonym_pair[1]
                        data["example_antonym_2"].append(example)
                except:
                    data["example_antonym_2"].append("")
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
df = convert_df(st.session_state["antonyms"], st.session_state["definitions"])

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
                polar_results, words, polar_dimensions = create_sense_polar(model, st.session_state["antonyms"], st.session_state["examples"], method, selected_dict, api_key)
                # Check if polar dimensions calculation was possible for all words otherwise the context didn't contain the subject word
                if None in polar_dimensions:
                    st.warning("The context must contain your example word", icon="⚠️")
                else:
                    create_visualisations(selected_options, words, polar_dimensions, k, x_axis, y_axis)
            except:
                st.warning("An error has occured. Please check your selected antonyms or API KEY", icon="⚠️")

# If results were calculated show download button for it
if polar_results:
        result_download = st.download_button(label="Download Results", data=polar_results, file_name="SensePolar_Results.csv", key="result_download")
 
# Signifies that first page load is over
if st.session_state["initial_page_load_example"]:
    st.session_state["initial_page_load_example"] = False

# Signifies that first page load is over
if st.session_state["initial_page_load_row"]:
    st.session_state["initial_page_load_row"] = False

# Changes Color of select box widget in row that can't be changed by streamlit itself
ColourWidgetText("Definition", "#FFFFFF")