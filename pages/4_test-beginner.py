import streamlit as st
from nltk.corpus import wordnet as wn
import uuid

from lookup import LookupCreator
from sensepolar.polarity import WordPolarity
from sensepolar.bertEmbed import BERTWordEmbeddings
from sensepolar.polarDim import PolarDimensions
from sensepolar.dictionaryapi import Dictionary
from sensepolar.plotter import PolarityPlotter
import streamlit.components.v1 as components

import pandas as pd
from io import BytesIO

st.set_page_config(layout="centered", page_title="SensePOLAR", page_icon="🌊")
st.elements.utils._shown_default_value_warning=True

# Removes menu and footer
# hide_streamlit_style = """
# <style>
# #MainMenu {visibility: hidden;}
# footer {visibility: hidden;}

# [data-testid="stVerticalBlock"] .e1tzin5v0 {
#     background-color: #daf5ff;
#     padding: 4px;
#     border-radius: 16px;
# }

# .stTextLabelWrapper.css-y4bq5x.e1j25pv61 {
#     padding: 13px;
# }

# .css-ocqkz7.e1tzin5v3 {
#     padding: 21px;
# }
# </style>
# """
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
st.title("🐣 Beginner Page", anchor=False)

#Subheadline with short explaination
with st.expander("Intro", expanded=True):
    st.write("""
        Welcome to our beginner page! 👋 Here, we introduce you to the SensePOLAR Framework, 
        a user-friendly and accessible tool for harnessing advanced technologies. 
        Designed for beginners and those with limited technical knowledge, this page simplifies the use of the SensePOLAR Framework 
        through an intuitive frontend and a clear step-by-step process. Embrace the SensePOLAR Framework to unlock new possibilities
        in NLP ✨. 
    """)

# ROW LOGIC

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
if "initial_page_load" not in st.session_state:
    st.session_state["initial_page_load"] = True

# entry in session state to check whether this is the first load up of the page
# This needs to be in here to fix some issues between multipages and the loadup
if "initial_page_load_row" not in st.session_state:
    st.session_state["initial_page_load_row"] = True

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

def get_wordnet_definition(word):
    """
    Return top 5 most common word net definitions for a given word.

    Parameters:
    -----------
    word : string
        A string containing a word.

    Returns:
    --------
    definitions : list
        The top 5 most common word net definitions for the given word.
    """
        
    # Fetch Synsets of the given word
    word_synsets = wn.synsets(word)
    # Sets the amount of defintions that will be displayed - 5 or less 
    i_range = 5 if len(word_synsets) > 5 else len(word_synsets)
    # Fetch definitions
    definitions = [word_synsets[i].definition() for i in range(i_range)]
    # Return definitions
    return definitions

def generate_row(row_id):
    """
    # Generates streamlit elements for a given row ID and saves antonym pairs and their definitions into the session state.

    Parameters:
    -----------
    row_id : string
        A string containing the unique row ID.
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
        textColumn.text(f"Pair: {ant1} - {ant2}")
    else:
        ant1 = st.session_state[f"mem_ant1_{row_id}"]
        ant2 = st.session_state[f"mem_ant2_{row_id}"]
        textColumn.text(f"Pair: {ant1} - {ant2}")

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
            definitions1 = get_wordnet_definition(ant1)
        if ant2:
            definitions2 = get_wordnet_definition(ant2)
        
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
    generate_row(row)

# Create button to add row to session state with unqiue ID
st.button("Add Item", on_click=add_row)


# EXAMPLE LOGIC

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
        textColumn.text(f"Example: {word} - {context}")
    else:
        word = st.session_state[f"mem_word_{example_id}"]
        context = st.session_state[f"mem_context_{example_id}"]
        textColumn.text(f"Example: {word} - {context}")

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

    # Load word and context into session state
    st.session_state["examples"][example_id] = [word, context]

# Adds one example at the first page load
if st.session_state["initial_page_load"]:
    add_example()

# Necessary to add and delete example
# Recreates example for every example contained in the session state
for idx, example in enumerate(st.session_state["rows_examples"], start=1):
    # Generate elements word context pairs
    generate_example(example)

# Create button to add example to session state with unqiue ID
st.button("Add Example", on_click=add_example)

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
    # TODO: Implement dict use
    # dictionary = Dictionary(selected_dict, api_key) 
    dictionary = Dictionary("wordnet") 
    lookupSpace = LookupCreator(dictionary, out_path, antonym_pairs=antonyms)
    lookupSpace.create_lookup_files()

    # Create polar Dimensions
    pdc = PolarDimensions(model, antonym_path=out_path + "antonym_wordnet_example_sentences_readable_extended.txt")
    pdc.create_polar_dimensions(out_path)

    # Calculate word polarity
    wp = WordPolarity(model, antonym_path=antonym_path, lookup_path=out_path, method=method, number_polar=len(antonyms))

    words = []
    polar_dimensions = []
    for word, context in examples:
        words.append(word)
        polar_dimensions.append(wp.analyze_word(word, context))

    return words, polar_dimensions

# TODO: Implement axes selection
@st.cache_data
def create_visualisations(options, words, polar_dimensions, k, axes):
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
        An integer to indicate how many antonym pairs to consider when selecting the most descriptive antonym pairs
    axes: list
        A nested list containing the axes that are to be displayed for a 2d plot
    """

    plotter = PolarityPlotter()

    tabs = st.tabs(options)
    count = 0

    if "Standard" in options:
        fig = plotter.plot_word_polarity(words, polar_dimensions)
        tabs[count].plotly_chart(fig, use_container_width=True)
        count += 1

    if "2d" in options:
        fig = plotter.plot_word_polarity_2d(words, polar_dimensions)
        tabs[count].plotly_chart(fig, use_container_width=True)
        count += 1

    if "Polar" in options:
        fig = plotter.plot_word_polarity_polar_fig(words, polar_dimensions)
        tabs[count].plotly_chart(fig, use_container_width=True)
        count += 1

    if "Most descriptive" in options:
        fig = plotter.plot_descriptive_antonym_pairs(words, polar_dimensions, words, k)
        tabs[count].plotly_chart(fig, use_container_width=True)
        count += 1

@st.cache_data
def to_excel(df):
    """
    # Converts a pandas dataframe to an excel file.

    Parameters:
    -----------
    df : pandas.DataFrame
        A pandas dataframe containing the input antonym and definition data.

    Returns:
    -----------
    preprocessed_data : bytes
        A bytes object (excel file) generated from a pandas dataframe.
    """

    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet.set_column('A:A', None, format1)  
    writer.close()
    processed_data = output.getvalue()
    return processed_data

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
        "antonym_1": [""],
        "antonym_2": [""],
        "definition_antonym_1": [""],
        "definition_antonym_2": [""],
    }   

    # Update data dict dependent on the state of the filed text inputs
    if antonyms:
            data["antonym1"] = [antonym_pair[0] if len(antonym_pair) > 0 else "" for antonym_pair in antonyms]
            data["antonym2"] = [antonym_pair[1] if len(antonym_pair) > 0 else "" for antonym_pair in antonyms]

    if definitions:
            data["definition_antonym1"] = [definition_pair[0] if len(definition_pair) > 0 else "" for definition_pair in definitions]
            data["definition_antonym2"] = [definition_pair[1] if len(definition_pair) > 0 else "" for definition_pair in definitions]

    # Transform data dict to pandas dataframe to excel file
    # Orient index and transpose are necessary to fix some bugs that were caused by misalligned array sizes
    df = to_excel(pd.DataFrame.from_dict(data, orient="index").transpose())

    return df

# Sidebar

with st.sidebar:
    # Select method (projection or base-change)
    method = st.selectbox("Please select a transformation method for the antonym space", ["base-change", "projection"])
    if len(st.session_state["antonyms"]) < 2:
        given_options = ["Standard", "Polar", "Most descriptive"]
    else:
        given_options = ["Standard", "2d", "Polar", "Most descriptive"]                 
                           
    # Multiselect to select plots that will be displayed
    selected_options = st.multiselect("Please select some of the given visualization options", given_options)

    # Axes choice for 2d plot
    selected_axes = []
    if "2d" in selected_options:
        selected_axes = st.multiselect("Please select two dimensions you want to display", st.session_state["antonyms"].values(), max_selections=2)

    # Number Input for most descriptive plot
    k = 3
    if "Most descriptive" in selected_options:
        k = st.number_input("Please select the amount of most descriptive antonym pairs to consider ", min_value=1, max_value=len(st.session_state["antonyms"]))

    # Dictionary selection
    selected_dict = st.selectbox("Please select a dictionary", ["wordnet", "wordnik", "oxford", "dictionaryapi"]) 
    if selected_dict in ["wordnik", "dictionaryapi"]:
        api_key = st.text_input("Please insert your API KEY").strip()
        # Changes Color of select box widget in row that can't be changed by streamlit itself
        ColourWidgetText("Please insert your API KEY", "#31333F")
    else:
        api_key = ""
        
# Create two columns for download and execute button, array of floats declares size in relation to the other columns
downloadCol, executeCol, _ = st.columns([0.2, 0.3, 0.8])

# Get excel file from inputs
df = convert_df(st.session_state["antonyms"], st.session_state["definitions"])

# Download button
download_button = downloadCol.download_button(label="Download", data=df, file_name="SensePolar.xlsx")

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
    
    # Again loop through examples
    for pair in examples:
        # Check whether context contains example word and display warning if not as well as return false
        if not f" {pair[0].lower()} " in f" {pair[1].lower()} ":
            st.warning("The context must contain your example word", icon="⚠️")
            return False
    
    # If everything is okay return true
    return True

# Load Bert model
model = load_bert_model()

# Execute button - Execute SensePOLAR calculation and visualization
if executeCol.button("Execute"):
    # Checks whether all relevant input fields are populated in a proper manner and then execute calculation and visualization
    if check_inputs(st.session_state["antonyms"], st.session_state["definitions"], st.session_state["examples"]):
        # Check whether visualization options have been selected
        if not selected_options:
            st.warning("Please select some visualization options", icon="⚠️")
        else:
            try:
                words, polar_dimensions = create_sense_polar(model, st.session_state["antonyms"], st.session_state["examples"], method, selected_dict, api_key)
                create_visualisations(selected_options, words, polar_dimensions, k, selected_axes)
            except:
                st.warning("An error has occured. Please check your selected antonyms", icon="⚠️")



# Signifies that first page load is over
if st.session_state["initial_page_load"]:
    st.session_state["initial_page_load"] = False

# Signifies that first page load is over
if st.session_state["initial_page_load_row"]:
    st.session_state["initial_page_load_row"] = False


# Changes Color of select box widget in row that can't be changed by streamlit itself
ColourWidgetText("Definition", "#FFFFFF")