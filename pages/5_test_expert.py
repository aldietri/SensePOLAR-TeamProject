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

# Changes block color and other attributes
upload_color = """
<style>
[data-testid="stFileUploader"]{
    background-color: #7895CB;
    padding: 4px;
    border-radius: 10px;
    color: #FFFFFF
}
</style>
"""
st.markdown(upload_color, unsafe_allow_html=True) 

# TODO: May need to be changed or removed, dependent on future adjustments of the site
# Changes color of text input labels
upload_label_color = """
<style>
[data-testid="stFileUploader"] > label{
    color: #FFFFFF
}
</style>
"""
st.markdown(upload_label_color, unsafe_allow_html=True) 


# Changes alignment of text - centers headerContainer text to buttons
text_alignment = """
<style>
[data-testid="stText"] {
    padding-top: 11px
}
</style>
"""
st.markdown(text_alignment, unsafe_allow_html=True) 

# Changes color of widget text with certain text
def ColourWidgetText(wgt_txt, wch_colour = '#000000'):
    htmlstr = """<script>var elements = window.parent.document.querySelectorAll('*'), i;
                    for (i = 0; i < elements.length; ++i) { if (elements[i].innerText == |wgt_txt|) 
                        elements[i].style.color = ' """ + wch_colour + """ '; } </script>  """

    htmlstr = htmlstr.replace('|wgt_txt|', "'" + wgt_txt + "'")
    components.html(f"{htmlstr}", height=0, width=0)

# Headline
st.title("🦅 Expert Page", anchor=False)

# Subheadline with short explaination
with st.expander("Intro", expanded=True):
    st.write("""
        Welcome to our expert page! 🚀 Here, we delve deeper into the powerful SensePOLAR Framework, 
        a versatile page that empowers users with extensive customization options and advanced capabilities. 
        Switch dictionaries, use in-field customization, and effortlessly manage files for optimal NLP performance. 
        Unleash the full potential of the SensePOLAR Framework and shape the future of NLP. Let your expertise soar! 🏆
    """)

# File upload
uploaded_file = st.file_uploader("Already prepared a file?")
# Check if file was uplaoded
if uploaded_file:
    # Excel
    if ".xlsx" in uploaded_file.name:
        dataframe = pd.read_excel(uploaded_file,header=0)
    # CSV
    else:
        dataframe = pd.read_csv(uploaded_file)
# If no file uploaded read template
else:
    dataframe = pd.read_csv("excel_files_for_experts/Empty_Dataframe.csv")


edited_df = st.data_editor(dataframe, num_rows="dynamic", use_container_width=True)

st.session_state["df_value"] = edited_df


# EXAMPLE LOGIC

# Creates entry in session state for example input
if "rows_examples" not in st.session_state:
    st.session_state["rows_examples"] = []

# Creates entry in session state for word and context examples
if "examples" not in st.session_state:
    st.session_state["examples"] = {}

# entry in session state to check whether this is the first load up of the page
if "initial_page_load" not in st.session_state:
    st.session_state["initial_page_load"] = True

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
        word = wordCol.text_input("Word", key=f"word_{example_id}", value=st.session_state[f"mem_word_{example_id}"]).strip()
        context = contextCol.text_input("Context", key=f"context_{example_id}", value=st.session_state[f"mem_context_{example_id}"]).strip()

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


# SensePOLAR Logic

@st.cache_resource
def load_bert_model():
    """
    # Load Bert model.
    """
    return BERTWordEmbeddings()

@st.cache_data
def create_sense_polar(_model_, df, examples, method):
    """
    # Generate word embeddings based on the SensePOLAR Framework implementation.

    Parameters:
    -----------
    model: BertWordEmbeddings
        A bert model
    df : pandas.DataFrame
        A pandas dataframe containing the input antonym and definition data.
    examples : dict
        A dict containing word and context pairs.
    method : string
        A string containg the transformation method for the antonym space.


    Returns:
    -----------
    words : list
        A list containing the words that were analyzed.
    polar_dimensions: list
        A list containing the polar dimensions of the analyzed words.
    """

    # Convert to list 
    examples = list(examples.values())

    # Define paths
    out_path = "./antonyms/"
    antonym_path = out_path + "polar_dimensions.pkl"

    # create lookup files
    lookupSpace = LookupCreator(out_path=out_path, antonyms_file_path=df)
    lookupSpace.create_lookup_files()

    # Create polar Dimensions
    pdc = PolarDimensions(model, antonym_path=out_path + "antonym_wordnet_example_sentences_readable_extended.txt")
    pdc.create_polar_dimensions(out_path)

    # Calculate word polarity
    wp = WordPolarity(model, antonym_path=antonym_path, lookup_path=out_path, method=method, number_polar=len(df))

    words = []
    polar_dimensions = []
    for word, context in examples:
        words.append(word)
        polar_dimensions.append(wp.analyze_word(word, context))

    return words, polar_dimensions

# TODO: Implemt axes selection
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

# Sidebar

with st.sidebar:
    # Select method (projection or base-change)
    method = st.selectbox("Please select a transformation method for the antonym space", ["base-change", "projection"])
    if len(st.session_state["df_value"]) < 2:
        given_options = ["Standard", "Polar", "Most descriptive"]
    else:
        given_options = ["Standard", "2d", "Polar", "Most descriptive"]                 
                           
    # Multiselect to select plots that will be displayed
    selected_options = st.multiselect("Please select some of the given visualization options", given_options)

    # Axes choice for 2d plot
    selected_axes = []
    if "2d" in selected_options:
        antonym_dims = st.session_state["df_value"]["antonym_1"] + "-" + st.session_state["df_value"]["antonym_2"]
        selected_axes = st.multiselect("Please select two dimensions you want to display", antonym_dims, max_selections=2)

    # Number Input for most descriptive plot
    k = 3
    if "Most descriptive" in selected_options:
        k = st.number_input("Please select the amount of most descriptive antonym pairs to consider ", min_value=1, max_value=len(st.session_state["df_value"]))

# Create two columns for download and execute button, array of floats declares size in relation to the other columns
downloadCol, executeCol, _ = st.columns([0.2, 0.3, 0.8])

# Safe edited dataframe to excel
df_edited_xls = to_excel(st.session_state["df_value"])

# Download button
download_button = downloadCol.download_button(label="Download", data=df_edited_xls, file_name="SensePolar.xlsx")

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

def check_inputs(ant_df, examples):
    """
    # Checks whether all relevant input fields are populated in a proper manner.

    Parameters:
    -----------
    ant_df : pandas.DataFrame
        A dataframe containing antonym pairs, definitions and more.
    examples : dict
        A dict containing word and context pairs.

    Returns:
    -----------
    boolean
        A boolean indicating whether all relevant input fields are populated in a proper manner.
    """

    # Convert to list s())
    examples = list(examples.values())

    # Check antonyms
    # If dataframe contains any null values parse warning and return false
    if ant_df.isnull().values.any():
        st.warning("Please check whether all necessary input fields have been populated before executing", icon="⚠️")
        return False

    # Check examples
    eval = check_input(examples)
    # If examples contains any null values parse warning and return false
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
    if check_inputs(st.session_state["df_value"], st.session_state["examples"]):
        # Check whether visualization options have been selected
        if not selected_options:
            st.warning("Please select some visualization options", icon="⚠️")
        else:
            try:
                words, polar_dimensions = create_sense_polar(model, st.session_state["df_value"], st.session_state["examples"], method)
                create_visualisations(selected_options, words, polar_dimensions, k, selected_axes)
            except:
                st.warning("An error has occured. Please check your selected antonyms", icon="⚠️")
    


# Signifies that first page load is over
if st.session_state["initial_page_load"]:
    st.session_state["initial_page_load"] = False

