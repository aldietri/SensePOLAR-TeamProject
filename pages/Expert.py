import streamlit as st
import uuid

from lookup import LookupCreator
from sensepolar.polarity import WordPolarity
from sensepolar.polarDim import PolarDimensions
from sensepolar.plotter import PolarityPlotter
import streamlit.components.v1 as components

from sensepolar.embed.bertEmbed import BERTWordEmbeddings
from sensepolar.embed.albertEmbed import ALBERTWordEmbeddings
from sensepolar.embed.gptEmbed import GPT2WordEmbeddings
from sensepolar.embed.robertaEmbed import RoBERTaWordEmbeddings

import pandas as pd
import numpy as np
import re

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
st.title("🦅 Expert", anchor=False)

# Subheadline with short explaination
with st.expander("Intro", expanded=True):
    st.write("""
        Welcome to our expert page! 🚀 Here, we delve deeper into the powerful SensePOLAR Framework, 
        a versatile page that empowers users with extensive customization options and advanced capabilities. 
        Use in-field customization and effortlessly manage files for optimal NLP performance to 
        unleash the full potential of the SensePOLAR Framework and shape the future of NLP. Let your expertise soar! 🏆
    """)

st.header("Antonyms")

@st.cache_data
def load_dataframe(uploaded_file, template_path):
    """
    # Load pandas dataframe from specified path.

    Parameters:
    -----------
    uploaded_file : UploadedFile class of streamlit
        An uploaded file.
    template_path: String
        A string of the template path.

    Returns:
    -----------
    dataframe/template : pandas.DataFrame
       A pandas dataframe.
    """

    # Load template
    template = pd.read_csv(template_path)
    
    # Check if file was uploaded
    if uploaded_file:
        # Excel
        if ".xlsx" in uploaded_file.name:
            dataframe = pd.read_excel(uploaded_file, header=0)
        # CSV
        else:
            dataframe = pd.read_csv(uploaded_file)

        # Check if uploaded file contains necessary columns of the template layout
        if set(template.columns).issubset(set(dataframe.columns)):
            return dataframe[template.columns]

        # Print warning if uplaoded file does not equal the template layout
        st.warning("Please check whether the uploaded file is formatted in the required way.", icon="⚠️")
    return template

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

def adjust_antonym_counts(df):
    """
    # Counts number of occurences of each antonym and appends them as an ID.

    Parameters:
    -----------
    df : pandas.DataFrame
        A panads dataframe containing the antonym pairs
    """    

    if not df["antonym_1"].isnull().all():
        # Append count as ID
        cum_count_ant1 = df.groupby(["antonym_1"]).cumcount().astype(str)
        df["antonym_1"] = df["antonym_1"] + "_" + cum_count_ant1

    if not df["antonym_2"].isnull().all():
        # Append count as ID
        cum_count_ant2 = df.groupby(["antonym_2"]).cumcount().astype(str)
        df["antonym_2"] = df["antonym_2"] + "_" + cum_count_ant2


# File upload for polar antoym pairs
uploaded_file_polar = st.file_uploader("Already prepared a file?", key="Polar_Upload")

# Load dataframe
dataframe_polar = load_dataframe(uploaded_file_polar, "excel_files_for_experts/Empty_Dataframe_Polar.csv")

# Create data editor
edited_df_polar = st.data_editor(dataframe_polar, num_rows="dynamic", use_container_width=True, key="Polar_data_editor")

# Append counts as ID to each antonym
adjust_antonym_counts(edited_df_polar)

# Save data to session state
st.session_state["df_value_polar"] = edited_df_polar

# Safe edited dataframe to csv
df_edited_polar = convert_df_to_csv(st.session_state["df_value_polar"])

# Download button
download_button = st.download_button(label="Download", data=df_edited_polar, file_name="SensePolar_Antonyms.csv", key="Polar_Download")


# Model name 
if "model_name" not in st.session_state:
    st.session_state["model_name"] = "bert-base-uncased"

# SUBJECT LOGIC

# Creates entry in session state for example input
if "rows_examples" not in st.session_state:
    st.session_state["rows_examples"] = []

# Creates entry in session state for word and context examples
if "examples" not in st.session_state:
    st.session_state["examples"] = {}

# entry in session state to check whether this is the first load up of the page
if "initial_page_load_example" not in st.session_state:
    st.session_state["initial_page_load_example"] = True

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
        word = wordCol.text_input("Word", key=f"word_{example_id}", value=st.session_state[f"mem_word_{example_id}"]).strip()
        context = contextCol.text_input("Context", key=f"context_{example_id}", value=st.session_state[f"mem_context_{example_id}"]).strip()

    # Get idx to have a boundary of which examples are to be checked, i.e., only previous subjects are to be checked and not all to achieve consistent numbering
    idx = st.session_state["rows_examples"].index(example_id)
    # Count number of occurences of same word in subjjects
    no_occurences = sum([word in example[0] for example in st.session_state["examples"].values()][:idx])
    # Append count as ID
    word = word + f"_{no_occurences}"

    # Load word and context into session state
    st.session_state["examples"][example_id] = [word, context]


st.header("Subjects")

# Selection of subject input
example_logic = st.radio(
    "What type of input are you looking for?",
    ("Simple", "Advanced"),
    horizontal=True
)

def adjust_subject_counts(df):
    """
    # Counts number of occurences of each word and appends them as an ID.

    Parameters:
    -----------
    df : pandas.DataFrame
        A panads dataframe containing the word and context of a subject
    """

    # Append count as ID
    if not df["word"].isnull().all():
        cum_count = df.groupby(["word"]).cumcount().astype(str)
        df["word"] = df["word"] + "_" + cum_count
        df["word"] = df["word"].str.replace("_0$", "", regex=True)


# Logic whether simple input or advanced input 

# If simple do normal example input
if example_logic == "Simple":

    # Adds one example at the first page load
    if st.session_state["initial_page_load_example"]:
        add_example()

    # Necessary to add and delete example
    # Recreates example for every example contained in the session state
    for idx, example in enumerate(st.session_state["rows_examples"], start=1):
        # Generate elements word context pairs
        generate_example(example)

    # Create two (three) columns for add example and execute button, array of floats declares size in relation to the other columns
    leftCol, rightCol, _ = st.columns([0.25, 0.25, 1])

    # Add example button
    add_example_button = leftCol.button("Add Subject", on_click=add_example, key="add_sub")
# else allow for advanced example input through file upload and data editor
else:
    # File upload
    uploaded_file_subject = st.file_uploader("Already prepared a file?", key="Subject_Upload")
    
    # Load dataframe
    dataframe_subject = load_dataframe(uploaded_file_subject, "excel_files_for_experts/Empty_Dataframe_Subject.csv")

    # Create data editor
    edited_df_subject = st.data_editor(dataframe_subject, num_rows="dynamic", use_container_width=True, key="Subject_data_editor")

    # Append counts as ID to each subject
    adjust_subject_counts(edited_df_subject)

    # Save data to session state
    st.session_state["df_value_subject"] = edited_df_subject

    # Create two (three) columns for download and execute button, array of floats declares size in relation to the other columns
    leftCol, rightCol, _ = st.columns([0.25, 0.25, 1.1])

    # Safe edited dataframe to csv
    df_edited_subject = convert_df_to_csv(st.session_state["df_value_subject"])

    # Download button
    download_button = leftCol.download_button(label="Download", data=df_edited_subject, file_name="SensePolar_Subject.csv", key="Polar_Subjects")



# SensePOLAR Logic

@st.cache_resource
def load_bert_model(model_name='bert-base-uncased'):
    """
    # Load Bert model.
    """

    if model_name == "albert-base-v2":
        return ALBERTWordEmbeddings(model_name=model_name)
    elif model_name == "bert-base-uncased":
        return BERTWordEmbeddings(model_name=model_name)
    elif model_name == "gpt2":
        return GPT2WordEmbeddings(model_name=model_name)
    elif model_name == "roberta-base":
        return RoBERTaWordEmbeddings(model_name=model_name)

@st.cache_data
def create_sense_polar(_model_, model_name, df, examples, method):
    """
    # Generate word embeddings based on the SensePOLAR Framework implementation.

    Parameters:
    -----------
    model: WordEmbeddings
        A bert model
    model_name: string
        A string for Streamlit cache to update and distinguish the current model in use.
    df : pandas.DataFrame
        A pandas dataframe containing the input antonym and definition data.
    examples : list
        A list containing word and context pairs.
    method : string
        A string containg the transformation method for the antonym space.


    Returns:
    -----------
    words : list
        A list containing the words that were analyzed.
    polar_dimensions: list
        A list containing the polar dimensions of the analyzed words.
    """

    # Define paths
    out_path = "./antonyms/"
    antonym_path = out_path + "polar_dimensions.pkl"

    # create lookup files
    lookupSpace = LookupCreator(dictionary=None, out_path=out_path, antonyms_file_path=df)
    lookupSpace.create_lookup_files()

    # Create polar Dimensions
    pdc = PolarDimensions(_model_, antonym_path=out_path + "antonym_wordnet_example_sentences_readable_extended.txt")
    pdc.create_polar_dimensions(out_path)

    # Calculate word polarity
    wp = WordPolarity(_model_, antonym_path=antonym_path, lookup_path=out_path, method=method, number_polar=len(df))

    words = []
    contexts = []
    polar_dimensions = []
    for word, context in examples:
        words.append(word)
        contexts.append(context)
        polar_dimensions.append(wp.analyze_word(word, context))

    # Create result dataframe

    # Value sorting of the respective columns
    antonym_1 = [dim[0][0] for dim in polar_dimensions[0]] * len(words)
    definition_1 = [dim[0][1] for dim in polar_dimensions[0]] * len(words)
    antonym_2 = [dim[1][0] for dim in polar_dimensions[0]] * len(words)
    definition_2 = [dim[1][1] for dim in polar_dimensions[0]] * len(words)
    polar_values = [dim[2] for subdim in polar_dimensions for dim in subdim]

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
def create_visualisations(options, words, contexts, polar_dimensions, k, x_axis, y_axis, ordering, polar_absolute, polar_axes):
    """
    # Creates visualizations for the word embeddings based on the SensePOLAR Framework implementation.

    Parameters:
    -----------
    options : list
        A list containing the specified plots that are to be returned.
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

    # TODO implement ordering properly
    ordering = "asec" if ordering == "Ascending" else "desc"
    plotter = PolarityPlotter(order_by=ordering)

    tabs = st.tabs(options)

    if "Standard" in options:
        fig = plotter.plot_word_polarity(words, contexts, polar_dimensions)
        tabs[options.index("Standard")].plotly_chart(fig, use_container_width=True)

    if "2D" in options:
        fig = plotter.plot_word_polarity_2d(words, contexts, polar_dimensions, x_axis=x_axis, y_axis=y_axis)
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

# Sidebar

with st.sidebar:
    st.markdown("# Visualization")
    # Select Bert Model
    model_name = st.selectbox("Please select the language model that you want to use", ["albert-base-v2", "bert-base-uncased", "gpt2", "roberta-base"], index=1)
    st.session_state["model_name"] = model_name

    # Select method (projection or base-change)
    method = st.selectbox("Please select a transformation method for the antonym space", ["base-change", "projection"])


    # Select visualisations
    if len(st.session_state["df_value_polar"]) < 2:
        given_options = ["Standard", "Polar", "Most discriminative"]
    else:
        given_options = ["Standard", "2D", "Polar", "Most discriminative"]                 
                           
    # Multiselect to select plots that will be displayed
    selected_options = st.multiselect("Please select some of the given visualization options", given_options)

    selected_ordering = ""
    if "Standard" in selected_options or "Most discriminative" in selected_options:
        st.markdown("## General")
        # Ascending or Descending ordering of Most descriminative antonym pairs
        selected_ordering = st.selectbox("Please select the ordering of the antonym pairs", options=["Ascending", "Descending"])

    # Axes choice for 2d plot
    x_axis_index = 0
    y_axis_index = 0
    axes_values = list(zip(st.session_state["df_value_polar"]["antonym_1"], st.session_state["df_value_polar"]["antonym_2"]))
    axes_values = [[re.sub("_\d", "", ant) for ant in axis] for axis in axes_values]
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

        polar_axes = st.multiselect("Please select the axis values that are to be displayed in the polar plot", axes_values, default=axes_values[0:2], format_func=lambda x: ", ".join(x))

    # Number Input for Most discriminative plot
    k = 3
    if "Most discriminative" in selected_options:
        st.write("## Most Discriminative")
        max_k_value = len(st.session_state["df_value_polar"])
        k = st.number_input("Please select the amount of most discriminative antonym pairs to consider ", min_value=1, max_value=max_k_value, value=max_k_value)

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

def check_inputs(antonyms, examples):
    """
    # Checks whether all relevant input fields are populated in a proper manner.

    Parameters:
    -----------
    ant_df : pandas.DataFrame
        A dataframe containing antonym pairs, definitions and more.
    examples : list
        A list containing word and context pairs.

    Returns:
    -----------
    boolean
        A boolean indicating whether all relevant input fields are populated in a proper manner.
    """

    # Check antonyms
    # If dataframe contains any null or empty string values parse warning and return false
    if antonyms.isnull().values.any() or antonyms.eq("").values.any():
        st.warning("Please check whether all necessary antonym fields have been populated before executing", icon="⚠️")
        return False

    # Check examples
    eval = check_input(examples)
    # If examples contains any null values parse warning and return false
    if not eval:
            st.warning("Please check whether all necessary subject fields have been populated before executing", icon="⚠️")
            return False
    
    # If everything is okay return true
    return True

# Load Bert model
model = load_bert_model(model_name=st.session_state["model_name"])

# Initialize
polar_results = []

# Session state to check whether the result download button has been clicked
# This allows the visualization to automatically reload
if "result_download" not in st.session_state:
    st.session_state["result_download"] = False

# Execute button - Execute SensePOLAR calculation and visualization
if rightCol.button(label="Execute", key="execute") or st.session_state["result_download"] :
    if example_logic == "Simple":
        sub = list(st.session_state["examples"].values())
        contexts = [elem[1] for elem in st.session_state["examples"].values()]
    else:
        sub = list(zip(st.session_state["df_value_subject"]["word"], st.session_state["df_value_subject"]["context"]))
        contexts = list(st.session_state["df_value_subject"]["context"])

    # Checks whether all relevant input fields are populated in a proper manner and then execute calculation and visualization
    if check_inputs(st.session_state["df_value_polar"], sub):
        # Check whether visualization options have been selected
        if not selected_options:
            st.warning("Please select some visualization options", icon="⚠️")
        else:
            # try:
                polar_results, words, polar_dimensions = create_sense_polar(model, st.session_state["model_name"], st.session_state["df_value_polar"], sub, method)
                # Check if polar dimensions calculation was possible for all words otherwise the context didn't contain the subject word
                if None in polar_dimensions:
                    st.warning("The context must contain your example word", icon="⚠️")
                else:
                    create_visualisations(selected_options, words, contexts, polar_dimensions, k, x_axis_index, y_axis_index, selected_ordering, polar_display, polar_axes)
            # except:
                # st.warning("An error has occured. Please check your selected Inputs", icon="⚠️")

# If results were calculated show download button for it
if polar_results:
        result_download = st.download_button(label="Download Results", data=polar_results, file_name="SensePolar_Results.csv", key="result_download")

# Signifies that first page load is over
if st.session_state["initial_page_load_example"]:
    st.session_state["initial_page_load_example"] = False

