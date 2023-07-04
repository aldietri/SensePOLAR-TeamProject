import streamlit as st
from nltk.corpus import wordnet as wn
import uuid

from lookup import LookupCreator
from sensepolar.polarity import WordPolarity
from sensepolar.bertEmbed import BERTWordEmbeddings
from sensepolar.polarDim import PolarDimensions
from sensepolar.dictionaryapi import Dictionary
from sensepolar.plotter import PolarityPlotter
# import streamlit.components.v1 as components

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

# Changes block color and other attributes
block_color = """
<style>
[data-testid="stHorizontalBlock"]:nth-of-type(1){
    background-color: #daf5ff;
    padding: 4px;
    border-radius: 10px;
}
</style>
"""
st.markdown(block_color, unsafe_allow_html=True) 

# Changes button allignment to righ (min and del button)
btn_alignment = """
<style>
[data-testid="stHorizontalBlock"]:nth-of-type(1){
    text-align: right;
}
</style>
"""
st.markdown(btn_alignment, unsafe_allow_html=True) 

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

# Headline
st.write("# Welcome to Beginner Page")

#Subheadline with short explaination
st.markdown(
    """
    WIP: This page allows you to use the SensePOLAR Framework in a very simple and straight forward manner.
    """
)

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

# Adds row to session state with unqiue ID
def add_row():
    row_id = str(uuid.uuid4())
    st.session_state["rows_antonyms"].append(row_id)
    st.session_state[f"ant1_{row_id}"] = ""
    st.session_state[f"ant2_{row_id}"] = ""
    st.session_state[f"mem_ant1_{row_id}"] = ""
    st.session_state[f"mem_ant2_{row_id}"] = ""
    st.session_state[f"min_{row_id}"] = True
    st.session_state[f"def1_{row_id}"] = ""
    st.session_state[f"def2_{row_id}"] = ""
    st.session_state[f"def1_index_{row_id}"] = 0
    st.session_state[f"def2_index_{row_id}"] = 0

# Remove row from session state for a give ID
def remove_row(row_id):
    st.session_state["rows_antonyms"].remove(row_id)
    del st.session_state["antonyms"][row_id]
    del st.session_state["definitions"][row_id]
    for entry in st.session_state:
        if row_id in entry:
            del st.session_state[entry]

def toggle_row_min_button(row_id):
    st.session_state[f"min_{row_id}"] = not st.session_state[f"min_{row_id}"]

    if not st.session_state[f"min_{row_id}"]:
        st.session_state[f"mem_ant1_{row_id}"] = st.session_state[f"ant1_{row_id}"]
        st.session_state[f"mem_ant2_{row_id}"] = st.session_state[f"ant2_{row_id}"]
    else:
        st.session_state[f"ant1_{row_id}"] = st.session_state[f"mem_ant1_{row_id}"]
        st.session_state[f"ant2_{row_id}"] = st.session_state[f"mem_ant2_{row_id}"]

def get_wordnet_definition(word):
    # Fetch Synsets of given word
    word_synsets = wn.synsets(word)
    # Sets the amount of defintions that will be displayed - 5 or less 
    i_range = 5 if len(word_synsets) > 5 else len(word_synsets)
    # Fetch definitions
    definitions = [word_synsets[i].definition() for i in range(i_range)]
    # Return definitions
    return definitions

# Generates streamlit elements with given row ID
def generate_row(row_id):
    # List containing antonym data
    antonym_pair = []

    # List containing antonym data
    definition_pair = []

    mainContainer = st.container()
    # header
    headerContainer = mainContainer.container()
    textColumn, buttonColumns = headerContainer.columns([4.7, 1])

    # Preserve antonym values when minimzing entry
    if st.session_state[f"min_{row_id}"]:
        ant1 = st.session_state[f"ant1_{row_id}"]
        ant2 = st.session_state[f"ant2_{row_id}"]
        textColumn.text(f"Pair: {ant1} - {ant2}")
    else:
        ant1 = st.session_state[f"mem_ant1_{row_id}"]
        ant2 = st.session_state[f"mem_ant2_{row_id}"]
        textColumn.text(f"Pair: {ant1} - {ant2}")

    # Icon of minimize button dependent on state
    minIcon = ":heavy_minus_sign:" #"🗕" if st.session_state[f"min_{row_id}"] else "🗖"
    delIcon = ":x:" #✖

    # Minimze and Delete buttons
    minCol, delCol = buttonColumns.columns(2)
    minCol.button(minIcon, key=f"minbtn_{row_id}", on_click=toggle_row_min_button, args=[row_id])
    delCol.button(delIcon, key=f"del_{row_id}", on_click=remove_row, args=[row_id])
    
    def1, def2 = st.session_state[f"def1_{row_id}"], st.session_state[f"def2_{row_id}"]

    # Form
    if st.session_state[f"min_{row_id}"]:
        # Container
        formContainer = mainContainer.container()
        antCol, meaningCol = formContainer.columns(2)

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
        def1 = meaningCol.selectbox("Definition", definitions1, index=st.session_state[f"def1_index_{row_id}"], key=f"select1_{row_id}")
        def2 = meaningCol.selectbox("Definition", definitions2, index=st.session_state[f"def2_index_{row_id}"], key=f"select2_{row_id}", label_visibility="hidden")

        # Preserve selected defintion values when minimizing entry
        if st.session_state[f"min_{row_id}"] and def1:
            st.session_state[f"def1_{row_id}"] = def1
            def1_index = definitions1.index(def1)
            st.session_state[f"def1_index_{row_id}"] = def1_index
            
        if st.session_state[f"min_{row_id}"] and def2:
            st.session_state[f"def2_{row_id}"] = def2
            def2_index = definitions2.index(def2)
            st.session_state[f"def2_index_{row_id}"] = def2_index

    # Add antonym pair
    if ant1 or ant2:
        antonym_pair = [ant1, ant2]

    # Add definitions of the antonym pair
    if def1 or def2:
        definition_pair = [def1, def2]


    # Safe antonym pair and definition data to session state
    st.session_state["antonyms"][row_id] = antonym_pair
    st.session_state["definitions"][row_id] = definition_pair
        

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


# Adds example to session state with unqiue ID
def add_example():
    example_id = str(uuid.uuid4())
    st.session_state["rows_examples"].append(example_id)
    st.session_state[f"word_{example_id}"] = ""
    st.session_state[f"mem_word_{example_id}"] = ""
    st.session_state[f"min_{example_id}"] = True
    st.session_state[f"context_{example_id}"] = ""
    st.session_state[f"mem_context_{example_id}"] = ""

# Remove example from session state for a give ID
def remove_example(example_id):
    st.session_state["rows_examples"].remove(example_id)
    del st.session_state["examples"][example_id]
    for entry in st.session_state:
        if example_id in entry:
            del st.session_state[entry]

def toggle_example_min_button(example_id):
    st.session_state[f"min_{example_id}"] = not st.session_state[f"min_{example_id}"]

    if not st.session_state[f"min_{example_id}"]:
        st.session_state[f"mem_word_{example_id}"] = st.session_state[f"word_{example_id}"]
        st.session_state[f"mem_context_{example_id}"] = st.session_state[f"context_{example_id}"]
    else:
        st.session_state[f"word_{example_id}"] = st.session_state[f"mem_word_{example_id}"]
        st.session_state[f"context_{example_id}"] = st.session_state[f"mem_context_{example_id}"]

# Generates streamlit elements with given row ID
def generate_example(example_id):
    mainContainer = st.container()
    # header
    headerContainer = mainContainer.container()
    textColumn, buttonColumns = headerContainer.columns([4.7, 1])

    # Preserve word and context values when minimzing entry
    if st.session_state[f"min_{example_id}"]:
        word = st.session_state[f"word_{example_id}"]
        context = st.session_state[f"context_{example_id}"]
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

    st.session_state["examples"][example_id] = [word, context]

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
    return BERTWordEmbeddings()

@st.cache_data
def create_sense_polar(antonyms, examples, method):
    antonyms = list(antonyms.values())
    examples = list(examples.values())

    out_path = "./antonyms/"
    antonym_path = out_path + "polar_dimensions.pkl"

    dictionary = Dictionary("wordnet") 
    lookupSpace = LookupCreator(dictionary, out_path, antonym_pairs=antonyms)
    lookupSpace.create_lookup_files()

    model = load_bert_model()

    pdc = PolarDimensions(model, antonym_path=out_path + "antonym_wordnet_example_sentences_readable_extended.txt")
    pdc.create_polar_dimensions(out_path)

    wp = WordPolarity(model, antonym_path=antonym_path, lookup_path=out_path, method=method, number_polar=len(antonyms))

    words = []
    polar_dimensions = []
    for word, context in examples:
        words.append(word)
        polar_dimensions.append(wp.analyze_word(word, context))

    return words, polar_dimensions

@st.cache_data
def create_visualisations(options, words, polar_dimensions):
    plotter = PolarityPlotter()

    if "Standard" in options:
        fig = plotter.plot_word_polarity(words, polar_dimensions)
        st.plotly_chart(fig, use_container_width=True)

    if "2d" in options:
        fig = plotter.plot_word_polarity_2d(words, polar_dimensions)
        st.plotly_chart(fig, use_container_width=True)

    if "Polar" in options:
        fig = plotter.plot_word_polarity_polar_fig(words, polar_dimensions)
        st.plotly_chart(fig, use_container_width=True)

    if "Most descriptive" in options:
        fig = plotter.plot_descriptive_antonym_pairs(words, polar_dimensions, words, 3)
        st.plotly_chart(fig, use_container_width=True)

with st.sidebar:
    # Select method (projection or base-change)
    method = st.selectbox("Please select a transformation method for the antonym space", ["base-change", "projection"])
    if len(st.session_state["antonyms"]) < 2:
        given_options = ["Standard", "Polar", "Most descriptive"]
    else:
        given_options = ["Standard", "2d", "Polar", "Most descriptive"]                 
                           
    # Multiselect to select plots that will be displayed
    selected_options = st.multiselect("Please select some of the given visualization options", given_options)

# Create two columns for download and execute button
downloadCol, executeCol, _ = st.columns([0.2, 0.3, 0.8])

@st.cache_data
def to_excel(df):
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
def convert_df(antonyms, definitions, examples):
    antonyms = list(antonyms.values())
    definitions = list(definitions.values())
    examples = list(examples.values())

    data = {
        "antonym1": [""],
        "antonym2": [""],
        "definition_antonym1": [""],
        "definition_antonym2": [""],
        "word": [""],
        "context": [""]
    }   

    if antonyms:
            data["antonym1"] = [antonym_pair[0] if len(antonym_pair) > 0 else "" for antonym_pair in antonyms]
            data["antonym2"] = [antonym_pair[1] if len(antonym_pair) > 0 else "" for antonym_pair in antonyms]

    if definitions:
            data["definition_antonym1"] = [definition_pair[0] if len(definition_pair) > 0 else "" for definition_pair in definitions]
            data["definition_antonym2"] = [definition_pair[1] if len(definition_pair) > 0 else "" for definition_pair in definitions]

    if examples:
            data["word"]= [example[0] if len(example) > 0 else "" for example in examples]
            data["context"] = [example[1] if len(example) > 0 else "" for example in examples]
    
    st.write(data)

    df = to_excel(pd.DataFrame.from_dict(data, orient="index").transpose())

    return df

df = convert_df(st.session_state["antonyms"], st.session_state["definitions"], st.session_state["examples"])

download_button = downloadCol.download_button(label="Download", data=df, file_name="SensePolar.xlsx")

if executeCol.button("Execute"):
    words, polar_dimensions = create_sense_polar(st.session_state["antonyms"], st.session_state["examples"], method)
    create_visualisations(selected_options, words, polar_dimensions)