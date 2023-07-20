import streamlit as st
from nltk.corpus import wordnet as wn
import uuid

from lookup import LookupCreator
from sensepolar.polarity import WordPolarity
from sensepolar.bertEmbed import BERTWordEmbeddings
from sensepolar.polarDim import PolarDimensions
from sensepolar.dictionaryapi import Dictionary
from sensepolar.plotter import PolarityPlotter

import plotly.graph_objects as go
import plotly.express as px
import random
import numpy as np
from collections import defaultdict

st.set_page_config(layout="centered", page_title="SensePOLAR", page_icon="🌊")

# Removes menu and footer
# hide_streamlit_style = """
# <style>
# #MainMenu {visibility: hidden;}
# footer {visibility: hidden;}
# </style>
# """
# st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

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
if "rows" not in st.session_state:
    st.session_state["rows"] = []

# collect row data (antonym pairs)
antonym_collection = []

# Adds row to session state with unqiue ID
def add_row():
    element_id = uuid.uuid4()
    st.session_state["rows"].append(str(element_id))

# Remove row from session state for a give ID
def remove_row(row_id):
    st.session_state["rows"].remove(f"{row_id}")

# Generates streamlit elements with given row ID
def generate_row(row_id):
    # Create delete button antonym pairs
    st.button("Delete item", key=f"del_{row_id}", on_click=remove_row, args=[row_id])
    # Container with two columns
    row_container = st.empty()
    row_columns = row_container.columns(2)
    
    # List containing antonym data
    antonym_pair = []

    for i in range(2):
        # Create text input field for item (word)
        row_word = row_columns[0].text_input("Item name", key=f"txt_{row_id}_{i}")

        # Check if text field contains word
        if row_word:
            # Fetch Synsets of given word
            word_synsets = wn.synsets(row_word)
            # Sets the amount of defintions that will be displayed - 5 or less 
            i_range = 5 if len(word_synsets) > 5 else len(word_synsets)
            # Check if any definitions at all (aka if given word is a proper word)
            if i_range == 0:
                # Print Error
                row_columns[1].write("🚨 Attention: Invalid input detected! 🚨")
                row_columns[1].write("No definitions were found. Please select a new item.")
            else:
                # Fetch definitions
                definitions = [word_synsets[i].definition() for i in range(i_range)]
                # Create selectbox containing definitions
                def_index = row_columns[1].selectbox("Select one of the following definitions:", options=range(len(definitions)), key=f"sb_{row_id}_{i}", format_func=lambda x: definitions[x])
                # Fill antonym pair data (word defintion in word net format)    
                antonym_pair.append(word_synsets[def_index].name())

    # Return antonym pair data
    return antonym_pair

# Necessary to add and delete rows
# Recreates rows for every row contained in the session state
for idx, row in enumerate(st.session_state["rows"], start=1):
    # Create subheader for every antonym pair
    st.subheader(f"Antonym Pair {idx}")   
    # Generate elements for antonym pairs
    row_data = generate_row(row)
    # Save antonym pair data to overall collection when antonym pair fully declared (aka both antonyms)
    if len(row_data) == 2:
        antonym_collection.append(row_data)

# Create button to add row to session state with unqiue ID
st.button("Add Item", on_click=add_row)

# EXAMPLE LOGIC

# TODO
# Creates entry in session state for example input
if "examples" not in st.session_state:
    st.session_state["examples"] = []

# TODO:
example_collection = []

# TODO
def add_example():
    element_id = uuid.uuid4()
    st.session_state["examples"].append(str(element_id))

# TODO
def remove_example(example_id):
    st.session_state["examples"].remove(f"{example_id}")

# TODO
def generate_example(example_id):

    st.button("Delete item", key=f"del_{example_id}", on_click=remove_example, args=[example_id])

    example_container = st.empty()
    example_columns = example_container.columns(2)

    example = example_columns[0].text_input("Word", key=f"word_{example_id}").strip()
    word_example = example_columns[1].text_input("Example", key=f"example_{example_id}").strip()

    if example and word_example:
        return [example, word_example]
    return []

# TODO
for idx, example in enumerate(st.session_state["examples"], start=1):
    st.subheader(f"Word {idx}")
    example_data = generate_example(example)
    if len(example_data) == 2:
        example_collection.append(example_data)

# TODO
# Create button to add row to session state with unqiue ID
st.button("Add Example", on_click=add_example)

# Debugging/Visualization
st.write(st.session_state)
# st.write(antonym_collection)
# st.write(example_collection)

# SensePOLAR Logic

@st.cache_resource
def load_bert_model():
    return BERTWordEmbeddings()

@st.cache_data
def create_sense_polar(antonyms, examples, dict_name, api_key):
    out_path = "./antonyms/"
    antonym_path = out_path + "polar_dimensions.pkl"

    # TODO in Dictionairy API
    antonyms = [[a.split(".")[0], b.split(".")[0]] for a,b in antonyms]

    dictionary = Dictionary(dict_name, api_key) 
    # dictionary = Dictionary('dictionaryapi', api_key='b4b51989-1b9d-4690-8975-4a83df13efc4 ')
    lookupSpace = LookupCreator(dictionary, out_path, antonym_pairs=antonyms)
    lookupSpace.create_lookup_files()

    model = load_bert_model()

    pdc = PolarDimensions(model, antonym_path=out_path + "antonym_wordnet_example_sentences_readable_extended.txt")
    pdc.create_polar_dimensions(out_path)
    
    wp = WordPolarity(model, antonym_path=antonym_path, lookup_path=out_path, method='projection', number_polar=len(antonyms))
    
    #TODO
    words = []
    polar_dimensions = []
    for word, context in examples:
        words.append(word)
        polar_dimensions.append(wp.analyze_word(word, context))

    return words,  polar_dimensions

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

dict_name = st.selectbox("Please select your preferred dictionary",
                        ["wordnet", "dictionaryapi", "oxford", "wordnik"])

api_key = ""
if dict_name != "wordnet":
    api_key = st.text_input("Please insert your API KEY")

options = st.multiselect("Please select some of the given visualization options",                      
                         ["Standard", "2d", "Polar", "Most descriptive"])

if st.button("Execute"):
    words, polar_dimensions = create_sense_polar(antonym_collection, example_collection, dict_name, api_key)
    create_visualisations(options, words, polar_dimensions)