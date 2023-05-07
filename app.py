import streamlit as st
from nltk.corpus import wordnet as wn
import uuid

from lookup import LookupCreator
from sensepolar.polarity import WordPolarity
from sensepolar.bertEmbed import BERTWordEmbeddings
from sensepolar.polarDim import PolarDimensions

import plotly.graph_objects as go
import plotly.express as px
import random
import numpy as np
from collections import defaultdict


st.set_page_config(layout="wide", page_title="SensePOLAR", page_icon="🌊")

# Removes menu and footer
# hide_streamlit_style = """
# <style>
# #MainMenu {visibility: hidden;}
# footer {visibility: hidden;}
# </style>
# """
# st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

st.write("# Welcome to SensePOLAR! 🌊")

st.markdown(
    """
    SensePOLAR is the first (semi-) supervised framework for augmenting interpretability into contextual word embeddings (BERT).
    Interpretability is added by rating words on scales that encode user-selected senses, like correctness or left-right-direction.
    """
)

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

# Debugging/Visualization
# st.write(st.session_state)
st.write(antonym_collection)

@st.cache_resource
def load_bert_model():
    return BERTWordEmbeddings()


def plot_word_polarity(words, polar_dimension):
    # create dictionary with antonyms as key and (word,polar) as value
    antonym_dict = defaultdict(list)
    for w_i in range(len(words)):
        for antonym1, antonym2, value in polar_dimension[w_i]:
            antonym_dict[(antonym1, antonym2)].append((words[w_i], value))
    fig = go.Figure()

    # Add axes
    fig.add_shape(type="line", x0=-1, y0=0, x1=1, y1=0, line=dict(color='black', width=1))
    fig.add_shape(type="line", x0=-1, y0=0, x1=-1, y1=len(antonym_dict)/10, line=dict(color='black', width=1))
    fig.add_shape(type="line", x0=1, y0=0, x1=1, y1=len(antonym_dict)/10, line=dict(color='black', width=1))
    fig.add_shape(type="line", x0=0, y0=0, x1=0, y1=len(antonym_dict)/10, line=dict(color='black', width=1, dash='dash'))

    # Define color scale for the words
    colors = np.linspace(0, 1, len(words))

    # Add lines and markers for each word's polarity score
    counter = 0.01
    offset = 0.005
    for i, (antonyms, polars) in enumerate(antonym_dict.items()):
        show_legend = True if i == 0 else False
        for polar in polars:
            # Define color of the line and marker based on word's position in the list
            color = f'rgb({int(colors[words.index(polar[0])] * 255)}, {int((1-colors[words.index(polar[0])]) * 255)}, 0)'
            fig.add_shape(type="line", x0=polar[1], y0=counter, x1=0, y1=counter, 
                          line=dict(color=color, width=1))
            fig.add_trace(go.Scatter(x=[polar[1]], y=[counter], mode='markers', 
                                     marker=dict(color=color, symbol='square', size=10),
                          name=polar[0], showlegend=show_legend))
            # fig.add_annotation(x=polar[1], y=counter, text=polar[0], font=dict(size=20), showarrow=True, xanchor='auto')
        fig.add_annotation(x=-1.1, y=counter, text=antonyms[0], font=dict(size=18), showarrow=False, xanchor='right')
        fig.add_annotation(x=1.1, y=counter, text=antonyms[1], font=dict(size=18), showarrow=False, xanchor='left')
        counter += offset + 0.1
    # Set x and y axis titles
    fig.update_layout(
        xaxis_title=f"Polarity",
        yaxis_title="Words",
        xaxis_range=[-1, 1],
        xaxis_autorange=True, yaxis_autorange=True
    )

    return fig

def plot_polar(word, polar_dimension):
    antonyms = []
    values = []
    for antonym1, antonym2, value in polar_dimension:
        values.append(abs(value))
        if value > 0:
            antonyms.append(antonym1)
        else:
            antonyms.append(antonym2)
    
    antonyms.append(antonyms[0])
    values.append(values[0])

    color = random.choice(px.colors.qualitative.Pastel)
    while color in st.session_state["colors"]:
        color = random.choice(px.colors.qualitative.Pastel)
    
    st.session_state["colors"].append(color)

    fig = go.Figure(
        go.Scatterpolar(
        name = word,
        r=values,
        theta=antonyms,
        fill="toself",
        line_color=color
        )   
    )

    fig.update_layout(
        height = 1000,
        polar=dict(
        radialaxis_angle = 45,
        angularaxis = dict(
            direction ="clockwise",
            period= len(antonyms)-1
            )
        )
    )

    return fig


if len(antonym_collection) >= 4:
    out_path = "./antonyms/"
    antonym_path = out_path + "polar_dimensions.pkl"

    lookupSpace = LookupCreator(antonym_pairs=antonym_collection, out_path=out_path)
    lookupSpace.create_lookup_files()

    model = load_bert_model()

    pdc = PolarDimensions(model, antonym_path=out_path + "antonym_wordnet_example_sentences_readable_extended.txt")
    pdc.create_polar_dimensions(out_path)

    wp = WordPolarity(model, antonym_path, method="base-change", number_polar=len(antonym_collection))

    word = "sun"
    context = "The sun is shining today."
    polarity_base_change = wp.analyze_word(word, context)
    
    word1 = "fire"
    context1 = "the fire is burning"
    polarity_base_change1 = wp.analyze_word(word1, context1)

    st.plotly_chart(plot_word_polarity([word, word1],[polarity_base_change, polarity_base_change1]), use_container_width=True)

    if "colors" not in st.session_state:
        st.session_state["colors"] = []

    col1, col2 = st.columns(2)
    col1.markdown(f"<h1 style='text-align: center;'>{word}</h1>", unsafe_allow_html=True)
    col1.plotly_chart(plot_polar(word, polarity_base_change), use_container_width=True)

    col2.markdown(f"<h1 style='text-align: center;'>{word1}</h1>", unsafe_allow_html=True)
    col2.plotly_chart(plot_polar(word1, polarity_base_change1), use_container_width=True)

