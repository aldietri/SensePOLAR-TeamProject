#import Files and Functions for Sense Polar
from lookup import LookupCreator
from sensepolar.polarity import WordPolarity
from sensepolar.bertEmbed import BERTWordEmbeddings
from sensepolar.polarDim import PolarDimensions
from sensepolar.dictionaryapi import Dictionary
from sensepolar.plotter import PolarityPlotter

#import streamlit functions
import streamlit as st
from nltk.corpus import wordnet as wn
import uuid
import pandas as pd
from io import StringIO

#import other functions
import plotly.graph_objects as go
import plotly.express as px
import random
import numpy as np
from collections import defaultdict
import openpyxl
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb


import plotly.graph_objects as go
import numpy as np
from collections import defaultdict


#Page Configuration
st.set_page_config(layout="centered", page_title="Expert", page_icon="🏆")
#Headline
st.write("# Welcome to the Expert Page")




import plotly.graph_objects as go
import numpy as np
from collections import defaultdict

#Subheadline with short explaination
st.markdown(
    """
    This page can be used to manually upload Sense Dimensions for a faster comparison of Word Meanings.
    Below you see the tables strucute you need to follow. You can also download an exmaple datafile.
    """
)

# Creates entry in session state for antonym pairs
if "antonyms" not in st.session_state:
    st.session_state["antonyms"] = {}

# Display how the Dataframe has to look
empty_df = pd.DataFrame(
    columns=('antonym_1','antonym_2','example_antonym_1','example_antonym_2','def1','def2'))
st.table(empty_df)

#Create Download Button for excel File
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet.set_column('A:A', None, format1)  
    writer.save()
    processed_data = output.getvalue()
    return processed_data
df = pd.read_excel('excel_files_for_experts/data_file.xlsx')
df_xlsx = to_excel(df)
st.download_button(label='Download Example Sheet',
                                data=df_xlsx ,
                                file_name= 'df_test.xlsx')


#Create Upload Button for Antonym words
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # Can be used wherever a "file-like" object is accepted:
    if ".xlsx" in uploaded_file.name:
        dataframe = pd.read_excel(uploaded_file,header=0)
    else:
        dataframe = pd.read_csv(uploaded_file)
    edited_df = st.experimental_data_editor(dataframe, num_rows="dynamic",use_container_width=True)
else:
    empty_df = pd.read_csv("excel_files_for_experts/Empty_Dataframe.csv")
    edited_df = st.experimental_data_editor(empty_df, num_rows="dynamic",use_container_width=True,)

if "df_value" not in st.session_state:
    st.session_state.df_value = empty_df

if edited_df is not None and not edited_df.equals(st.session_state["df_value"]):
    st.session_state["df_value"] = edited_df

#sidebar slection 
with st.sidebar:
    # Select method (projection or base-change)
    method = st.selectbox("Please select a transformation method for the antonym space", ["base-change", "projection"])
    if len(st.session_state["df_value"]) < 2:
        given_options = ["Standard", "Polar", "Most descriptive"]
    else:
        given_options = ["Standard", "2d", "Polar", "Most descriptive"]                 
                           
    # Multiselect to select plots that will be displayed
    selected_options = st.multiselect("Please select some of the given visualization options", given_options)


@st.cache_resource
def load_bert_model():
    return BERTWordEmbeddings()



plotter = PolarityPlotter() 

def create_sense_polar(data_frame, word_collections, method):
    out_path = './antonyms/'
    antonym_path = out_path + "polar_dimensions.pkl"
    
    lookupSpace = LookupCreator()
    antonyms, definitions, examples = lookupSpace.retrieve_from_file(data_frame)
    lookupSpace.antonym_pairs,lookupSpace.definitions,lookupSpace.examples = antonyms, definitions, examples
    lookupSpace.create_lookup_files()

    model = load_bert_model()

    pdc = PolarDimensions(model, antonym_path=out_path + "antonym_wordnet_example_sentences_readable_extended.txt")
    pdc.create_polar_dimensions(out_path)
    wp = WordPolarity(model, antonym_path=antonym_path, method=method, number_polar=len(data_frame))

    words, contexts, polar_dimensions = [],[],[]
    for i in range(len(word_collections)):
        word = word_collections[i][0] # first item = word
        context = word_collections[i][1] #second item = context
        
        words.append(word) 
        contexts.append(context) 

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
    
    
    #st.plotly_chart(plotter.plot_word_polarity(words, polar_dimensions), use_container_width=True)
    #st.write("Word polarity using base change method: ", polarity_base_change)

#download Button for edited Dataframe - is is safed as a Excel file
df_edited_xls = to_excel(edited_df)
st.download_button(label='Download Sheet',
                                data=df_edited_xls ,
                                file_name= 'df_edited.xlsx')

# collect row data (words to analyze)
word_collection = []

# Creates entry in session state for rows
if "rows" not in st.session_state:
    st.session_state["rows"] = []

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
    
    # List containing word data
    word_pair = []
    for i in range(1):
        # Create text input field for item (word)
        row_word = row_columns[0].text_input("Item name", key=f"txt_word_{row_id}_{i}")

        # Check if text field contains word
        if row_word:
            def_index = row_columns[1].text_input("Context", key=f"txt_context_{row_id}_{i}")
            
            word_pair = [row_word,def_index]
    
    # Return antonym pair data
    return word_pair


# Necessary to add and delete rows
# Recreates rows for every row contained in the session state
for idx, row in enumerate(st.session_state["rows"], start=1):
    # Create subheader for every antonym pair
    st.subheader(f"Word to analyze {idx}")   
    # Generate elements for antonym pairs
    row_data = generate_row(row)
    # Save antonym pair data to overall collection when antonym pair fully declared (aka both antonyms)
    if len(row_data) == 2:
        word_collection.append(row_data)

# Create button to add row to session state with unqiue ID
st.button("Add Item", on_click=add_row)

#execute Buttion
if st.button("Execute"):
    words, polar_dimensions = create_sense_polar(st.session_state["df_value"], word_collection, method)
    create_visualisations(selected_options, words, polar_dimensions)





