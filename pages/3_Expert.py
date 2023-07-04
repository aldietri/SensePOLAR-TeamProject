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


#Page Configuration
st.set_page_config(layout="centered", page_title="Expert", page_icon="🏆")
#Headline
st.write("# Welcome to the Expert Page")

#Subheadline with short explaination
st.markdown(
    """
    This page can be used to manually upload Sense Dimensions for a faster comparison of Word Meanings.
    Below you see the tables strucute you need to follow. You can also download an exmaple datafile.
    """
)
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


@st.cache_resource
def load_bert_model():
    return BERTWordEmbeddings()


import plotly.graph_objects as go
import numpy as np
from collections import defaultdict

plotter = PolarityPlotter() 

def create_sense_polar(data_frame, word_collections):
    out_path = './antonyms/'
    antonym_path = out_path + "polar_dimensions.pkl"
    
    lookupSpace = LookupCreator()
    antonyms, definitions, examples = lookupSpace.retrieve_from_file(data_frame)
    lookupSpace.antonym_pairs,lookupSpace.definitions,lookupSpace.examples = antonyms, definitions, examples
    lookupSpace.create_lookup_files()

    model = load_bert_model()

    pdc = PolarDimensions(model, antonym_path=out_path + "antonym_wordnet_example_sentences_readable_extended.txt")
    pdc.create_polar_dimensions(out_path)
    wp = WordPolarity(model, antonym_path=antonym_path, method='projection', number_polar=data_frame.shape[0]-1)

    words, contexts, polarity_base_change = [],[],[]
    for i in range(len(word_collections)):
        word = word_collections[i][0] # first item = word
        context = word_collections[i][1] #second item = context
        
        words.append(word) 
        contexts.append(context) 

        polarity_base_change.append(wp.analyze_word(word, context))

    
    st.plotly_chart(plotter.plot_word_polarity(words, polarity_base_change), use_container_width=True)
    #st.write("Word polarity using base change method: ", polarity_base_change)


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


if st.button("Execute"):
    words = create_sense_polar(edited_df,word_collection)






