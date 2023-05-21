#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
from collections import defaultdict
from bertFuncs import analyzeWord, getBert
from createDims import createPolarDimension
import nltk
from nltk.corpus import wordnet as wn
import numpy as np
import pickle
import json
import string
import ast
import torch


# In[2]:

def text_lowercase(text):
    return text.lower()

def remove_whitespace(text):
    return  " ".join(text.split())

def get_examples_files(antonym,dictionary):
    
    examples=dictionary[antonym].split(".")
    #save only examples that containt the required word
    correct_examples=[]
    for example in examples:
        if re.search(r'\b'+text_lowercase(str(antonym))+'\\b', text_lowercase(example), re.I) is not None:
            correct_examples.append(text_lowercase(remove_whitespace(example)))
    

    
    examples = [sent.translate(str.maketrans({k: " " for k in string.punctuation})) for sent in correct_examples]
    # add a space after each sentence
    return ['{} '.format(sent) for sent in examples]
'''def get_examples_files(antonym,dictionary):
    
    examples=dictionary[antonym].split(".")
    #save only examples that containt the required word
    correct_examples=[]
    for example in examples:
        if re.search(r'\b'+str(antonym)+'\\b', example, re.I) is not None:
            correct_examples.append(example)
    

    
    examples = [sent.translate(str.maketrans({k: " " for k in string.punctuation})) for sent in correct_examples]
    # add a space after each sentence
    return ['{} '.format(sent) for sent in examples]'''


# In[17]:


def create_lookup_files_fromFile(antonyms, out_path,definitions,examples):
    if len(np.unique(antonyms, axis=0)) != len(antonyms):
        print("Your antonym list contains duplicates. Please try again!")
        return
    
    # get all word sense definitions
    synset_defs = [[definitions[anto] for anto in pair] for pair in antonyms]
    # get example sentences from wordnet
    examples_readable = {str(pair):{str(anto): get_examples_files(anto,examples) for anto in pair} for pair in antonyms}
    examples_lookup = [[[str(anto), get_examples_files(anto,examples)] for anto in pair] for pair in antonyms]
    
    # save 
    with open(out_path + 'lookup_synset_dict.txt', 'w') as t:
        t.write(json.dumps(antonyms, indent=4))
    with open(out_path + 'lookup_synset_dict.pkl', 'wb') as p:
        pickle.dump(antonyms, p)
    with open(out_path + 'lookup_synset_definition.txt', 'w') as t:
        t.write(json.dumps(synset_defs, indent=4))  
    with open(out_path + 'lookup_synset_definition.pkl', 'wb') as p:
        pickle.dump(synset_defs, p)        
    with open(out_path + 'antonym_wordnet_example_sentences_readable_extended.txt', 'w') as t:
        t.write(json.dumps(examples_readable, indent=4))  
    with open(out_path + 'lookup_anto_example_dict.txt', 'w') as t:
        t.write(json.dumps(examples_lookup, indent=4))      
    with open(out_path + 'lookup_anto_example_dict.pkl', 'wb') as p:
        pickle.dump(examples_lookup, p)
    return


# In[18]:


def create_lookup_from_data_file(file_name,out_path):
    file=r'{}'.format(file_name)
    data = pd.read_excel(file)
    dimensions=[]
    ant_example_dict=defaultdict()
    definition_dict=defaultdict()
    for index,value in enumerate(data.iloc[:,0]):
        dimensions.append(list([value,data.iloc[:,1][index]]))
        definition_dict[value]=data.iloc[:,4][index]
        definition_dict[data.iloc[:,1][index]]=data.iloc[:,5][index]
        ant_example_dict[value]=data.iloc[:,2][index]
        ant_example_dict[data.iloc[:,1][index]]=data.iloc[:,3][index]
    
    create_lookup_files_fromFile(dimensions,out_path,definition_dict,ant_example_dict)
    
    return 
    

