import numpy as np
import nltk
from nltk.corpus import wordnet as wn
import string
import json
import pickle
import pandas as pd
from collections import defaultdict
import re

class LookupCreator:
    """
    Class for creating and storing lookup files for antonym pairs.

    Attributes:
    ----------
    antonym_pairs: list
        A list of antonym pairs.
    out_path: str
        The directory path to store the lookup files.
    """

    def __init__(self, dictionary, out_path="./antonyms/", antonym_pairs=None, antonyms_file_path=None):
        """
        Initialize the LookupCreator.

        Parameters:
        ----------
        antonym_pairs: list
            A list of antonym pairs.
        out_path: str
            The directory path to store the lookup files.
        """
        self.antonym_pairs = antonym_pairs
        self.definitions = None
        self.examples = None
        if antonyms_file_path is not None:
            self.antonym_pairs, self.definitions, self.examples = self.retrieve_from_file(antonyms_file_path)
            self.out_path = out_path
        self.dictionary = dictionary

    def get_name(self, antonym):
        """
        Return the name of a synset.

        Parameters:
        ----------
        antonym: str
            The synset to get the name of.

        Returns:
        ----------
        str
            The name of the synset.
        """
        return wn.synset(antonym).lemma_names()[0]

    def get_examples(self, antonym):
        """
        Return example sentences for a synset.

        Parameters:
            antonym (str): the synset to get example sentences for

        Returns:
            list: a list of example sentences
        """
        examples = self.dictionary.get_examples(antonym)
        # replace punctuation symbols with spaces
        examples = [sent.translate(str.maketrans({k: " " for k in string.punctuation})) for sent in examples]
        # add a space after each sentence
        return ['{} '.format(sent) for sent in examples]

    def retrieve_from_file(self, file_path):
        """
        Retrieve antonym pairs from a file.

        Parameters:
        ----------
        file_path: str
            The path to the file containing the antonym pairs.

        Returns:
        ----------
        list
            A list of antonym pairs.
        """
        data = pd.read_excel(file_path, header=0)
        antonyms = []
        definitions = defaultdict()
        examples = defaultdict()

        for index, row in data.iterrows():
            antonym_1 = row['antonym_1']
            antonym_2 = row['antonym_2']
            example_antonym_1 = row['example_antonym_1']
            example_antonym_2 = row['example_antonym_2']
            def1 = row['def1']
            def2 = row['def2']

            antonyms.append([antonym_1, antonym_2])
            definitions[antonym_1] = def1
            definitions[antonym_2] = def2
            examples[antonym_1] = example_antonym_1
            examples[antonym_2] = example_antonym_2
        return antonyms, definitions, examples

    def get_examples_files(self, antonym, dictionary):
        """
        Return example sentences for a synset from file.

        """
        examples=dictionary[antonym].split(".")
        #save only examples that containt the required word
        correct_examples=[]
        for example in examples:
            if re.search(r'\b'+ str(antonym).lower()+'\\b', example.lower(), re.I) is not None:
                correct_examples.append(" ".join(example.split()).lower())
        # replace punctuation symbols with spaces
        examples = [sent.translate(str.maketrans({k: " " for k in string.punctuation})) for sent in correct_examples]
        # add a space after each sentence
        return ['{} '.format(sent) for sent in examples]

    def create_lookup_files(self):
        """Create and store the lookup files."""
        antonyms = np.unique(self.antonym_pairs, axis=0)
        if len(antonyms) != len(self.antonym_pairs):
            print("Your antonym list contains duplicates. Please try again!")
            return

        if self.definitions is None:
            synset_defs = [[self.dictionary.get_definitions(anto) for anto in pair] for pair in antonyms]
        else:
            synset_defs = [[self.definitions[anto] for anto in pair] for pair in antonyms]
        if self.examples is None:
            examples_readable = {str(pair):{anto: self.get_examples(anto) for anto in pair} for pair in antonyms}
            examples_lookup = [[[anto, self.get_examples(anto)] for anto in pair] for pair in antonyms]
        else:
            examples_readable = {str(pair):{str(anto): self.get_examples_files(anto, self.examples) for anto in pair} for pair in antonyms}
            examples_lookup = [[[str(anto), self.get_examples_files(anto, self.examples)] for anto in pair] for pair in antonyms]
        # save 
        with open(self.out_path + 'lookup_synset_dict.txt', 'w') as t:
            t.write(json.dumps(antonyms.tolist(), indent=4))
        with open(self.out_path + 'lookup_synset_dict.pkl', 'wb') as p:
            pickle.dump(antonyms, p)
        with open(self.out_path + 'lookup_synset_definition.txt', 'w') as t:
            t.write(json.dumps(synset_defs, indent=4))  
        with open(self.out_path + 'lookup_synset_definition.pkl', 'wb') as p:
            pickle.dump(synset_defs, p)        
        with open(self.out_path + 'antonym_wordnet_example_sentences_readable_extended.txt', 'w') as t:
            t.write(json.dumps(examples_readable, indent=4))  
        with open(self.out_path + 'lookup_anto_example_dict.txt', 'w') as t:
            t.write(json.dumps(examples_lookup, indent=4))      
        with open(self.out_path + 'lookup_anto_example_dict.pkl', 'wb') as p:
            pickle.dump(examples_lookup, p)