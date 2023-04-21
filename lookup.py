import numpy as np
import nltk
from nltk.corpus import wordnet as wn
import string
import json
import pickle
from polarDim import PolarDimensions
from polarity import WordPolarity
from bertEmbed import BERTWordEmbeddings


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

    def __init__(self, antonym_pairs, out_path):
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
        self.out_path = out_path

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
        examples = wn.synset(antonym).examples()
        # replace punctuation symbols with spaces
        examples = [sent.translate(str.maketrans({k: " " for k in string.punctuation})) for sent in examples]
        # add a space after each sentence
        return ['{} '.format(sent) for sent in examples]

    def create_lookup_files(self):
        """Create and store the lookup files."""
        antonyms = np.unique(self.antonym_pairs, axis=0)
        if len(antonyms) != len(self.antonym_pairs):
            print("Your antonym list contains duplicates. Please try again!")
            return

        # get all word sense definitions
        synset_defs = [[wn.synset(anto).definition() for anto in pair] for pair in antonyms]
        # get example sentences from wordnet
        examples_readable = {str(pair):{self.get_name(anto): self.get_examples(anto) for anto in pair} for pair in antonyms}
        examples_lookup = [[[self.get_name(anto), self.get_examples(anto)] for anto in pair] for pair in antonyms]

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