import numpy as np
import nltk
from nltk.corpus import wordnet as wn
import string
import json
import pickle
import pandas as pd
from collections import defaultdict
import re
import itertools
from sensepolar.oracle.examples import ExampleGenerator
from nltk.stem import PorterStemmer

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

    def __init__(self, dictionary, out_path="./antonyms/", antonym_pairs=None, antonyms_file_path=None, generate_examples=False, num_examples=5):
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
        self.generate_examples = generate_examples
        self.num_examples = num_examples
        self.stemmer = PorterStemmer()
        self.example_generator = ExampleGenerator()
        self.example_cache = {} 

        

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

    def get_examples(self, antonym, index=0):
        """
        Return example sentences for a synset.

        Parameters:
            antonym (str): the synset to get example sentences for

        Returns:
            list: a list of example sentences
        """
        if antonym in self.example_cache:
            return self.example_cache[antonym]
        examples = self.dictionary.get_examples(antonym)[index]
        if type(examples) != list:
            examples = [examples]
        definition = self.dictionary.get_definitions(antonym)[index]
        if self.generate_examples:
                examples.extend(list(self.example_generator.generate_examples(antonym, definition, self.num_examples)))
        examples = [sent.translate(str.maketrans({k: " " for k in string.punctuation})) for sent in examples]
        examples = [' '.join(re.sub(r"<[^>]+>", "", example).split()) for example in examples]
        stemmer = PorterStemmer()
        replaced_examples = []
        for example in examples:
            words = example.split()
            replaced_words = [antonym if stemmer.stem(w) == stemmer.stem(antonym) else w for w in words]
            replaced_example = ' '.join(replaced_words)
            replaced_examples.append(replaced_example)
        examples = replaced_examples
        correct_examples=[]
        print('Examples', examples)
        for example in examples:
            if re.search(r'\b'+ str(antonym).lower()+'\\b', example.lower(), re.I) is not None:
                correct_examples.append(" ".join(example.split()).lower())
        self.example_cache[antonym] = ['{} '.format(sent) for sent in correct_examples]
        return self.example_cache[antonym]

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
        examples = [' '.join(re.sub(r"<.*?>", "", example).split()) for example in examples]
        # add a space after each sentence
        return ['{} '.format(sent) for sent in examples]

    def create_lookup_files(self, indices=None):
        """Create and store the lookup files."""
        if indices is None:
            indices = [[0,0] for i in range(len(self.antonym_pairs))]
        if self.examples is None:
            antonyms = self.antonym_pairs
            antonyms = [pair for pair in self.antonym_pairs if min(len(self.get_examples(pair[0])),
                                                    len(self.get_examples(pair[1]))) != 0]
        else:
            antonyms = [pair for pair in self.antonym_pairs if min(len(self.get_examples_files(pair[0], self.examples)),
                                                    len(self.get_examples_files(pair[1], self.examples))) != 0]
        if self.definitions is None:
            synset_defs = [[self.dictionary.get_definitions(anto)[indices[i][j]] for j, anto in enumerate(pair)] for i, pair in enumerate(antonyms)]
            self.definitions = synset_defs
        else:
            synset_defs = [[self.definitions[anto] for anto in pair] for pair in antonyms]
        if self.examples is None:
            self.examples = []
            for i, pair in enumerate(antonyms):
                pair_examples = []
                for j, anto in enumerate(pair):
                    pair_examples.append(self.get_examples(anto, index=indices[i][j]))
                self.examples.append(pair_examples)
            examples_readable = {str(pair):{anto: self.examples[i][j] for j, anto in enumerate(pair)} for i, pair in enumerate(antonyms)}
            examples_lookup = [[[anto, self.examples[i][j]] for j, anto in enumerate(pair)] for i, pair in enumerate(antonyms)]
        else:
            examples_readable = {str(pair):{anto: self.get_examples_files(anto, self.examples) for anto in pair} for pair in antonyms}
            examples_lookup = [[[anto, self.get_examples_files(anto, self.examples)] for anto in pair] for pair in antonyms]
        # save 
        with open(self.out_path + 'lookup_synset_dict.txt', 'w') as t:
            t.write(json.dumps(antonyms, indent=4))
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