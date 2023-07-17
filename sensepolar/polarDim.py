from typing import List, Dict
import torch
import numpy as np
import json
import pickle
from sensepolar.embed.bertEmbed import BERTWordEmbeddings


class PolarDimensions:
    """
    A class for creating polar dimensions from antonyms and their example sentences.

    Attributes:
        bert (BERTWordEmbeddings): A BERTWordEmbeddings object that provides word embeddings.
        model (BertModel): A BertModel object from PyTorch, which is part of the BERT model architecture.
        tokenizer (BertTokenizer): A BertTokenizer object from PyTorch, which is used to tokenize text.
        antonym_path (str): The path to a JSON file that contains antonyms and their example sentences.

    Methods:
        load_antonyms_from_json(dict_path: str) -> Dict[str, List[str]]:
            Loads antonyms and their example sentences from a JSON file.
            Returns a dictionary with the antonyms as keys and a list of their corresponding example sentences as values.

        check_sentences(sentences: Dict[str, List[str]]) -> bool:
            Checks if each antonym is paired with exactly one list of example sentences.
            Returns True if each antonym has one corresponding list of example sentences, and False otherwise.

        get_average_embedding(name: str, sentences: List[str]) -> np.ndarray:
            Computes the average word embedding for a given word or phrase and a list of example sentences.
            Returns a numpy array with the average word embedding.

        create_polar_dimensions(out_path: str):
            Creates a list of direction vectors for each antonym in the antonym dictionary.
            Saves the list of direction vectors to a pickle file.
    """
    def __init__(self, model: BERTWordEmbeddings, antonym_path: str):
        """
        Initializes a PolarDimensions object.

        Args:
            model (BERTWordEmbeddings): A BERTWordEmbeddings object that provides word embeddings.
            antonym_path (str): The path to a JSON file that contains antonyms and their example sentences.
        """
        self.model = model
        self.antonym_path = antonym_path

    @staticmethod
    def load_antonyms_from_json(dict_path: str) -> Dict[str, List[str]]:
        """
        Loads antonyms and their example sentences from a JSON file.

        Args:
            dict_path (str): The path to a JSON file that contains antonyms and their example sentences.

        Returns:
            A dictionary with the antonyms as keys and a list of their corresponding example sentences as values.
        """
        with open(dict_path) as f:
            antonym_dict = json.load(f)
        return antonym_dict

    @staticmethod
    def check_sentences(sentences: Dict[str, List[str]]) -> bool:
        """
        Checks if each antonym is paired with exactly one list of example sentences.

        Args:
            sentences (Dict[str, List[str]]): A dictionary with the antonyms as keys and a list of their corresponding example sentences as values.

        Returns:
            True if each antonym has one corresponding list of example sentences, and False otherwise.
        """
        antonym_names = list(sentences.keys())
        if len(antonym_names) != 2:
            print("Each words needs to be paired with exactly one antonym.")
            print(antonym_names)
            return False

        for word, sent_list in sentences.items():
            if len(sent_list) < 1:
                print("Please enter at least one example sentence for the word: " + word)
                return False

            if not all(part in sent.split(" ") for part in word.split(" ") for sent in sent_list):
                print("This word is missing in a corresponding example sentence: " + word)
                return False
        return True
    
    def get_average_embedding(self, name: str, sentences: List[str]) -> np.ndarray:
        """
        Computes the average embedding of a word from its occurrences in a list of sentences.

        Args:
            name (str): The name of the word to compute the embedding for.
            sentences (List[str]): A list of sentences containing the word.

        Returns:
            np.ndarray: The average embedding of the word.

        Raises:
            None.
        """
        words = name.split(" ")
        embedding_list = []
        for sent in sentences:
            wordpart_list = [self.model.get_word_embedding(sent, w) for w in words]
            cur_embedding = torch.mean(torch.stack(wordpart_list), dim=0)
            if torch.isnan(cur_embedding).any():
                print("Nan in sentence: " + sent)
            embedding_list.append(cur_embedding)
        av_embedding = torch.mean(torch.stack(embedding_list), dim=0).numpy()
        if len(av_embedding) != 768 and len(av_embedding) != 1024:
            print(len(av_embedding))
            print(words)
        return av_embedding

    def create_polar_dimensions(self, out_path: str):
        """
        Creates polar dimensions based on antonyms and saves them to a file.

        Args:
            out_path (str): The path to the output directory for saving the polar dimensions.

        Returns:
            None.

        Raises:
            None.
        """
        antonym_dict = self.load_antonyms_from_json(self.antonym_path)
        print(antonym_dict)
        direction_vectors = []
        for antonym_wn, sentences in antonym_dict.items():
            if not self.check_sentences(sentences):
                print("Unable to create POLAR dimensions.")
                return

            anto1_embedding, anto2_embedding = [self.get_average_embedding(name, sents) for name, sents in sentences.items()]
            cur_direction_vector = anto2_embedding - anto1_embedding

            if np.isnan(cur_direction_vector).any():
                print("Nan... Unable to create POLAR dimensions.")
                return

            direction_vectors.append(cur_direction_vector)

        out_dir_path = out_path + "/polar_dimensions.pkl"
        with open(out_dir_path, 'wb') as handle:
            pickle.dump(direction_vectors, handle, protocol=pickle.HIGHEST_PROTOCOL)
