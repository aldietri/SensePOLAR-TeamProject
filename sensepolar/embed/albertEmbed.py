from transformers import AlbertTokenizerFast, AlbertModel
import torch
import numpy as np

class ALBERTWordEmbeddings:
    """
    A class that provides ALBERT word embeddings.

    Attributes:
    -----------
    tokenizer : AlbertTokenizerFast
        A fast tokenizer object from transformers package
    model : AlbertModel
        An ALBERT model object from transformers package

    Methods:
    --------
    get_hidden_states(encoded)
        Takes an encoded sentence and returns the hidden states of the model.
    get_word_embedding(sentence, word)
        Takes a sentence and a word and returns the word embedding of that word in the sentence.
    """

    def __init__(self, model_name='albert-base-v2'):
        """
        Initializes an ALBERT fast tokenizer and model object.

        Parameters:
        -----------
        model_name : str, optional
            The name of the ALBERT model to be used for generating embeddings, by default 'albert-base-v2'
        """
        self.tokenizer = AlbertTokenizerFast.from_pretrained(model_name)
        self.model = AlbertModel.from_pretrained(model_name, output_hidden_states=True)
        self.model.eval()

    def get_hidden_states(self, encoded):
        """
        Takes an encoded sentence and returns the hidden states of the model.

        Parameters:
        -----------
        encoded : dictionary
            A dictionary containing the encoded sentence.

        Returns:
        --------
        states : tuple
            A tuple containing the hidden states of the model.
        """
        with torch.no_grad():
            output = self.model(**encoded)
        states = output.hidden_states
        return states

    def get_word_embedding(self, sentence, word):
        """
        Takes a sentence and a word and returns the word embedding of that word in the sentence.

        Parameters:
        -----------
        sentence : str
            The input sentence.
        word : str
            The word to get the embedding for.

        Returns:
        --------
        word_tokens_output.mean(dim=0) : torch.Tensor
            The word embedding of the word in the sentence.
        """
        idx = sentence.split(" ").index(word)
        if idx == -1:
            return None
        encoded = self.tokenizer(sentence, return_tensors="pt", return_token_type_ids=False)
        token_ids_word = np.where(encoded.word_ids()[0] == idx)
        states = self.get_hidden_states(encoded)
        output = states[-2][0]
        word_tokens_output = output[token_ids_word]
        return word_tokens_output.mean(dim=0)