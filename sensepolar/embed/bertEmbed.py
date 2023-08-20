import torch
from transformers import BertTokenizerFast, BertModel
import numpy as np

class BERTWordEmbeddings:
    """
    A class that provides BERT word embeddings.

    Attributes:
    -----------
    tokenizer : BertTokenizerFast
        A tokenizer object from transformers package
    model : BertModel
        A BERT model object from transformers package

    Methods:
    --------
    get_hidden_states(encoded)
        Takes an encoded sentence and returns the hidden states of the model.
    get_word_embedding(sentence, word)
        Takes a sentence and a word and returns the word embedding of that word in the sentence.
    """

    def __init__(self, model_name='bert-base-uncased', layer=2):
        """
        Initializes a BERT tokenizer and model object.

        Parameters:
        -----------
        model_name : str, optional
            The name of the BERT model to be used for generating embeddings, by default 'bert-base-uncased'
        """
        self.model_name = model_name
        self.layer = layer
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name, output_hidden_states=True)
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
        encoded = self.tokenizer.encode_plus(sentence, return_tensors="pt")
        token_ids_word =np.where(np.array(encoded.word_ids()) == idx)
        states = self.get_hidden_states(encoded)
        output = states[-self.layer][0]
        word_tokens_output = output[token_ids_word]
        return word_tokens_output.mean(dim=0)
