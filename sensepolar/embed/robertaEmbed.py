from transformers import RobertaTokenizerFast, RobertaModel
import numpy as np
import torch

class RoBERTaWordEmbeddings:
    """
    A class that provides RoBERTa word embeddings.

    Attributes:
    -----------
    tokenizer : RobertaTokenizer
        A tokenizer object from the transformers package.
    model : RobertaModel
        A RoBERTa model object from the transformers package.

    Methods:
    --------
    get_hidden_states(encoded)
        Takes an encoded sentence and returns the hidden states of the model.
    get_word_embedding(sentence, word, layer=-2)
        Takes a sentence, a word, and an optional layer index and returns the word embedding of that word in the sentence
        at the specified layer.
    """

    def __init__(self, model_name='roberta-base'):
        """
        Initializes a RoBERTa tokenizer and model object.

        Parameters:
        -----------
        model_name : str, optional
            The name of the RoBERTa model to be used for generating embeddings, by default 'roberta-base'.
        """
        self.tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name, output_hidden_states=True)
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
        output = states[-5][0]
        word_tokens_output = output[token_ids_word]
        return word_tokens_output.mean(dim=0)
