import torch
from transformers import  GPT2TokenizerFast, GPT2Model
import numpy as np

class GPT2WordEmbeddings:
    """
    A class that provides GPT-2 word embeddings.

    Attributes:
    -----------
    tokenizer : GPT2TokenizerFast
        A fast tokenizer object from the transformers package
    model : GPT2Model
        A GPT-2 model object from the transformers package

    Methods:
    --------
    get_hidden_states(encoded, layer=-1)
        Takes an encoded sentence and returns the hidden states of the specified layer in the model.
    get_word_embedding(sentence, word, layer=-1)
        Takes a sentence and a word and returns the word embedding of that word in the sentence from the specified layer.
    """

    def __init__(self, model_name='gpt2', layer=2, avg_layers=False):
        """
        Initializes a GPT-2 fast tokenizer and model object.

        Parameters:
        -----------
        model_name : str, optional
            The name of the GPT-2 model to be used for generating embeddings, by default 'gpt2'
        """
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        self.model_name = model_name
        self.layer = layer
        self.avg_layers = avg_layers
        self.model = GPT2Model.from_pretrained(model_name, output_hidden_states=True)
        self.model.eval()

    def get_hidden_states(self, encoded, layer=-1):
        """
        Takes an encoded sentence and returns the hidden states of the specified layer in the model.

        Parameters:
        -----------
        encoded : dictionary
            A dictionary containing the encoded sentence.
        layer : int, optional
            The index of the layer to retrieve the hidden states from, by default -1 (last layer)

        Returns:
        --------
        states : torch.Tensor
            The hidden states of the specified layer.
        """
        with torch.no_grad():
            output = self.model(**encoded)
        states = output.hidden_states[layer]
        return states

    def get_word_embedding(self, sentence, word):
        """
        Takes a sentence, a word, and an optional layer index and returns the word embedding of that word in the sentence
        at the specified layer.

        Parameters:
        -----------
        sentence : str
            The input sentence.
        word : str
            The word to get the embedding for.
        layer : int, optional
            The index of the layer to extract the word embedding from, by default -1 (last layer).

        Returns:
        --------
        word_embedding : torch.Tensor or None
            The word embedding of the word in the sentence at the specified layer, or None if the word is not found.
        """
        tokenized_sentence = self.tokenizer.tokenize(sentence)
        idx = tokenized_sentence.index(word) if word in tokenized_sentence else -1
        encoded = self.tokenizer.encode_plus(sentence, return_tensors="pt")
        token_ids_word = np.where(encoded.input_ids[0] == encoded.input_ids[0][idx])
        with torch.no_grad():
            output = self.model(**encoded, output_hidden_states=True)
        states = output.hidden_states
        if self.avg_layers:
            embeddings_to_average = states[-self.layer:]
            word_tokens_output = torch.cat([output[0][token_ids_word] for output in embeddings_to_average], dim=0)
            word_embedding = word_tokens_output.mean(dim=0)
        else:
            output = states[-self.layer][0]
            word_embedding = output[token_ids_word].mean(dim=0)
        return word_embedding
