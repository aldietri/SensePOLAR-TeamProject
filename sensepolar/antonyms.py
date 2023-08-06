import numpy as np
from scipy import linalg

class AntonymSpace:
    " AntonymSpace class provides a way to obtain word embeddings for a given word, based on a set of antonyms and definitions."
    def __init__(self, antonym_path, definition_path):
        """
        Initializes the AntonymSpace object by loading the antonyms and definitions data.

        Args:
            antonym_path (str): File path for antonyms data in numpy .npy format.
            definition_path (str): File path for definitions data in numpy .npy format.
        """
        self.antonyms = np.load(antonym_path, allow_pickle=True)
        self.definitions = np.load(definition_path, allow_pickle=True)
        self.W_norm, self.W_inverse = None, None
    
    def get_W(self):
        """
        Returns two matrices: the normalized version of the antonym space matrix, and the inverse of the transpose of the normalized antonym space matrix.

        Returns:
            W_norm (np.ndarray): Normalized antonym space matrix.
            W_inverse (np.ndarray): Inverse of the transpose of the normalized antonym space matrix.
        """
        if len(self.antonyms[0]) == 3:
            # Case [anto-1, anto1, direction]
            axisList=[]
            for antony in self.antonyms:
                axisList.append(antony[2])
        else:
            # Case [direction1, direction2]
            axisList=self.antonyms #[0:768] # 1763 pairs
        W = np.matrix(axisList)
        W_inverse = linalg.pinv(np.transpose(W))
        W_norm = W/np.linalg.norm(W, axis=1, keepdims=True)
        return W_norm, W_inverse
    
    def get_word_embedding(self, word, use_definition=True):
        """
        Returns the word embedding for a given word, based on the antonyms and definitions data.

        Args:
            word (str): The word to get the embedding for.
            use_definition (bool, optional): Whether to use the definition data or antonym data to get the word embedding. Defaults to True.

        Returns:
            np.ndarray: The word embedding for the given word, or None if the word is not found in the antonyms or definitions data.
        """
        if use_definition:
            definition_index = np.where(self.definitions[:, 0] == word)[0]
            if definition_index.size == 0:
                return None
            embedding = self.definitions[definition_index, 1:]
        else:
            antonym_index = np.where(np.array(self.antonyms)[:, 0] == word)[0]
            if antonym_index.size == 0:
                return None
            embedding = self.antonyms[antonym_index, 1:]
        return embedding.astype(float)