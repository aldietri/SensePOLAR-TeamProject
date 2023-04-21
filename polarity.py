import pickle
import torch
from antonyms import AntonymSpace

class WordPolarity:
    "WordPolarity class is used for analyzing the polarity of a word in a given context."
    def __init__(self, model, antonym_path="", lookup_path="./antonyms/", normalize_term_path="antonyms/wordnet_normalize.pkl", number_polar=5, method="base-change"):
        """
        Initialize a WordPolarity object.

        Args:
            model: A pre-trained word embedding model, with a get_word_embedding method that takes in a context and a word.
            antonym_path (str, optional): Path to the pickled antonyms file. Defaults to "".
            lookup_path (str, optional): Path to the antonym lookup dictionary. Defaults to "./antonyms/".
            normalize_term_path (str, optional): Path to the pickled normalize_term file. Defaults to "antonyms/wordnet_normalize.pkl".
            number_polar (int, optional): Number of polar dimensions to consider. Defaults to 5.
            method (str, optional): Transformation method for antonyms space. Valid options are "base-change" and "projection". Defaults to "base-change".
        """
        self.antonym_path = antonym_path
        self.lookup_path = lookup_path
        self.normalize_term_path = normalize_term_path
        self.number_polar = number_polar
        self.method = method
        self.model = model
        self.antonyms = None
        self.definitions = None
        self.normalize_term = None
        self.W_norm_np = None
        self.W_inv_np = None
        self.W_torch = None

        if self.antonym_path == "":
            self.antonym_path = self.lookup_path + "antonym_wordnet_base_change.pkl"

        if self.normalize_term_path is not None:
            with open(self.normalize_term_path, 'rb') as f:
                self.normalize_term = pickle.load(f)

        self.load_antonyms()
        self.load_definitions()
        self.load_W()

    def load_antonyms(self):
        """
        Load antonyms from pickled file.
        """
        print("Loading antonyms from " + self.lookup_path + "lookup_anto_example_dict.pkl")
        with open(self.lookup_path + "lookup_anto_example_dict.pkl", 'rb') as f:
            self.antonyms = pickle.load(f)

    def load_definitions(self):
        """
        Load word definitions from pickled file.
        """
        with open(self.lookup_path + "lookup_synset_definition.pkl", 'rb') as f:
            self.definitions = pickle.load(f)

    def load_W(self):
        """
        Load antonyms space and transform it based on method parameter.
        """
        antonym_space = AntonymSpace(antonym_path=self.antonym_path, definition_path=self.lookup_path + "lookup_synset_definition.pkl")
        self.W_norm_np, self.W_inv_np = antonym_space.get_W()

        if self.method == "base-change":
            self.W_torch = torch.from_numpy(self.W_inv_np)
        elif self.method == "projection":
            self.W_torch = torch.from_numpy(self.W_norm_np)
        else:
            raise ValueError("Invalid transformation method. Valid options are 'base-change' and 'projection'.")

    def analyze_word(self, word, context):
        """
        Analyze the polarity of a word in a given context.

        Args:
            word (str): The word to analyze.
            context (str): The context in which to analyze the word.

        Returns:
            axis_list (list): The top dimensions for the given word with the polar_value.
        """
        if word not in context.split():
            print("Warning: The context must contain the *exact* word you want to analyze!")
            return None

        cur_word_emb = self.model.get_word_embedding(context, word)

        if self.normalize_term is not None:
            cur_word_emb -= self.normalize_term

        polar_emb = torch.matmul(self.W_torch, cur_word_emb)
        polar_emb_np = polar_emb.numpy()

        axis_list = self.get_top_word_dimensions(polar_emb_np)
        return axis_list
    

    def get_top_word_dimensions(self, word_embedding):
        """
        Retrieves the top dimensions for the given word based on its word embedding.

        Parameters:
            word_embedding (numpy.ndarray): A numpy array representing the word embedding.

        Returns:
            List of tuples: A list of tuples, where each tuple contains the two antonyms that define a dimension.
        """
        thisdict = {i: v for i, v in enumerate(word_embedding)}
        sorted_dic = sorted(thisdict.items(), key=lambda item: abs(item[1]), reverse=True)

        axis_list = []
        for i in range(self.number_polar):
            cur_index = sorted_dic[i][0]
            cur_value = sorted_dic[i][1]
            left_polar = self.antonyms[cur_index][0][0]
            left_definition = self.definitions[cur_index][0]
            right_polar = self.antonyms[cur_index][1][0]
            right_definition = self.definitions[cur_index][1]
            axis = (left_polar, right_polar, cur_value)
            axis_list.append(axis)
            print("Top:", i + 1)
            print("Dimension:", left_polar, "<------>", right_polar)
            print("Definitions: ", left_definition+ "<------>" + right_definition)
            print("Value: " + str(cur_value) if cur_value < 0 else f"Value:{cur_value}")
            print("\n")
        return axis_list  # top dimensions for given word (string)