import requests

class BaseDictionary:
    """
    A base class for accessing online dictionaries and getting synonyms, antonyms,
    definitions, and examples of a given word.

    Attributes:
    api_key (str): the API key to use for the dictionary (None if not required)
    source_lang (str): the source language for the dictionary
    urls (dict): a dictionary mapping URL types to URLs for the dictionary

    Methods:
    _get_urls(): get the URLs for the dictionary
    _make_request(url): make a request to the dictionary with the given URL
    get_synonyms(word): get the synonyms for a given word from the dictionary
    get_antonyms(word): get the antonyms for a given word from the dictionary
    get_definitions(word): get the definitions for a given word from the dictionary
    get_examples(word): get the examples for a given word from the dictionary
    """
    def __init__(self, api_key=None, source_lang='en_US'):
        self.api_key = api_key
        self.source_lang = source_lang
        self.urls = self._get_urls()
        self.examples_cache = {}
        self.definitions_cache = {}

    def _get_urls(self):
        """
        Get the URLs for the dictionary (to be implemented by subclasses).
        """
        raise NotImplementedError("Subclasses must implement _get_urls() method.")

    def _make_request(self, url):
        """
        Make a request to the dictionary with the given URL.
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            return None

    def get_synonyms(self, word):
        """
        Get the synonyms for a given word from the dictionary (to be implemented by subclasses).
        """
        return None
        # raise NotImplementedError("Subclasses must implement get_synonyms(word) method.")

    def get_antonyms(self, word):
        """
        Get the antonyms for a given word from the dictionary (to be implemented by subclasses).
        """
        return None
        # raise NotImplementedError("Subclasses must implement get_antonyms(word) method.")

    def get_definitions(self, word):
        """
        Get the definitions for a given word from the dictionary (to be implemented by subclasses).
        """
        raise NotImplementedError("Subclasses must implement get_definitions(word) method.")

    def get_examples(self, word):
        """
        Get the examples for a given word from the dictionary (to be implemented by subclasses).
        """
        raise NotImplementedError("Subclasses must implement get_examples(word) method.")
