import requests
from nltk.corpus import wordnet
from sensepolar.oracle.dictionary import BaseDictionary  
import time

class Dictionary:
    """
    A class used to access various online dictionaries and get synonyms, antonyms,
    definitions, and examples of a given word.

    Args:
    dictionary_id (str): the id of the dictionary to use (one of 'dictionaryapi',
                         'oxford', 'wordnik', or 'wordnet')
    api_key (str, optional): the api key to use for the chosen dictionary
    source_lang (str, optional): the source language to use for the chosen dictionary

    Attributes:
    dictionary_id (str): the id of the dictionary being used
    api_key (str): the api key being used for the chosen dictionary (None if not required)
    source_lang (str): the source language being used for the chosen dictionary
    dictionary (BaseDictionary): An instance of the specific dictionary implementation.

    Methods:
    _get_urls(): get the urls for the chosen dictionary
    _make_request(url): make a request to the chosen dictionary
    get_synonyms(word): get the synonyms for a given word from the chosen dictionary
    get_antonyms(word): get the antonyms for a given word from the chosen dictionary
    get_definitions(word): get the definitions for a given word from the chosen dictionary
    get_examples(word): get the examples for a given word from the chosen dictionary
    """
    def __init__(self, dictionary_id, api_key=None, source_lang='en_US'):
        """
        Initialize the Dictionary object with the given dictionary_id, api_key (if required),
        and source_lang (if applicable).
        """
        self.dictionary_id = dictionary_id
        self.api_key = api_key
        self.source_lang = source_lang
        self.dictionary = self._get_dictionary_instance()
        self.examples_cache = {}
        self.definitions_cache = {}

    def _get_dictionary_instance(self):
        if self.dictionary_id == 'dictionaryapi':
            return DictionaryAPI(self.api_key, self.source_lang)
        elif self.dictionary_id == 'wordnik':
            return WordnikDictionary(self.api_key, self.source_lang)
        elif self.dictionary_id == 'wordnet':
            return WordnetDictionary(self.source_lang)
        else:
            raise ValueError(f"Dictionary '{self.dictionary_id}' not supported")

    def get_synonyms(self, word):
        """
        Get the synonyms for a given word from the chosen dictionary.
        """
        return self.dictionary.get_synonyms(word)

    def get_antonyms(self, word):
        """
        Get the antonyms for a given word from the chosen dictionary.
        """
        return self.dictionary.get_antonyms(word)

    def get_definitions(self, word):
        """
        Get the definitions for a given word from the chosen dictionary.
        """
        if word in self.definitions_cache:
            return self.definitions_cache[word]
        self.definitions_cache[word] = self.dictionary.get_definitions(word)
        return self.definitions_cache[word]

    def get_examples(self, word):
        """
        Get the examples for a given word from the chosen dictionary.
        """
        if word in self.examples_cache:
            return self.examples_cache[word]
        self.examples_cache[word] = self.dictionary.get_examples(word)
        return self.examples_cache[word]



class DictionaryAPI(BaseDictionary):
    """
    A class for accessing the DictionaryAPI dictionary and getting synonyms, antonyms,
    definitions, and examples of a given word.

    Args:
    api_key (str, optional): the API key to use for DictionaryAPI
    source_lang (str, optional): the source language to use for DictionaryAPI

    Attributes:
    api_key (str): the API key being used for DictionaryAPI (None if not required)
    source_lang (str): the source language being used for DictionaryAPI

    Methods:
    get_synonyms(word): get the synonyms for a given word from DictionaryAPI
    get_antonyms(word): get the antonyms for a given word from DictionaryAPI
    get_definitions(word): get the definitions for a given word from DictionaryAPI
    get_examples(word): get the examples for a given word from DictionaryAPI
    """
    def __init__(self, api_key=None, source_lang='en_US'):
        self.api_key = api_key
        self.source_lang = source_lang
        self.urls = self._get_urls()
        
    def _get_urls(self):
        """
        Get the URLs for DictionaryAPI.
        """
        urls = {
            'definitions': f'https://api.dictionaryapi.dev/api/v2/entries/{self.source_lang}/{{word}}',
            'examples': f'https://api.dictionaryapi.dev/api/v2/entries/{self.source_lang}/{{word}}'
        }
        return urls

    def get_synonyms(self, word):
        """
        Get the synonyms for a given word from DictionaryAPI.
        """
        url = self.urls['definitions'].format(word=word)
        response = self._make_request(url)
        if response and isinstance(response, list):
            synonyms = []
            for entry in response:
                meanings = entry.get("meanings", [])
                for meaning in meanings:
                    definitions = meaning.get("definitions", [])
                    if definitions:
                        for definition in definitions:
                            synonym = definition.get("synonym")
                            if synonym:
                                synonyms.extend(synonym)
            return synonyms
        return []

    def get_antonyms(self, word):
        """
        Get the antonyms for a given word from DictionaryAPI.
        """
        url = self.urls['definitions'].format(word=word)
        response = self._make_request(url)
        if response and isinstance(response, list):
            antonyms = []
            for entry in response:
                meanings = entry.get("meanings", [])
                for meaning in meanings:
                    definitions = meaning.get("definitions", [])
                    if definitions:
                        for definition in definitions:
                            antonym = definition.get("antonym")
                            if antonym:
                                antonyms.extend(antonym)
            return antonyms
        return []

    def get_definitions(self, word):
        """
        Get the definitions for a given word from DictionaryAPI.
        """
        url = self.urls['definitions'].format(word=word)
        response = self._make_request(url)
        all_definitions = []
        if response and isinstance(response, list):
            definitions = []
            for entry in response:
                meanings = entry.get("meanings", [])
                for meaning in meanings:
                    definitions = meaning.get("definitions", [])
                    if definitions:
                        for definition in definitions:
                            example_uses = definition.get("example")
                            if example_uses and len(example_uses) > 0:
                                all_definitions.append([definition.get("definition")])
        return list(all_definitions)

    def get_examples(self, word):
        """
        Get the examples for a given word from DictionaryAPI.
        """
        all_examples = []
        url = self.urls['examples'].format(word=word)
        response = self._make_request(url)
        if response and isinstance(response, list):
            examples = []
            for entry in response:
                meanings = entry.get("meanings", [])
                for meaning in meanings:
                    definitions = meaning.get("definitions", [])
                    if definitions:
                        for definition in definitions:
                            example_uses = definition.get("example")
                            if example_uses and len(example_uses) > 1:
                                all_examples.append(example_uses)
        return all_examples

    def _make_request(self, url):
        """
        Make a request to DictionaryAPI with the given URL.
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        return None



class WordnikDictionary(BaseDictionary):
    """
    A class for accessing the Wordnik dictionary and getting synonyms, antonyms,
    definitions, and examples of a given word.

    Args:
    api_key (str, optional): the API key to use for Wordnik
    source_lang (str, optional): the source language to use for Wordnik

    Attributes:
    api_key (str): the API key being used for Wordnik (None if not required)
    source_lang (str): the source language being used for Wordnik

    Methods:
    get_synonyms(word): get the synonyms for a given word from Wordnik
    get_antonyms(word): get the antonyms for a given word from Wordnik
    get_definitions(word): get the definitions for a given word from Wordnik
    get_examples(word): get the examples for a given word from Wordnik
    """
    def __init__(self, api_key=None, source_lang='en_US'):
        self.api_key = api_key
        self.source_lang = source_lang
        self.urls = self._get_urls()

    def _get_urls(self):
        """
        Get the URLs for Wordnik.
        """
        urls = {
            'definitions': f'https://api.wordnik.com/v4/word.json/{{word}}/definitions?limit=5&includeRelated=false&useCanonical=false&includeTags=false&api_key={self.api_key}',
            'examples': f'https://api.wordnik.com/v4/word.json/{{word}}/definitions?limit=5&includeRelated=false&useCanonical=false&includeTags=false&api_key={self.api_key}'
        }
        return urls

    def get_definitions(self, word):
        """
        Get the definitions for a given word from Wordnik.
        """
        url = self.urls['definitions'].format(word=word)
        response = self._make_request(url)
        all_definitions = []
        if response and isinstance(response, list):
            definitions = []
            for entry in response:
                if 'text' in entry.keys():
                    definition = entry['text']
                    examples = [example['text'] for example in entry.get('exampleUses', []) if example['text']]
                    if len(examples) > 0:
                        all_definitions.append([definition])
        return list(all_definitions)

    def get_examples(self, word):
        """
        Get the examples for a given word from Wordnik.
        """
        url = self.urls['examples'].format(word=word)
        response = self._make_request(url)
        all_examples = []
        if response and isinstance(response, list):
            examples = []
            for entry in response:
                if 'text' in entry.keys():
                    definition = entry['text']
                    examples = [example['text'] for example in entry['exampleUses'] if 'text' in example.keys()]
                    if len(examples) > 0:
                        all_examples.append(examples)
        return all_examples

    def _make_request(self, url):
        """
        Make a request to Wordnik with the given URL.
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        for _ in range(10):
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                time.sleep(3)
        return None


class WordnetDictionary(BaseDictionary):
    """
    A class for accessing WordNet dictionary and getting synonyms, antonyms,
    definitions, and examples of a given word.

    Methods:
    get_synonyms(word): get the synonyms for a given word from WordNet
    get_antonyms(word): get the antonyms for a given word from WordNet
    get_definitions(word): get the definitions for a given word from WordNet
    get_examples(word): get the examples for a given word from WordNet
    """
   
    def _get_urls(self):
        """
        Get the URLs for DictionaryAPI.
        """
        return None
    
    def get_definitions(self, word):
        """
        Get the definitions for a given word from WordNet.
        """
        definitions = []
        for synset in wordnet.synsets(word):
            if len(synset.examples()) == 0:
                continue
            definitions.append([synset.definition()])
        return definitions

    def get_examples(self, word):
        """
        Get the examples for a given word from WordNet.
        """
        examples = []
        for synset in wordnet.synsets(word):
                if len(synset.examples()) == 0:
                    continue
                examples.append(synset.examples())
        return examples
