from bertFuncs import forwardWord
import numpy as np
import json
import torch
import pickle


def loadAntonymsFromJson(dict_path):
  # Read the antonyms and their example sentences from a json 
  if "txt" in dict_path: 
    with open(dict_path) as f:
      antonym_dict = json.load(f)
  return antonym_dict


def checkSentences(sentences): 
  # Check format of antonym pairs and example sentences
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


def getAverageEmbedding(name, sentences, model, tokenizer):
  # compute the average BERT embedding of all subwords in all provided example sentences
  words = name.split(" ")  # Ex: [disentangle], or [put, in]
  embedding_list= []
  for sent in sentences: 
    # average of all word parts
    wordpart_list=[forwardWord(tokenizer, model, sent, w) for w in words]
    cur_embedding = torch.mean(torch.stack(wordpart_list), dim=0)
    if torch.isnan(cur_embedding).any():
      print("Nan in sentence: " + sent)
    embedding_list.append(cur_embedding)
  # average over all example sentences
  av_embedding=torch.mean(torch.stack(embedding_list), dim=0).numpy()
  if len(av_embedding) != 768 and len(av_embedding) != 1024:   # ?
    print(len(av_embedding))
    print(words)
  return av_embedding


def createPolarDimension(model, tokenizer, out_path, antonym_path=""):
# creates SensePOLAR dimensions from a predefined set of antonym pairs
  print("Start forwarding the Polar opposites ...")
  if antonym_path == "":  # default path
    dict_path = "antonyms/antonym_wordnet_example_sentences_readable_extended.txt" 
  else:
    dict_path = antonym_path
  antonym_dict = loadAntonymsFromJson(dict_path)
  direction_vectors=[]

  for antonym_wn, sentences in antonym_dict.items():
    if not checkSentences(sentences):
      print("Unable to create POLAR dimensions.")
      return
        
    # compute direction vector
    anto1_embedding, anto2_embedding = [getAverageEmbedding(name, sents, model, tokenizer) for name, sents in sentences.items()]
    cur_direction_vector = anto2_embedding - anto1_embedding
        
    if np.isnan(cur_direction_vector).any(): 
      print("Nan... Unable to create POLAR dimensions.")
      return

    direction_vectors.append(cur_direction_vector)
        
  # safe direction vectors
  out_dir_path = out_path+"polar_dimensions.pkl"
  with open(out_dir_path, 'wb') as handle:
    pickle.dump(direction_vectors, handle, protocol=pickle.HIGHEST_PROTOCOL)
  return