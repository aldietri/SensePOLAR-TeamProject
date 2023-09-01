import sys
relative_path = "../../"
sys.path.append(relative_path)
import os
from lookup import LookupCreator
from sensepolar.polarity import WordPolarity
from sensepolar.embed.bertEmbed import BERTWordEmbeddings
from sensepolar.embed.albertEmbed import ALBERTWordEmbeddings
from sensepolar.embed.robertaEmbed import RoBERTaWordEmbeddings
from sensepolar.polarDim import PolarDimensions
from sensepolar.oracle.dictionaryapi import Dictionary
import nltk
import pandas as pd
from nltk.stem import PorterStemmer
from datasets import load_dataset
from datasets import Dataset as Data
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from transformers import BertTokenizer, BertModel
from IPython.utils import io
import re
import torch.nn.utils.rnn as rnn_utils

class PoemSentimentDataset(Dataset):
    def __init__(self, verse_text, labels, word_polarity_model, method='cls', dimension=39):
        self.verse_text = verse_text
        self.labels = labels
        self.word_polarity_model = word_polarity_model
        self.polar_embeddings_cache = {}
        self.dimension = dimension
        self.method = method

    def __len__(self):
        return len(self.verse_text)

    def get_sense_polar_embedding(self, word, context):
        if (word, context) not in self.polar_embeddings_cache:
            with io.capture_output() as captured:
                polar_embedding = self.word_polarity_model.analyze_word(word, context)
            antonym_dict = {}
            for pair in polar_embedding:
                antonym_dict[(pair[0], pair[1])] = antonym_dict.get((pair[0], pair[1]), []) + [pair[2]]
            sorted_antonym_dict = dict(sorted(antonym_dict.items(), key=lambda item: item[0]))
            self.polar_embeddings_cache[(word, context)] = list(sorted_antonym_dict.values())
        return self.polar_embeddings_cache[(word, context)]

    def __getitem__(self, idx):
        verse = self.verse_text[idx]
        labels = self.labels[idx]
        verse_polar_embeddings = None

        if self.method == 'cls':
            verse += ' [CLS]'
            cls_polar_embedding = self.get_sense_polar_embedding('[CLS]', verse)
            verse_polar_embeddings = torch.tensor(cls_polar_embedding, dtype=torch.float)
        else:
            polar_embeddings_list = []
            for word in verse.split():
                polar_embedding = self.get_sense_polar_embedding(word, verse)
                polar_embeddings_list.append(polar_embedding)
            verse_polar_embeddings = torch.tensor(polar_embeddings_list, dtype=torch.float)
            verse_polar_embeddings = torch.mean(verse_polar_embeddings, dim=0)
            
        verse_polar_embeddings = verse_polar_embeddings.long()
        verse_polar_embeddings = verse_polar_embeddings.squeeze(dim=1)
        verse_polar_embeddings = verse_polar_embeddings[:self.dimension]
        label = torch.tensor(labels, dtype=torch.long)

        return {
            'polar_embeddings': verse_polar_embeddings,
            'label': label
        }


class PolarEmbeddingClassifier(nn.Module):
    def __init__(self, num_classes, polar_dimension, model_name='sense_polar_model'):
        super(PolarEmbeddingClassifier, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(polar_dimension, num_classes)
        self.model_name = model_name

    def forward(self, polar_embeddings):
        output = self.dropout(polar_embeddings.float())
        logits = self.fc(output)
        return logits

    def train_model(self, train_loader, valid_loader, num_epochs, patience, optimizer, loss_fn, device):
        best_valid_loss = float('inf')
        epochs_without_improvement = 0
        train_losses = []
        valid_losses = [] 

        for epoch in range(num_epochs):
            self.train()
            train_loss = 0.0
            for batch in train_loader:
                polar_embeddings = batch['polar_embeddings'].to(device)
                labels = batch['label'].to(device)

                optimizer.zero_grad()

                outputs = self(polar_embeddings)  
                logits = outputs
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {avg_train_loss}")

            # Validation
            self.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for batch in valid_loader:
                    polar_embeddings = batch['polar_embeddings'].to(device)  
                    labels = batch['label'].to(device)

                    outputs = self(polar_embeddings)
                    logits = outputs
                    loss = loss_fn(logits, labels)

                    valid_loss += loss.item()

            avg_valid_loss = valid_loss / len(valid_loader)
            valid_losses.append(avg_valid_loss)
            print(f"Epoch {epoch+1}/{num_epochs} - Validation Loss: {avg_valid_loss}")

            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss
                epochs_without_improvement = 0
                torch.save(self.state_dict(), "model/" + self.model_name + ".pth")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping. No improvement in {patience} epochs.")
                    break

        self.load_state_dict(torch.load("model/" + self.model_name +".pth"))
        # Save results in a file
        with open('results/' + self.model_name+'_train.txt', 'w') as f:
            for epoch, train_loss, valid_loss in zip(range(1, num_epochs + 1), train_losses, valid_losses):
                f.write(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}\n")


    def test_model(self, test_loader, loss_fn, device):
        self.eval()
        test_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        all_predictions = []  # Initialize list to collect predictions
        all_labels = [] 

        with torch.no_grad():
            for batch in test_loader:
                polar_embeddings = batch['polar_embeddings'].to(device) 
                labels = batch['label'].to(device)

                outputs = self(polar_embeddings) 
                logits = outputs
                loss = loss_fn(logits, labels)
                test_loss += loss.item()

                _, predicted = torch.max(logits, dim=1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)
                all_predictions.extend(predicted.tolist())
                all_labels.extend(labels.tolist())

        avg_test_loss = test_loss / len(test_loader)
        accuracy = correct_predictions / total_samples
        print(f"Test Loss: {avg_test_loss}, Accuracy: {accuracy}")
        classification_rep = classification_report(all_labels, all_predictions, digits=4)
        with open('results/' + self.model_name+'_test.txt', 'w') as f:
            f.write(f"Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.4f}\n")
            f.write("\nClassification Report:\n")
            f.write(classification_rep)

    def predict(self, polar_embeddings, device): 
        polar_embeddings = polar_embeddings.to(device)
        
        self.eval()
        with torch.no_grad():
            outputs = self(polar_embeddings)
            logits = outputs

        _, predicted = torch.max(logits, dim=1)
        return predicted.item()

from itertools import product
from sklearn.metrics import classification_report

experiment_settings = {
        "embed_model": [RoBERTaWordEmbeddings, BERTWordEmbeddings, ALBERTWordEmbeddings],  
        "polar_dimension": [ 786, 1586], 
        "WordPolarity_method": ["base-change", "projection"], 
        "PoemSentimentDataset_method": ["avg", "cls"], 
        "layer": [2, 3, 4, 5], 
        "avg_embed": [True]
    }
# Create a list of lists containing values for each setting
setting_values = [values for values in experiment_settings.values()]

# Iterate through all combinations of experiment settings
for setting_combination in product(*setting_values):
    setting = {
        key: value for key, value in zip(experiment_settings.keys(), setting_combination)
    }
    print(setting)
    with io.capture_output() as captured:
        # Extract the values from the current setting
        embed_model = setting["embed_model"]
        polar_dimension = setting["polar_dimension"]
        WordPolarity_method = setting["WordPolarity_method"]
        PoemSentimentDataset_method = setting["PoemSentimentDataset_method"]
        layer = setting["layer"]
        avg_embed = setting["avg_embed"]
        
        print('Setting', embed_model, polar_dimension, WordPolarity_method, PoemSentimentDataset_method, layer)
        dataset = load_dataset("poem_sentiment")
        out_path = './antonyms/'
        antonym_path = "data/polars_all_combined.xlsx"
        embed_model = embed_model(layer=layer, avg_layers=avg_embed)
        
        dictionary = Dictionary('wordnet', api_key='')    
        lookupSpace = LookupCreator(dictionary, out_path, antonyms_file_path=antonym_path)
        lookupSpace.create_lookup_files()
        antonym_path = out_path + "polar_dimensions.pkl"

        pdc = PolarDimensions(embed_model, antonym_path=out_path + "antonym_wordnet_example_sentences_readable_extended.txt")
        pdc.create_polar_dimensions(out_path,"/polar_dimensions.pkl" )

        wp = WordPolarity(embed_model, antonym_path=antonym_path, method=WordPolarity_method)
        num_classes = 4

    # Define your model
    sensepolar_model = PolarEmbeddingClassifier(num_classes=num_classes, polar_dimension=polar_dimension, model_name=f'sense_polar_{embed_model.model_name}_dim{polar_dimension}_{WordPolarity_method}_{PoemSentimentDataset_method}_layer{layer}_avg{avg_embed}')


    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    sensepolar_model.to(device)

    optimizer = torch.optim.AdamW(sensepolar_model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    num_epochs = 1000
    patience = 50

    preprocess_text = lambda verse: re.sub(r'\W+', ' ', re.sub(r'_([^_]+)_', r'\1', verse))
    train_texts = [preprocess_text(verse) for verse in dataset["train"]["verse_text"]]
    test_texts = [preprocess_text(verse) for verse in dataset["test"]["verse_text"]]
    valid_texts = [preprocess_text(verse) for verse in dataset["validation"]["verse_text"]]
    train_labels = dataset["train"]["label"]
    test_labels = dataset["test"]["label"]
    valid_labels = dataset["validation"]["label"]

    # train_texts_filtered = []
    # train_labels_filtered = []
    # for text, label in zip(train_texts, train_labels):
    #     if label != 3:
    #         train_texts_filtered.append(text)
    #         train_labels_filtered.append(label)`

    train_dataset = PoemSentimentDataset(train_texts, train_labels, wp, method=PoemSentimentDataset_method, dimension=polar_dimension)
    valid_dataset = PoemSentimentDataset(valid_texts, valid_labels, wp, dimension=polar_dimension, method=PoemSentimentDataset_method)
    test_dataset = PoemSentimentDataset(test_texts, test_labels, wp, dimension=polar_dimension, method=PoemSentimentDataset_method)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    sensepolar_model.train_model(train_loader, valid_loader, num_epochs, patience, optimizer, loss_fn, device)

    sensepolar_model.test_model(test_loader, loss_fn, device)
