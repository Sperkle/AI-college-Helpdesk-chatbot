from spacy.lang.en import English
import numpy
from flask import Flask, render_template, request
import pandas as pd
import time
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import GlobalMaxPooling1D, Dense, Activation, Dropout, Embedding,Conv1D
import random

nlp = English()
tokenizer = nlp.tokenizer
PAD_Token=0

app = Flask(__name__)
     
model= load_model('mymodel.h5')
        
# Load the model
with open("intents.json") as file:
    intents = json.load(file)

df = pd.DataFrame(columns=['Pattern', 'Tag'])

def extract_json_info(json_file, df):
    # Iterate over each intent in the JSON file
    for intent in json_file['intents']:
        # Iterate over each pattern in the current intent
        for pattern in intent['patterns']:
            # Create a list containing the pattern and its associated tag
            sentence_tag = [pattern, intent['tag']]
            # Append the pattern and tag to the DataFrame
            df.loc[len(df.index)] = sentence_tag
     # Return the updated DataFrame
    return df
df = extract_json_info(intents, df)
labels = df['Tag'].unique().tolist()
labels = [s.strip() for s in labels]
#count each tag
tag_counts = df['Tag'].value_counts()

# Tokenize the text patterns
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['Pattern'])

X = tokenizer.texts_to_sequences(df['Pattern'])

max_sequence_length = max(len(seq) for seq in X)
X_padded = pad_sequences(X, maxlen=max_sequence_length, padding='post')


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Tag'])

def preprocess_input_sentence(sentence, tokenizer, max_sequence_length):
    # Tokenize input sentence
    input_sequence = tokenizer.texts_to_sequences([sentence])
    # Pad sequences
    padded_sequence = pad_sequences(input_sequence, maxlen=max_sequence_length, padding='post')
    return padded_sequence

# Function to get response
def get_response(sentence, loaded_model, tokenizer, max_sequence_length, intents):
    # Preprocess input sentence
    preprocessed_input = preprocess_input_sentence(sentence, tokenizer, max_sequence_length)
    # Predict label
    predicted_label = loaded_model.predict(preprocessed_input).argmax(axis=-1)
    # Convert label to tag
    predicted_tag = label_encoder.inverse_transform(predicted_label)
    
    # Iterate through the intents to find the matching tag
    for intent in intents["intents"]:
        if intent["tag"] == predicted_tag[0]:
            # Select a random response from the list of responses
            response = random.choice(intent["responses"])
            return response
    
    # If no matching intent is found, return a default message
    return "I'm sorry, I didn't understand that."



@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    sentence = request.args.get('msg')
    time.sleep(0.5)
    response = get_response(sentence, model, tokenizer, max_sequence_length, intents)
    return response

if __name__ == "__main__":
        app.run(debug=True)
 
