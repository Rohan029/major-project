import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras

import re
import string
import nltk

from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import joblib

import streamlit as st

#loading the model globally
model = tf.keras.models.load_model('medicalspecalty.h5')

# loading the tokenizer globally
tokenizer = joblib.load('tokenfinal.pkl')

# setting up the values as gloabl variables
value = {
0:'surgery',
1:'consult history and phy',
2:'cardiovascular pulmonary',
3:'orthopedic',
4:'radiology',
5:'general medicine',
6:'gastroenterology',
7:'neurology',
8:'obstetrics gynecology',
9:'urology',
10:'ent otolaryngology',
11:'neurosurgery',
12:'hematology oncology',
13:'ophthalmology',
14:'nephrology',
15:'pediatrics neonatal',
16:'pain management',
17:'psychiatry psychology',
18:'podiatry',
}


#cleaning 
def cleaning(text):
    text = str(text)
    text = text.lower()
    pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    clean = re.compile('<.*?>')
    text = re.sub(clean,'',text)
    text = pattern.sub('', text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"did't", "did not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"couldn't", "could not", text)
    text = re.sub(r"have't", "have not", text)

    text = re.sub(r"[,.\"!@#$%^&*(){}?/;`~:<>+=-]", "", text)
    tokens = word_tokenize(text)
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    text = ' '.join(words)
    return text

#creating a function for prediction 
def medical_specialty(text):

    text = cleaning(text)
    
    text = tokenizer.texts_to_sequences([text])
    
    text = pad_sequences(text,maxlen=300,padding='post')
    
    predict = model.predict(text)
    prediction = value[np.argmax(predict)]
    
    return prediction


def main():
    #giving a title
    st.title("Medical Specialty Predictor Web App")

    #getting the input data from the user
    text_new = st.text_input("Enter the description")

    #code for prediction
    medic = ''

    #creating a button for prediction 
    if st.button('Submit'):
        medic = medical_specialty(text_new)

    st.success(medic)


if __name__ == '__main__':
    main()
    