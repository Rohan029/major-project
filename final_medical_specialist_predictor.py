# -*- coding: utf-8 -*-
"""final_medical_specialist_predictor.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Rf3vBTW_9DueW5cYXl0mF03VuARO0qjD
"""

# importing the libraries
import numpy as np
import pandas as pd
import string
import re
import nltk
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split as tts

nltk.download('punkt')

print("All the modules imported....")

pip install joblib

import joblib

train = pd.read_csv("/content/mtsamples.csv")
print('Dataset loaded....')

train.head()

train.tail()

train.info()

train.describe()

train.shape

# preprocessing the data

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

print("Cleaning is taking place....")
train['text_new'] = train['description'] + ' ' + train['sample_name'] + ' ' + train['transcription'] + ' ' + train['keywords']
train['text_new'] = train['text_new'].map(cleaning)
train["medical_specialty"] = train['medical_specialty'].map(cleaning)
print("Cleaning is done....")

train["medical_specialty"].value_counts()

sns.countplot(y='medical_specialty', data=train)

train = train.drop(index=train[train['medical_specialty'] == 'allergy immunology'].index)
train = train.drop(index=train[train['medical_specialty'] == 'lab medicine pathology'].index)
train = train.drop(index=train[train['medical_specialty'] == 'autopsy'].index)
train = train.drop(index=train[train['medical_specialty'] == 'speech language'].index)
train = train.drop(index=train[train['medical_specialty'] == 'diets and nutritions'].index)
train = train.drop(index=train[train['medical_specialty'] == 'rheumatology'].index)
train = train.drop(index=train[train['medical_specialty'] == 'chiropractic'].index)
train = train.drop(index=train[train['medical_specialty'] == 'imeqmework comp etc'].index)
train = train.drop(index=train[train['medical_specialty'] == 'bariatrics'].index)
train = train.drop(index=train[train['medical_specialty'] == 'endocrinology'].index)
train = train.drop(index=train[train['medical_specialty'] == 'sleep medicine'].index)
train = train.drop(index=train[train['medical_specialty'] == 'physical medicine rehab'].index)
train = train.drop(index=train[train['medical_specialty'] == 'letters'].index)
train = train.drop(index=train[train['medical_specialty'] == 'dentistry'].index)
train = train.drop(index=train[train['medical_specialty'] == 'cosmetic plastic surgery'].index)
train = train.drop(index=train[train['medical_specialty'] == 'dermatology'].index)
train = train.drop(index=train[train['medical_specialty'] == 'hospice palliative care'].index)
train = train.drop(index=train[train['medical_specialty'] == 'soap chart progress notes'].index)
train = train.drop(index=train[train['medical_specialty'] == 'discharge summary'].index)
train = train.drop(index=train[train['medical_specialty'] == 'emergency room reports'].index)
train = train.drop(index=train[train['medical_specialty'] == 'office notes'].index)

train["medical_specialty"].value_counts()

train.shape

sns.countplot(y='medical_specialty', data=train)

train.shape

train.info()

train.isna().sum()

train=train.dropna()

train.isna().sum()

train.shape

train["medical_specialty"].unique()

value = {
'surgery':0,
'consult history and phy':1,
'cardiovascular pulmonary':2,
'orthopedic':3,
'radiology':4,
'general medicine':5,
'gastroenterology':6,
'neurology':7,
'obstetrics gynecology':8,
'urology':9,
'ent otolaryngology':10,
'neurosurgery':11,
'hematology oncology':12,
'ophthalmology':13,
'nephrology':14,
'pediatrics neonatal':15,
'pain management':16,
'psychiatry psychology':17,
'podiatry':18,
}

train['medical_specialty'] = train["medical_specialty"].map(value)

value['podiatry']

value['urology']

train["medical_specialty"].unique()

print('Formulating the dependent and independent variable....')
train = train[['text_new','medical_specialty']]
X = train['text_new'].values
y = train['medical_specialty'].values

print("Splitting the data into train and test....")
xtrain,xtest,ytrain,ytest = tts(X,y,test_size=0.2,random_state=42,stratify=y)
print("Splitting done....")
print("Total data points in Train data....", ytrain.shape[0])
print("Total data points in the Test data....", ytest.shape[0])

# converting dependent variable into categorical variable
ytrainf = to_categorical(ytrain)
ytestf = to_categorical(ytest)

# converting to text to sequences
tokenizer=Tokenizer(20000,lower=True,oov_token='UNK')
tokenizer.fit_on_texts(xtrain)
xtrain = tokenizer.texts_to_sequences(xtrain)
xtest = tokenizer.texts_to_sequences(xtest)

xtrain = pad_sequences(xtrain,maxlen=300,padding='post')
xtest = pad_sequences(xtest,maxlen=300,padding='post')
print("Data preprocessing is over....")

joblib.dump(tokenizer,"tokenfinal.pkl")

# making the model
print("Making the model....")
model = Sequential()
model.add(Embedding(20000,64,input_length=300))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(64,return_sequences=True)))
model.add(Bidirectional(LSTM(128)))
model.add(Dropout(0.3))
model.add(Dense(128))
model.add(Dense(19,activation="softmax"))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
print("Model making done....")
print(model.summary())

import keras
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.utils import plot_model

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

!nvidia-smi

# fitting it into the data
print("Running the model....")
hist = model.fit(xtrain,ytrainf,epochs=500,validation_data=(xtest,ytestf))

print("Saving the model into the disk....")
model.save('medicalspecalty.h5')
print("Model saved into the disk....")

# plotting the figures
print("Plotting the figures....")
plt.figure(figsize=(15,10))
plt.plot(hist.history['accuracy'],c='b',label='train')
plt.plot(hist.history['val_accuracy'],c='r',label='validation')
plt.title("Model Accuracy vs Epochs")
plt.xlabel("EPOCHS")
plt.ylabel("ACCURACY")
plt.legend(loc='lower right')
plt.savefig('accuracy.jpg')


plt.figure(figsize=(15,10))
plt.plot(hist.history['loss'],c='orange',label='train')
plt.plot(hist.history['val_loss'],c='g',label='validation')
plt.title("Model Loss vs Epochs")
plt.xlabel("EPOCHS")
plt.ylabel("LOSS")
plt.legend(loc='upper right')
plt.savefig('loss.jpg')
print("Figures saved in the disk....")

# testing the model
print("Testing the model....")
print("The result obtained is...\n")
model.evaluate(xtest,ytestf)

ypred = model.predict(xtest)
ypred = np.argmax(ypred,axis=1)

print("Classification Report:\n",classification_report(ytest,ypred))
cf = confusion_matrix(ytest,ypred)
print("The confusion matrix is: \n",cf)

plt.figure(figsize=(15,10))
sns.heatmap(cf,annot=True,cmap='crest')
plt.title("Confusion Matrix")
plt.savefig("confusion.jpg")

print(ypred)

sns.heatmap(cf, annot=True)

sns.heatmap(cf, annot=True, linewidth=.75)

from sklearn.utils.multiclass import unique_labels
unique_labels(ytest)

# combining labels with the confusion matrix

def plot(ytest,ypred):
  labels = unique_labels(ytest)
  column = [f'Predicted {label}' for label in labels]
  indices = [f'Actual {label}' for label in labels]
  table = pd.DataFrame(confusion_matrix(ytest,ypred), columns=column, index=indices)

  return table

plot(ytest,ypred)

# same plot as above but in heatmap

def plot2(ytest,ypred):
  labels = unique_labels(ytest)
  column = [f'Predicted {label}' for label in labels]
  indices = [f'Actual {label}' for label in labels]
  table = pd.DataFrame(confusion_matrix(ytest,ypred), columns=column, index=indices)

  return sns.heatmap(table, annot=True, fmt='d', cmap='viridis')

plot2(ytest,ypred)

model = keras.models.load_model('medicalspecalty.h5')

model.summary()

test = "I am having chest pain and also breathing problem"

tokenizer = joblib.load('tokenfinal.pkl')
print("Data preprocessing is over....")

test = tokenizer.texts_to_sequences([test])

test = pad_sequences(test,maxlen=300,padding='post')

predict = model.predict([test])

predict.shape

np.argmax(predict)

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

value[2]

prediction = value[np.argmax(predict)]
print(prediction)

