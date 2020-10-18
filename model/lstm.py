import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D
from tensorflow.keras.models import Model
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers

## for plotting
import matplotlib.pyplot as plt
import seaborn as sns

## for processing
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

from time import time
from gensim.models.phrases import Phrases, Phraser
import spacy  # For preprocessing
import en_core_web_sm

## for bag-of-words
from sklearn import feature_extraction, model_selection, naive_bayes, pipeline, manifold, preprocessing

## for explainer
from lime import lime_text

## for word embedding
import gensim
import gensim.downloader as gensim_api

###################################################################################################################

nlp = en_core_web_sm.load()

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


# https://spacy.io/usage/linguistic-features
# Text: The original word text.
# Lemma: The base form of the word.
# POS: The simple UPOS part-of-speech tag.
# Tag: The detailed part-of-speech tag.
# Dep: Syntactic dependency, i.e. the relation between tokens.
# Shape: The word shape – capitalization, punctuation, digits.
# is alpha: Is the token an alpha character?
# is stop: Is the token part of a stop list, i.e. the most common words of the language?

def cleaning(doc):
    # Lemmatizes and removes stopwords
    # doc needs to be a spacy Doc object
    txt = [token.lemma_ for token in doc if not token.is_stop]
    # Word2Vec uses context words to learn the vector representation of a target word,
    # if a sentence is only one or two words long,
    # the benefit for the training is very small
    if len(txt) > 2:
        return ' '.join(txt)


# re - regex sub
brief_cleaning = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in df['text'])

t = time()

txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000, n_threads=-1)]

print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))

from gensim.models.phrases import Phrases, Phraser
df_clean = pd.DataFrame({'text':df['text'],'text_clean':txt,'real':df['real']})
df_clean.dropna(subset=['text_clean'],inplace=True)

## split dataset
df_train, df_test = model_selection.train_test_split(df_clean, test_size=0.3)
## get target
y_train = df_train["real"].values
y_test = df_test["real"].values

tweets_train = df_train['text_clean']
tweets_test = df_test['text_clean']

y = df_train['real']

# tokenize text
max_features = 20000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(tweets_train))
list_tokenized_train = tokenizer.texts_to_sequences(tweets_train)
list_tokenized_test = tokenizer.texts_to_sequences(tweets_test)

# need to make vectors the same size
maxlen = 200
X_train = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_test = pad_sequences(list_tokenized_test, maxlen=maxlen)

# create LSTM NN

inp = Input(shape=(maxlen,))

embed_size = 128
x = Embedding(max_features,embed_size)(inp)

x = LSTM(10, return_sequences=True,name='lstm_layer')(x)
x = GlobalMaxPool1D()(x)
x = Dropout(0.1)(x)
x = Dense(9,activation='relu')(x)

x = Dropout(0.1)(x)
x = Dense(1,activation='sigmoid')(x)

model = Model(inputs=inp,outputs=x)
model.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

# early stopping
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss',verbose=1,patience=3)

# train model
batch_size = 32
epochs = 100
model.fit(X_train,y,batch_size=batch_size,epochs=epochs,validation_split=0.1,callbacks=[early_stop])

# validation
predicted_prob = model.predict(X_test)

#create binary values
predicted = []
for x in predicted_prob:
    if x < .5:
        predicted.append(0)
    elif x >= .5:
        predicted.append(1)


from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,predicted))
print(confusion_matrix(y_test,predicted))

model.save('lstm_model.h5')