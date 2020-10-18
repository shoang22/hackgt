from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import altair as alt
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D
from tensorflow.keras.models import Model
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers

import streamlit as st
import sqlalchemy
import tensorflow as tf
import pandas as pd
from pandas import DataFrame
import seaborn as sns

## for processing
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
import numpy as np

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
from gensim.models.phrases import Phrases, Phraser

#####################################################################################################################

username = 'postgres'  # DB username
password = 'COVID_type8eat'  # DB password
host = '34.86.177.25'  # Public IP address for your instance
port = '5432'
database = 'postgres'  # Name of database ('postgres' by default)

db_url = 'postgresql+psycopg2://{}:{}@{}:{}/{}'.format(
    username, password, host, port, database)

engine = sqlalchemy.create_engine(db_url)

print("Connecting")

conn = engine.connect()
print("Connected")

testquery = "select tweet,retweet_count from twittertweet;"
result = conn.execute(testquery)
df = DataFrame(result.fetchall())
df.columns = result.keys()
print(df)
#result_as_list = result.fetchall()
#for row in result_as_list:
    #print(row)

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
brief_cleaning = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in df['tweet'])

t = time()

txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000, n_threads=-1)]

print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))

####################################################################################################

df_clean = pd.DataFrame({'tweet':df['tweet'],'text_clean':txt,'retweets':df['retweet_count']})
df_clean.dropna(subset=['text_clean'],inplace=True)

max_features = 20000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(df_clean['text_clean']))
list_tokenized = tokenizer.texts_to_sequences(df_clean['text_clean'])

maxlen = 200
tweets = pad_sequences(list_tokenized, maxlen=maxlen)

model = tf.keras.models.load_model('./model/gthack_model.h5')
predictions = model.predict(tweets)

predicted = [] #create binary values

for x in predictions:
    if x < .5:
        predicted.append(0)
    elif x >= .5:
        predicted.append(1)

df_clean['predicted_sentiment'] = predicted

st.subheader('Retweet Count by Sentiment')


chart = alt.Chart(df_clean).mark_bar().encode(
    alt.X("retweets", bin=True),
    y='count()',
    color = 'predicted_sentiment'
).interactive()
st.altair_chart(chart)

st.subheader('Tweet Count by Sentiment')

chart2 = alt.Chart(df_clean).mark_bar().encode(
    alt.X("tweet", bin=True),
    y='count()',
    color = 'predicted_sentiment'
).interactive()
st.altair_chart(chart2)

#word_count = DataFrame(df_clean['text_clean'].str.split(expand=True).stack().value_counts()[:10])

#wcdf = pd.DataFrame(df_clean['text_clean'].str.split(expand=True).stack().value_counts()[:10],columns=['word','frequency'])

#chart3 = alt.Chart(wcdf).mark_bar().encode(
#     alt.X('frequency', bin=True),
#     y='count()',
#     color = 'predicted_sentiment'
# ).interactive()
# st.altair_chart(chart3)


conn.close()
print("Connection Closed")