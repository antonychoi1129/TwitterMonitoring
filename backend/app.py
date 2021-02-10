import sys
import os
import random
from flask import Blueprint, request, Flask
from flask_jsonpify import jsonpify
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow import keras
import credentials
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
import tweepy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import pandas as pd
import itertools
import math
import base64
import psycopg2
import datetime
import time
import re
import nltk

nltk.download('punkt')
nltk.download('stopwords')
app = Flask(__name__)

# Twitter credentials
auth = tweepy.OAuthHandler(credentials.API_KEY, credentials.API_SECRET_KEY)
auth.set_access_token(credentials.ACCESS_TOKEN, credentials.ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

import pickle
#load vectorizer
from_disk = pickle.load(open("./vectorizer.pkl", "rb"))
vectorizer = TextVectorization.from_config(from_disk['config'])
# You have to call `adapt` with some dummy data (BUG in Keras)
vectorizer.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
vectorizer.set_weights(from_disk['weights'])
#load Model
model = keras.models.load_model('./my_model')
string_input = keras.Input(shape=(1,), dtype="string")
x = vectorizer(string_input)
preds = model(x)
end_to_end_model = keras.Model(string_input, preds)


# def scraptweets(search_words, date_since, numTweets):
def scraptweets(search_words):
    numTweets = 200
    df_tweets = pd.DataFrame(columns=['created_at', 'text'])
    tweets = tweepy.Cursor(api.search,
                q=search_words,
                lang="en").items(numTweets)

    tweet_list = [tweet for tweet in tweets]
    for tweet in tweet_list:
        created_at = tweet.created_at
        text = tweet.text
        ith_tweet = [created_at, text]
        df_tweets.loc[len(df_tweets)] = ith_tweet
    # Convert UTC into PDT
    df_tweets['created_at'] = pd.to_datetime(df_tweets['created_at']).apply(
        lambda x: str(x - datetime.timedelta(hours=7)))
    return df_tweets

def clean_tweet(text):
    #Clean tweet text by removing links and special characters
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())

def deEmojify(text):
    #Strip all non-ASCII characters to remove emoji
    if text:
        return text.encode('ascii', 'ignore').decode('ascii')
    else:
        return None

def textClassification(df_tweets):
    polarity = []
    for tweet in df_tweets['text']:
        tweet = clean_tweet(tweet)
        tweet = deEmojify(tweet)
        probabilities = end_to_end_model.predict(
            [[tweet]]
        )
        # index (0,neg) (1,neu) (2,pos)
        class_index = np.argmax(probabilities[0])
        if class_index == 2:
            polarity.append(1)
        elif class_index == 0:
            polarity.append(-1)
        else:
            polarity.append(0)
    df_tweets['polarity'] = polarity
    return df_tweets

@app.route('/analytics', methods=['POST'])
def analyze():
    df_tweets = scraptweets(request.form['search_words'])
    df_tweets = textClassification(df_tweets)
    df_tweets = df_tweets.values.tolist()
    JSONP_data = jsonpify(df_tweets)
    return JSONP_data






# app.register_blueprint(api, url_prefix='/api')

if __name__ == '__main__':
    app.run(debug=True)
