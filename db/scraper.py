import user

from tweepy import Cursor
from tweepy import API
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import pandas as pd

import psycopg

### TWITTER CLIENT ###
class TwitterClient():
    def __init__(self,twitter_param=None): # twitter_user=None is to specify default argument as None
        self.auth = TwitterAuthenticator().authenticate_twitter_app()
        self.twitter_client = API(self.auth,wait_on_rate_limit=True,wait_on_rate_limit_notify=True)

        # Get tweets for another user
        self.twitter_param = twitter_param
        # self.twitter.query = twitter_query

    def get_user_timeline_tweets(self,num_tweets):
        tweets = []
        for tweet in Cursor(self.twitter_client.user_timeline,id=self.twitter_param).items(num_tweets):
            tweets.append(tweet)
        return tweets

    def get_friend_list(self,num_friends):
        friend_list = []
        for friend in Cursor(self.twitter_client.friends).items(num_friends):
            friend_list.append(friend)
        return friend_list

    def get_home_timeline_tweets(self,num_tweets):
        home_timeline_tweets = []
        for tweet in Cursor(self.twitter_client.home_timeline,id=self.twitter_param).items(num_tweets):
            home_timeline_tweets.append(tweet)
        return home_timeline_tweets

    def searchTweets(self,num_tweets):
        searched_tweets = []
        for tweet in Cursor(self.twitter_client.search,q=self.twitter_param).items(num_tweets):
            searched_tweets.append((tweet.created_at,tweet.id,tweet.user.name,tweet.user.screen_name,tweet.user.location,tweet.text))
        return searched_tweets

    # def getStatus(self,):

### TWITTER AUTHENTICATOR ###

# Create class to authenticate for other purposes

class TwitterAuthenticator():
    def authenticate_twitter_app(self):
        auth = OAuthHandler(user.CONSUMER_KEY, user.CONSUMER_KEY_SECRET)
        auth.set_access_token(user.ACCESS_TOKEN, user.ACCESS_TOKEN_SECRET)
        return auth

### TWITTER STREAMER ###
class TwitterStreamer():
    """
    Class for streaming and processing live tweets.
    """
    def __init__(self):
        self.twitter_authenticator = TwitterAuthenticator()

    def stream_tweets(self, fetched_tweets_filename, hash_tag_list):
        # This handles Twitter authetification and the connection to Twitter Streaming API
        listener = TwitterListener(fetched_tweets_filename)
        auth = self.twitter_authenticator.authenticate_twitter_app()
        stream = Stream(auth, listener)

        # This line filter Twitter Streams to capture data by the keywords:
        stream.filter(track=hash_tag_list)

### TWITTER STREAM LISTENER ###
class TwitterListener(StreamListener):
    """
    This is a basic listener that just prints received tweets to stdout.
    """

    def __init__(self, fetched_tweets_filename):
        self.fetched_tweets_filename = fetched_tweets_filename

    def on_data(self, data):
        try:
            print(data)
            with open(self.fetched_tweets_filename, 'a') as tf:
                tf.write(data)
            return True
        except BaseException as e:
            print("Error on_data %s" % str(e))
        return True

    # Error to catch rate limit violation
    def on_error(self, status):
        if status == 420:
            # return false on_data method in case tate limit occurs
            return False
        print(status)


if __name__ == '__main__':
    # Authenticate using config.py and connect to Twitter Streaming API.
    # text = 'hate_keywords.txt'
    # with open(text, encoding='utf8') as fl:
        # file_contents = [x.rstrip() for x in fl]

    # query = '(covid OR rona OR virus OR kung flu OR pandemic)' \
            #'(Sputnik-V OR Pfizer OR Modena OR AstraZeneca OR Oxford OR immunity ' \
            #'OR vaxx OR vaccin OR trial OR CDC OR FDA OR Regeneron) lang: en'

    # Authenticate using config.py and connect to Twitter Streaming API
    hash_tag_list = ['covid','coronavirus','pandemic','covid19','covid-19']
    fetched_tweets_filename = "tweets.txt"

    # items = 3
    # file_name = 'data.csv'
    # twitter_client = TwitterClient(query)
    # df = pd.DataFrame(twitter_client.searchTweets(items))
    # df.to_csv(file_name,encoding='utf-8',index=False)
    twitter_streamer = TwitterStreamer()
    twitter_streamer.stream_tweets(fetched_tweets_filename, hash_tag_list)

