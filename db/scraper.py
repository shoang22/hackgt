'''
*    Title: GraphicsDrawer source code
*    Author: Smith, J
*    Date: 2011
*    Code version: 2.0
*    Availability: http://www.graphicsdrawer.com
'''

import time
import user
import tweepy
import psycopg2

auth = tweepy.OAuthHandler(user.CONSUMER_KEY, user.CONSUMER_KEY_SECRET)
auth.set_access_token(user.ACCESS_TOKEN, user.ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)


class MyStreamListener(tweepy.StreamListener):

    def __init__(self, time_limit=300):
        self.start_time = time.time()
        self.limit = time_limit
        super(MyStreamListener, self).__init__()

    def on_connect(self):
        print("Connected to Twitter API.")

    def on_status(self, status):

        # Tweet ID
        tweet_id = status.id

        # User ID
        user_id = status.user.id
        # Username
        username = status.user.name

        # Tweet
        if status.truncated == True:
            tweet = status.extended_tweet['full_text']
            hashtags = status.extended_tweet['entities']['hashtags']
        else:
            tweet = status.text
            hashtags = status.entities['hashtags']

        # Read hastags
        hashtags = read_hashtags(hashtags)

        # Retweet count
        retweet_count = status.retweet_count
        # Language
        lang = status.lang

        # If tweet is not a retweet and tweet is in English
        if not hasattr(status, "retweeted_status") and lang == "en":
            # Connect to database
            dbConnect(user_id, username, tweet_id, tweet, retweet_count, hashtags)

        if (time.time() - self.start_time) > self.limit:
            print(time.time(), self.start_time, self.limit)
            return False

    def on_error(self, status_code):
        if status_code == 420:
            # Returning False in on_data disconnects the stream
            return False

# Extract hashtags
def read_hashtags(tag_list):
    hashtags = []
    for tag in tag_list:
        hashtags.append(tag['text'])
    return hashtags

# commands = (# Table 1
#             '''Create Table TwitterUser(User_Id BIGINT PRIMARY KEY, User_Name TEXT);''',
#             # Table 2
#             '''Create Table TwitterTweet(Tweet_Id BIGINT PRIMARY KEY,
#                                          User_Id BIGINT,
#                                          Tweet TEXT,
#                                          Retweet_Count INT,
#                                          CONSTRAINT fk_user
#                                              FOREIGN KEY(User_Id)
#                                                  REFERENCES TwitterUser(User_Id));''',
#             # Table 3
#             '''Create Table TwitterEntity(Id SERIAL PRIMARY KEY,
#                                          Tweet_Id BIGINT,
#                                          Hashtag TEXT,
#                                          CONSTRAINT fk_user
#                                              FOREIGN KEY(Tweet_Id)
#                                                  REFERENCES TwitterTweet(Tweet_Id));''')

# Connection to database server
# need to allow ip address on GCP first - remember to convert to CIDR format with "to" address
conn = psycopg2.connect(host="34.86.177.25", database="postgres", user='postgres', password = 'COVID_type8eat')

# Create cursor to execute SQL commands
cur = conn.cursor()

# Execute SQL commands
# for command in commands:
#     # Create tables
#     cur.execute(command)

# Close communication with server
conn.commit()
cur.close()
conn.close()

# Insert Tweet data into database
def dbConnect(user_id, user_name, tweet_id, tweet, retweet_count, hashtags):
    # need to allow ip address first - remember to convert to CIDR format with "to" address
    conn = psycopg2.connect(host="34.86.177.25", database="postgres", user= 'postgres', password = 'COVID_type8eat')

    cur = conn.cursor()

    # insert user information
    command = '''INSERT INTO TwitterUser (user_id, user_name) VALUES (%s,%s) ON CONFLICT
                 (User_Id) DO NOTHING;'''
    cur.execute(command, (user_id, user_name))

    # insert tweet information
    command = '''INSERT INTO TwitterTweet (tweet_id, user_id, tweet, retweet_count) VALUES (%s,%s,%s,%s);'''
    cur.execute(command, (tweet_id, user_id, tweet, retweet_count))

    # insert entity information
    for i in range(len(hashtags)):
        hashtag = hashtags[i]
        command = '''INSERT INTO TwitterEntity (tweet_id, hashtag) VALUES (%s,%s);'''
        cur.execute(command, (tweet_id, hashtag))

    # Commit changes
    conn.commit()

    # Disconnect
    cur.close()
    conn.close()

myStreamListener = MyStreamListener()
myStream = tweepy.Stream(auth=api.auth, listener=myStreamListener,
                        tweet_mode="extended")
myStream.filter(track=['covid','coronavirus','pandemic','covid19','covid-19'])