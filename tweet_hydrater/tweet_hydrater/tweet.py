import csv
import json
import twarc

t_ids = []
t_sentiments = []
t = twarc.Twarc()
pos = 0
neg = 0
count = 0

with open('corona_tweets_212.csv', 'r') as csvfile:
    file_reader = csv.reader(csvfile)

    for line in file_reader:
        if( pos < 20000 or neg < 20000):
            if ("-" in line[1]):
                neg+= 1
            else:
                pos+= 1
            t_ids.append(line[0])
            t_sentiments.append(line[1])

def ids():
    for id in t_ids:
        yield id

with open('coronavirus_tweets.JSON', 'w') as output:
    for tweet in t.hydrate(ids()):
        tweet['sentiment'] = t_sentiments[count]
        count = count + 1
        data = json.dumps(tweet, sort_keys=True)
        output.write(data)