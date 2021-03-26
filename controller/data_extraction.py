import tweepy
import csv

CONSUMER_KEY = ""
CONSUMER_SECRET = ""
ACCESS_TOKEN = ""
ACCESS_TOKEN_SECRET = ""

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
csvFile = open('DataSet_tweets_ecuador.csv', 'a')

csvWriter = csv.writer(csvFile)


for tweet in tweepy.Cursor(api.search,q="#Covid_19ec",count=100,
                           lang="es",
                           since="2017-04-03").items(10):

 print(tweet.created_at, format(str(tweet.text).encode('utf-8').decode('utf-8')))
 csvWriter.writerow([tweet.created_at, tweet.text])