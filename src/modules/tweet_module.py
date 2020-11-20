import tweepy
import os
from dotenv import load_dotenv, find_dotenv
import pandas as pd

# find .env automagically by walking up directories until it's found
dotenv_path = find_dotenv()

# load up the entries as environment variables
load_dotenv(dotenv_path)

class Twitter_api():
    def __init__(self):
        load_dotenv(dotenv_path)

        # Getting API keys from the environment variables
        try:
            consumer_key = os.environ.get("CONSUMER_KEY")
            consumer_secret = os.environ.get("CONSUMER_SECRET")
            access_token = os.environ.get("ACCESS_TOKEN")
            access_token_secret = os.environ.get("ACCESS_TOKEN_SECRET")
        except:
            raise ValueError('Provide the correct tokens and keys')

        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)

        self.api = tweepy.API(auth, wait_on_rate_limit=True,
            wait_on_rate_limit_notify=True)
    
    def search(self, savepath=None, query="(#corona OR #covid) AND #singapore", searchcount=100, geocode="1.3521,103.8198,30km"):
        """ Return the twitter search result from the query, and extract the 
        relevant data into a pandas dataframe
        
        Args:
            savepath (str, optional): The path to save the dataframe as
                csv. If None, the dataframe will not be saved Defaults to None.
            query (str, optional): query for twitter search.
                Defaults to "(#corona OR #covid) ".
        
        Returns:
            [type]: [description]
        """
        _max_queries = 100  # arbitrarily chosen value
        result = tweet_batch = self.api.search(q=query,
                                    lang="en",
                                    geocode=geocode,
                                    count=searchcount,
                                    result_type='recent')
        ct = 1
        while len(result) < searchcount and ct < _max_queries:
            tweet_batch = self.api.search(q=query,
                                    lang="en",
                                    geocode=geocode,
                                    count=searchcount - len(result),
                                    result_type='recent',
                                    max_id=tweet_batch.max_id)
            result.extend(tweet_batch)
            ct += 1
        # Initialised the list
        tweet_id = []
        text = []
        created_date = []

        # Extract relevant info to append to the list
        for tweet in result:
            tweet_id.append(tweet.id_str)
            text.append(tweet.text)
            created_date.append(tweet.created_at)
        
        # Pack all the list into a dataframe
        data = {'created_date': created_date, 'tweet': text, 'tweet_id': tweet_id}
        df_tweet = pd.DataFrame(data)

        # Save the dataframe (if savepath is provided)
        if savepath is not None:
            df_tweet.to_csv(savepath, index=False)

        return df_tweet
