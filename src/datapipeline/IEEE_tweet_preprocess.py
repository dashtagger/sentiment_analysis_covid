import re
import pandas as pd

# datapath = 'coronasample1.txt'

def preprocess_tweet(datapath):
    """ Take in the data from IEEE tweet dataset, extract tweet and id

    Args:
        datapath (str): datapath to the tweet file

    Returns:
        df (pandas df): 3 columns in dataframe
                        -'created_date': The date of the tweet created
                        -'tweet': The tweet full text
                        -'tweet_id': The id of the tweet
    """
    with open(datapath, 'rt', encoding="utf8") as f:
        data = f.read()

    # Split the data into separate tweet
    data = data.split('\n')
    # Remove artifects that are too short
    data = [entry for entry in data if len(entry) > 20]
    # Define pattern for different field
    date_pattern = r'"created_at":".+?","'
    tweet_pattern = r',"full_text":".+?"id_str":".+?","'
    id_pattern = r'"id_str":".+?","'

    created_date = []
    tweet = []
    tweet_id = []

    #Extract date, id, and tweet text
    for i,t in enumerate(data):
        temp_date = re.findall(date_pattern, t)[0][14:-3]
        temp_tweet = re.findall(tweet_pattern, t)[0]
        temp_id = re.findall(id_pattern, temp_tweet)[0][10:-3]
        temp_tweet = temp_tweet.split(',"')[1][14:-1]

        created_date.append(temp_date)
        tweet.append(temp_tweet)
        tweet_id.append(temp_id)

    # Pack extracted data into datafram
    df = pd.DataFrame(data={'created_date': created_date,
                                'tweet': tweet,
                                'tweet_id': tweet_id})

    return df

def get_sentiment(df_tweet, df_sentiment):
    """ Merge the tweet dataframe from preprocess_tweet() with the
    sentiment labels from IEEE
    
    Args:
        df_tweet (pandas df): the output from preprocess_tweet()
        df_sentiment (pandas df): the sentiment labels with tweet id
    
    Returns:
        [type]: [description]
    """
    df_tweet['tweet_id'] = df_tweet.tweet_id.astype('int64')
    df_clean = pd.merge(df_tweet, df_sentiment, how='inner')

    return df_clean
