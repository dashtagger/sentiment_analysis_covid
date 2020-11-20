import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import words
import nltk

nltk.download('punkt')
nltk.download('words')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger') 

stop_word = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
english_word = set(words.words())

def clean_pipeline(df_tweet, min_length=33, dropdup=True):
    """ Data cleaning pipeline for covid tweets dataset.
        It also remove cleaned tweets that are shorter than
        the minimum length.
        It also removes duplicated tweets and created_date is change to
        date_time format

    Args:
        df_tweet (pandas df): dataframe containing covid tweet
        min_length (int, optional): Minimum length of cleaned tweet. 
                                    Defaults to 33.

    Returns:
        df_tweet (pandas df): cleaned covid dataset
    """

    # Remove username from tweet
    df_tweet['tweet'] = df_tweet.tweet.apply(remove_username)
    # Remove emoticon from tweet
    df_tweet['tweet'] = df_tweet.tweet.apply(remove_emoticon)
    # Remove url from tweet
    df_tweet['tweet'] = df_tweet.tweet.apply(remove_url)
    # Remove html from tweet
    df_tweet['tweet'] = df_tweet.tweet.apply(remove_html)
    # Remove stop words from tweet
    df_tweet['tweet'] = df_tweet.tweet.apply(remove_stop_word)
    # Perform lemmatisation on tweet
    df_tweet['tweet'] = df_tweet.tweet.apply(lemmat)
    # Remove unknown words from tweet
    df_tweet['tweet'] = df_tweet.tweet.apply(remove_unknown_word)
    # Removing tweet below minimum length
    df_tweet = remove_short_tweet(df_tweet, min_length)
    # Removing duplicates
    if dropdup:
        df_tweet = remove_dup(df_tweet)
    # Datetime parser
    df_tweet.loc[:,'created_date'] = pd.to_datetime(df_tweet['created_date'])

    return df_tweet


# Secondary function (to be used in pd.df.apply)

def remove_username(entry):
    """ Remove username from tweet (use in pd.df.apply)
    
    Args:
        entry (entry of pandas df): an entry of the tweet column of the
        tweet dataframe
    
    Returns:
        output: tweet with @username remove
    """
    pattern = r'@.+?\s'
    output = re.sub(pattern, '', entry).strip()
    return output


def remove_emoticon(entry):
    """ Remove emoticon from tweet (use in pd.df.apply)
    
    Args:
        entry (entry of pandas df): an entry of the tweet column of the
        tweet dataframe
    
    Returns:
        output: tweet with emoticon remove
    """
    output = entry.encode('ascii', 'ignore').decode('ascii')
    return output

def remove_url(entry):
    """ Remove url from tweet (use in pd.df.apply)
    
    Args:
        entry (entry of pandas df): an entry of the tweet column of the
        tweet dataframe
    
    Returns:
        output: tweet with url remove
    """
    pattern = r'http\S+'
    output = re.sub(pattern, '', entry).strip()
    return output

def remove_html(entry):
    """ Remove html tags from tweet (use in pd.df.apply)
    
    Args:
        entry (entry of pandas df): an entry of the tweet column of the
        tweet dataframe
    
    Returns:
        output: tweet with html tags remove
    """
    pattern = r'<.+?>'
    output = re.sub(pattern, '', entry).strip()
    return output

def remove_stop_word(entry):
    """ Remove stop words from tweet (use in pd.df.apply)
    
    Args:
        entry (entry of pandas df): an entry of the tweet column of the
        tweet dataframe
    
    Returns:
        output: tweet with stop words remove
    """
    output = [w for w in word_tokenize(entry)
              if w.lower() not in stop_word]
    output = ' '.join(output)
    return output

def lemmat(entry):
    """ Perform Lemmatisation on tweet (use in pd.df.apply)
    
    Args:
        entry (entry of pandas df): an entry of the tweet column of the
        tweet dataframe
    
    Returns:
        output: Lemmatised tweets
    """
    tokens = [word for word in word_tokenize(entry.lower())
              if (word.isalpha() or word.isnumeric())]
    tags = nltk.pos_tag_sents([tokens])
    output = []
    for i,tk in enumerate(tokens):
        tag = tags[0][i][1][0]
        try:
            word = lemmatizer.lemmatize(tk, pos=tag.lower())
        except KeyError:
            word = tk
        output.append(word)
    output = ' '.join(output)
    return output

def remove_unknown_word(entry):
    """ Remove unknown words from tweet (use in pd.df.apply)
    
    Args:
        entry (entry of pandas df): an entry of the tweet column of the
        tweet dataframe
    
    Returns:
        output: tweet with unknown words remove
    """
    output = [w for w in word_tokenize(entry)
              if w.lower() in english_word]
    output = ' '.join(output)
    return output

def remove_short_tweet(df_tweet, min_length):
    """ Remove tweets that are too short
    
    Args:
        df_tweet (pandas df): dataframe containing covid tweet
        min_length (int): Minimum length of cleaned tweet. 
    
    Returns:
        df_tweet (pandas df): dataframe with short tweets removed
    """
    long_tweet_logi = df_tweet['tweet'].str.len() > min_length
    df_tweet = df_tweet[long_tweet_logi]
    return df_tweet

def remove_dup(df_tweet):
    """ Remove dupicated tweets (based on the 'tweet' text column)
        For duplicated tweets, it will keep the first tweeet
    Args:
        df_tweet (pandas df): dataframe containing covid tweet

    Returns:
        df_tweet (pandas df): dataframe with duplicated tweets removed
    """
    df_tweet = df_tweet.drop_duplicates(subset='tweet', keep='first')
    return df_tweet