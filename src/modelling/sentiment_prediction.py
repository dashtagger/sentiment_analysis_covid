import pickle
import math
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error



def predict_tweet(tweetDFpath, modelfile, tokenizerfile):
    '''
    params:
        tweetDFpath : path to tweets.csv
        modelfile: your .h5 modelfile
        tokenizerfile: your pickled tokenizer
    
    Returns csv file of individual sentiment score for each tweet
    '''
    tweetDF = pd.read_csv(tweetDFpath)
    tweet = tweetDF.tweet  # to predict a tweet
          
    try:
        
        # load sklearn tokenizer and model
        tf1 = pickle.load(open(tokenizerfile, 'rb'))
        X_test_transformed = tf1.transform(tweet)
        tfidfmodel = pickle.load(open(modelfile, 'rb'))
        testPredict = tfidfmodel.predict(X_test_transformed)
        sentimentDF = pd.DataFrame({'sentiment': testPredict})
        output = pd.concat([tweetDF, sentimentDF], axis=1)

        
    except:
        # load keras tokenizer and model 
        with open(tokenizerfile, 'rb') as handle:
            tokenizer_obj = pickle.load(handle, encoding='iso-8859-1')
        # tokenize    
        tweet_tokens = tokenizer_obj.texts_to_sequences(tweet)
        # pad sequences
        tweet_pad = pad_sequences(tweet_tokens, 
                                maxlen = 9, 
                                padding='post')
        tweet_pad = tweet_pad.astype(int)
        # make predictions   
        # load model
        model = load_model(modelfile)
        model._make_predict_function()  
        testPredict = model.predict(tweet_pad)
        testPredict = testPredict.ravel()
        sentimentDF = pd.DataFrame({'sentiment': testPredict})
        output = pd.concat([tweetDF, sentimentDF], axis=1)        
        
    return(output.to_csv("SentimentPrediction.csv", index=False))
    

if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser(description='load conditions')
    parser.add_argument('--tweetDFpath', help='tweet csv file for sentiment prediction')
    parser.add_argument('--modelfile', help='model.h5 file')
    parser.add_argument('--tokenizerfile', help='tokenizer.pickle file')

    args = parser.parse_args()
        
    predict_tweet(args.tweetDFpath, args.modelfile, args.tokenizerfile)