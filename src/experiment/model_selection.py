import pickle
import math
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error


class Model():
    
    def __init__(self):
        pass
    
    
    def selection(self, network, X_train, X_val, y_train, y_val, params):
        
        if network == 'tfidf':
            from .model_tfidf import tfidf_train
            model_type = tfidf_train(X_train, X_val, y_train, y_val)
            return model_type
            
        elif network =='lstm':
            from .model_LSTM import lstm_train
            model_type = lstm_train(X_train, X_val, y_train, y_val, params)
            return model_type
            
        elif network == 'gru':
            from .model_GRU import gru_train
            model_type = gru_train(X_train, X_val, y_train, y_val, params)
            return model_type
            
        else:
            raise ValueError('Please select only \'bow\', \'lstm\' or \'gru\' for model_type.')
                    

    def evaluation(self, X_test, y_test, modelfile, tokenizerfile):    
        '''
        params:
            X_test : Series of tweets
            y_test: series of sentiments
            modelfile: your .h5 modelfile
            tokenizerfile: your pickled tokenizer
        
        Returns test set RMSE
        '''
        
        # load tokenizer    
        with open(tokenizerfile, 'rb') as handle:
            tokenizer_obj = pickle.load(handle, encoding='iso-8859-1')
                    
        # tokenize    
        X_test_tokens = tokenizer_obj.texts_to_sequences(X_test)
        
        # pad sequences
        X_test_pad = pad_sequences(X_test_tokens, 
                                    maxlen = 9, 
                                    padding='post')
        X_test_pad = X_test_pad.astype(int)
        
        
        # make predictions   
        # load model
        model = load_model(modelfile)
        model._make_predict_function()  

        testPredict = model.predict(X_test_pad)
        # RMSE metric
        testScore = math.sqrt(mean_squared_error(y_test, testPredict[:,0]))
        print("RMSE on test set:" ,testScore)
        
        return testScore
    
    
    def predict_tweet(self, tweetDFpath, modelfile, tokenizerfile):
        '''
        params:
            tweetDFpath : path to tweets.csv
            modelfile: your .h5 modelfile
            tokenizerfile: your pickled tokenizer
        
        Returns individual sentiment score for each tweet
        '''
        tweetDF = pd.read_csv(tweetDFpath)
        tweet = tweetDF.tweet  # to predict a tweet
        
        try:
            
            tf1 = pickle.load(open(tokenizerfile, 'rb'))
            X_test_transformed = tf1.transform(tweet)
            model = pickle.load(open(modelfile, 'rb'))
            testPredict = model.predict(X_test_transformed)
            test_scores = testPredict.tolist()

        except:
            
            # load keras tokenizer    
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
            test_scores = testPredict.tolist()
        
        return(test_scores)