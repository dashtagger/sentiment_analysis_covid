import pickle
import math
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error


def evaluation(tweetDFpath, modelfile, tokenizerfile):    
    '''
    params:
        X_test : Series of tweets
        y_test: series of sentiments
        modelfile: your .h5 modelfile
        tokenizerfile: your pickled tokenizer
    
    Returns test set RMSE
    '''
    
    tweetDF = pd.read_csv(tweetDFpath)
    X_test = tweetDF.tweet
    y_test = tweetDF.sentiment
    
    try:
    
        # load sklearn tokenizer and model
        tf1 = pickle.load(open(tokenizerfile, 'rb'))
        X_test_transformed = tf1.transform(X_test)
        tfidfmodel = pickle.load(open(modelfile, 'rb'))
        testPredict = tfidfmodel.predict(X_test_transformed)
        testScore = math.sqrt(mean_squared_error(y_test, testPredict))
        print("RMSE on test set:" ,testScore)


    except:

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


if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser(description='load conditions')
    parser.add_argument('--tweetDFpath', help='tweet csv file for sentiment prediction')
    parser.add_argument('--modelfile', help='model.h5 file')
    parser.add_argument('--tokenizerfile', help='tokenizer.pickle file')

    args = parser.parse_args()
        
    evaluation(args.tweetDFpath, args.modelfile, args.tokenizerfile)