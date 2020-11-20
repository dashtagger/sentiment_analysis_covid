import logging
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from datapipeline.loader import Loader
from modelling.model_selection import Model


def run_prediction(train_path, val_path, network, params):
    
        print('Starting experiment...')

        print('Importing CLEANED tweets...')
        # loading 
        train_c = pd.read_csv(train_path)
        val_c = pd.read_csv(val_path)
        
        # TRAIN SETS
        X_train = train_c.tweet
        y_train = train_c.sentiment
        
        # VAL SETS
        X_val = val_c.tweet
        y_val = val_c.sentiment
        
        tokenizer_obj = Tokenizer()
        tokenizer_obj.fit_on_texts(X_train)
    
        print('Training model...')
        mdl = Model()    


        if network =='tfidf':
            model, X_train_transformed, X_val_transformed = mdl.selection(network, 
                                                                          X_train, 
                                                                          X_val, 
                                                                          y_train, 
                                                                          y_val,
                                                                          params)
            train_predict = model.predict(X_train_transformed)
            trainScore = np.sqrt(mse(y_train, train_predict))
            print('RMSE on train set:', trainScore)

            val_predict = model.predict(X_val_transformed)
            valScore = np.sqrt(mse(y_val, val_predict))
            print("RMSE on val set:", valScore)
            return model

            
        else:    
            model, X_train_pad, X_val_pad = mdl.selection(network, 
                                                        X_train, 
                                                        X_val, 
                                                        y_train, 
                                                        y_val,
                                                        params)        
                        
            trainPredict = model.predict(X_train_pad)
            trainScore = np.sqrt(mse(y_train, trainPredict[:,0]))
            print('RMSE on train set:', trainScore)
            
            # make predictions     
            valPredict = model.predict(X_val_pad)
            # RMSE metric
            valScore = np.sqrt(mse(y_val, valPredict[:,0]))
            print("RMSE on val set:" ,valScore)
            return model

        
if __name__ == '__main__':

    arguments = ['--train_path','--val_path', '--network', '--units',
                 '--epochs', '--batch_size', '--bidirectional']    

    parser = argparse.ArgumentParser()

    for a in arguments:
        parser.add_argument(a)
        
    args, unknown = parser.parse_known_args()
    
    if args.network == 'tfidf':
         params = {}
         
    else:
        params = {
             'units' : int(args.units),
             'epochs' : int(args.epochs),
             'batch_size': int(args.batch_size),
             'bidirectional': bool(args.bidirectional)
             }  

    run_prediction(args.train_path, args.val_path, args.network, params)
    
    