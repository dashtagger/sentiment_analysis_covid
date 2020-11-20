import logging
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from .model_selection import Model
from polyaxon_client.tracking import Experiment


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("polyaxon")


def run_experiment(train_path, val_path, network, params):
    
    experiment = Experiment()
    try:
        
        logger.info('Starting experiment...')
        experiment.log_params(network=network)
        experiment.log_params(**params)
        
        logger.info('Importing CLEANED tweets...')
        train_c = pd.read_csv(train_path)
        val_c = pd.read_csv(val_path)
        # TRAIN SETS
        X_train = train_c.tweet
        y_train = train_c.sentiment
        # VAL SETS
        X_val = val_c.tweet
        y_val = val_c.sentiment

        logger.info('Training model...')
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
            
        else:
            model, X_train_pad, X_val_pad = mdl.selection(network, 
                                                          X_train, 
                                                          X_val, 
                                                          y_train, 
                                                          y_val,params)  
                    
            trainPredict = model.predict(X_train_pad)
            trainScore = np.sqrt(mse(y_train, trainPredict[:,0]))
            print('RMSE on train set:', trainScore)
            
            # make predictions     
            valPredict = model.predict(X_val_pad)
            # RMSE metric
            valScore = np.sqrt(mse(y_val, valPredict[:,0]))
            print("RMSE on val set:" ,valScore)


        logger.info('Logging metrics...')
        experiment.log_metrics(trainRMSE=trainScore, valRMSE=valScore)
        
        experiment.succeeded()
        logger.info('Experiment completed')
    
    except Exception as e:
        experiment.failed()
        logger.error(f'Experiment failed: m{str(e)}')
        
        
if __name__ == '__main__':

    arguments = ['--train_path','--val_path', '--network', '--units',
                 '--epochs', '--batch_size', 
                 '--seq_length', '--bidirectional']    

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
             'seq_length': int(args.seq_length),
             'bidirectional': bool(args.bidirectional)
             }  

    run_experiment(args.train_path, args.val_path, args.network, params)
