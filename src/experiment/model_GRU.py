import numpy as np
import math
import pickle
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dropout, Embedding, Dense, GRU, Bidirectional)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from polyaxon_client.tracking import Experiment

  
def gru_train(X_train, X_val, y_train, y_val, params):
    
    tokenizer_obj = Tokenizer()
    tokenizer_obj.fit_on_texts(X_train)
    
    # save tokenizer
    experiment = Experiment()
    save_tokenizer = experiment.get_outputs_path() + '/GRU_tokenizer.pickle'
    with open(save_tokenizer, 'wb') as handle:
        pickle.dump(tokenizer_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # for pad sequences
    seq_length = params['seq_length']
    #seq_length = int(max([len(s.split()) for s in X_train]))
    
    # vocab size
    vocab_size = len(tokenizer_obj.word_index) + 1
    
    # tokenize    
    X_train_tokens = tokenizer_obj.texts_to_sequences(X_train)
    X_val_tokens = tokenizer_obj.texts_to_sequences(X_val)
    
    # pad sequences
    X_train_pad = pad_sequences(X_train_tokens, 
                                maxlen = seq_length, 
                                padding='post')
    X_train_pad = X_train_pad.astype(int)

    X_val_pad = pad_sequences(X_val_tokens,
                               maxlen = seq_length, 
                               padding ='post')
    X_val_pad = X_val_pad.astype(int)
    
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, 
                        output_dim=100, 
                        input_length=seq_length))

    if params['bidirectional'] == True:
        model.add(Bidirectional(GRU(units=params['units'])))
    else:
        model.add(GRU(units=params['units']))    
        
    model.add(Dense(1))
    
    model.compile(loss='mean_squared_error', optimizer='Adam')

    experiment = Experiment()
    saved_path = experiment.get_outputs_path() + '/GRU_coronatweetmodel.h5'
    es = EarlyStopping(monitor='val_loss', mode='min', patience=3)
    cp = ModelCheckpoint(saved_path, monitor='val_loss', 
                        mode='max',
                        save_best_only=True)

    model.fit(X_train_pad, y_train,
            epochs = params['epochs'],
            batch_size = params['batch_size'],
            validation_data = (X_val_pad, y_val),
            callbacks=[es,cp])
        
    return model, X_train_pad, X_val_pad