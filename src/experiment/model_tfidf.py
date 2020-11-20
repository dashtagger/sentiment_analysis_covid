import os
from collections import Counter
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from polyaxon_client.tracking import Experiment

cv = CountVectorizer(strip_accents='ascii', stop_words='english')
vectorizer = TfidfVectorizer()
experiment = Experiment()


def tfidf_train(X_train, X_val, y_train, y_val):
    model = DecisionTreeRegressor()
    # Save the tfidf model
    X_train_transformed = _tokenize(X_train)
    X_val_transformed = _transform_tfidf(X_val)
    # Train, save and evaluate model
    saved_path = experiment.get_outputs_path() + '/tfidf_coronatweetmodel.h5'
    saved_model = model.fit(X_train_transformed, y_train)
    pickle.dump(saved_model, open(saved_path, 'wb'))
    print('model saved')

    return saved_model, X_train_transformed, X_val_transformed

def _tokenize(train_df):
    corpus = train_df.values
    vec_cv = vectorizer.fit_transform(corpus)
    pickel_save = experiment.get_outputs_path() + '/tfidf.pickle'
    pickle.dump(vectorizer, open(pickel_save, "wb"))
    print('tfidf saved')

    return vec_cv    

def _transform_tfidf(df):
    corpus = df.values
    tfidt = vectorizer.transform(corpus)
  
    return tfidt

def tfidf_predict(model,X_test, y_test):
    # model = load_model(model)
    tf1 = pickle.load(open("/Users/sukyee/Desktop/team5/tfidf.pickle", 'rb'))
    X_test_transformed = tf1.transform(X_test.values)
    print(X_test_transformed.shape)

    test_predict = model.predict(X_test_transformed)
    test_score = np.sqrt(mse(y_test, test_predict))
    print("RMSE on test set:", test_score)


