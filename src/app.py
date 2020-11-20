

import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import re
from PIL import Image

from modules.wordcloud import Wordcloud_gen
from modules.tweet_module import Twitter_api
from datapipeline.clean_data import clean_pipeline
from modelling.model_selection import Model
from modules.utils import *


# App starts here

st.title('Sentiment Analysis for Corona Virus in Singapore')
# initialise defaults

initialised = False
datapath = './data/'


# Side bar of modules

module_selectbox = st.sidebar.selectbox(
    'Tools',
    ('Sentiment Analysis', 'Data Analysis', 'Sentiment Generator', 'Model Analysis', 'Project Architecture'))
filename = file_selector()

df_default = pd.read_csv(filename)

model_selectbox = st.sidebar.selectbox(
    'Models',
    ('TFidf', 'GRU','LSTM'))   

# Module selection

if module_selectbox == 'Sentiment Analysis':
    st.subheader("Sentiment Trends")
    st.write("Select Filters")

    if not check_sentiment_exists(df_default):
        st.write("csv doesn't have sentiment and created_date columns")
    
    sortby_selection = st.selectbox('Display by', ('Date', 'Hour', 'Seconds'))
    if sortby_selection == 'Date':
        df_default = pd.read_csv(filename)
        df_sentiment = get_sorted_by_datetime(df_default)
        df_sentiment_dateday = groupby_sentiment_day(df_sentiment)
        df_selected = df_sentiment_dateday
        column = 'created_dateday'
        reload_sentiment_analysis_ui(df_selected, column)
    elif sortby_selection == 'Hour':
        df_default = pd.read_csv(filename)
        df_sentiment = get_sorted_by_datetime(df_default)
        df_sentiment_datehour = groupby_sentiment_hour(df_sentiment)
        df_selected = df_sentiment_datehour
        column = 'created_datehour'
        reload_sentiment_analysis_ui(df_selected, column)
    elif sortby_selection == 'Seconds':
        df_default = pd.read_csv(filename)
        df_sentiment = get_sorted_by_datetime(df_default)
        df_sentiment_seconds = groupby_sentiment_seconds(df_sentiment)
        df_selected = df_sentiment_seconds
        column = 'created_date'
        reload_sentiment_analysis_ui(df_selected, column)


elif module_selectbox == 'Data Analysis':
    st.subheader("Word cloud for words used in Model training")
    model_wc_pos_image_path ='./data/pos_wc.png'
    model_wc_neg_image_path ='./data/neg_wc.png'
    model_wc_pos_image = Image.open(model_wc_pos_image_path)
    model_wc_neg_image = Image.open(model_wc_neg_image_path)
    st.image(model_wc_pos_image, caption='', use_column_width=True)
    st.image(model_wc_neg_image, caption='', use_column_width=True)
    
    st.subheader("Word cloud analysis of selected dataframe")
    savepath = './data/generated_wordcloud.png'
    st.write("Sentiment Dataframe Preview")
    st.write(pd.read_csv(filename))
    generate_wordcloud = st.button('Generate Word Cloud')
    if generate_wordcloud:
        with st.spinner("Generating wordcloud..."):
            wc = Wordcloud_gen(datapath=filename, uniquewords='./data/unique_word.txt')
            wc.pipeline(savepath)
            st.image(savepath, caption='', use_column_width=True)
        st.success('Done!')

elif module_selectbox == 'Sentiment Generator':
    st.subheader('Twitter Data Generator')
    st.write("This module allows users to generate their own sentiment with given filters")
    input_keywords = st.text_input("Enter targeted keywords", "#corona OR #covid")
    country_dict = {'Singapore':'1.3521,103.8198,30km','London':'51.509865,-0.118092,70km','New York':'40.730610,-73.935242,530km'}
    select_country =  st.selectbox('Select Country', ('Singapore','London','New York'))
    input_geocode = country_dict[select_country]
    input_filename = st.text_input("Enter output csv filename", "user_tweets.csv")
    input_num_tweets = st.number_input("Enter number of tweets to download", 100)
    twitter_savefile = './data/' + input_filename

    gen_twitter_button = st.button('Generate csv')
    tm = Twitter_api()
    model = Model()
    if gen_twitter_button:
        model_selected = model_selectbox 
        st.write("Model selected: " + model_selected)
        st.write("Loaction: " + input_geocode)
        st.write("Number of tweets to generate: " + str(input_num_tweets))
        with st.spinner("Pulling Twitter Sentiment..."):
            df_raw_tweets = tm.search(savepath=twitter_savefile, query=input_keywords, searchcount=input_num_tweets, geocode=input_geocode)
            st.write("Raw tweets")
            st.write(df_raw_tweets)
            st.write("Cleaned tweets")
            df_raw_tweets_cleaned = clean_pipeline(df_raw_tweets,dropdup=False)
            st.write(df_raw_tweets_cleaned.reset_index().drop(columns=['index']))
            modelpath, tokenpath = get_model_path(model_selectbox)
            output = model.predict_tweet(df_raw_tweets_cleaned.reset_index(), modelpath, tokenpath)
            output = output.drop(columns=['index'])
            st.write("Sentiments")
            st.write(output)
            output.to_csv(datapath+input_filename, index=False)
        st.success('Done! Refresh page and check left side for generated csv file ')

elif module_selectbox == 'Model Analysis':
    st.subheader('Model Analysis')
    st.write("Model Training Summary")
    df_rsme= pd.DataFrame({"Train RMSE":[0.00067516, 0.071759, 0.072704],
                        "Validation RSME":[0.125097,0.1168388,0.1168379],
                        "Test RSME":[0.1359301,0.1283949,0.1262522]})
    df_rsme.index = ['tfidf', 'GRU', 'LSTM'] 
    st.write(df_rsme)
    sentiment_TFIDF = pd.read_csv('./data/sg_tweets_TFIDF.csv')
    sentiment_GRU = pd.read_csv('./data/sg_tweets_GRU.csv')
    sentiment_LSTM = pd.read_csv('./data/sg_tweets_LSTM.csv')
    sentiment_TFIDF_only = pd.DataFrame()
    sentiment_TFIDF_only['sentiment_TFIDF'] = sentiment_TFIDF.sentiment
    sentiment_GRU_only = pd.DataFrame()
    sentiment_GRU_only['sentiment_GRU'] = sentiment_GRU.sentiment
    sentiment_LSTM_only = pd.DataFrame()
    sentiment_LSTM_only['sentiment_LSTM'] = sentiment_LSTM.sentiment
    df_compare = pd.concat([sentiment_TFIDF['created_date'],sentiment_TFIDF_only, sentiment_GRU_only, sentiment_LSTM_only ],axis=1)
    st.write(df_compare)
    df_compare = groupby_sentiment_hour(df_compare)
    column = 'created_datehour'
    reload_sentiment_analysis_ui(df_compare, column)
    #model = Model()

elif module_selectbox == 'Project Architecture':
    st.subheader('Project Architecture')
    datapipeline_image_path ='./data/Project_design.jpg'
    datapipeline_image = Image.open(datapipeline_image_path)

    st.image(datapipeline_image, caption='Project Architecture', use_column_width=True)
