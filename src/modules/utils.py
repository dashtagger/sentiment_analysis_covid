import os
import streamlit as st
import re

def file_selector(folder_path='./data/', file_filter=r'\S+.csv'):
    '''returns files in given folder path'''
    filenames = [f for f in os.listdir(folder_path) if re.match(file_filter , f)]
    selected_filename = st.sidebar.selectbox('Select a file (from data folder)', filenames)
    return os.path.join(folder_path, selected_filename)

def check_sentiment_exists(dataframe):
    if 'sentiment' in dataframe.columns and 'created_date' in dataframe.columns:
        return True
    else:
        return False

def get_sorted_by_datetime(dataframe):
    df_sentiment = dataframe.sort_values(by=['created_date'])
    df_sentiment = df_sentiment.drop(columns=['tweet_id','tweet'])
    return df_sentiment

def groupby_sentiment_seconds(dataframe):
    df_sentiment_seconds = dataframe.groupby(['created_date']).mean()
    return df_sentiment_seconds

def groupby_sentiment_hour(dataframe):
    df_sentiment = dataframe
    df_day = df_sentiment['created_date'].str.split(' ', n = 1, expand = True)
    df_hour = df_day[1].str.split(':', n = 1, expand = True)
    df_sentiment['created_datehour'] = df_day[0] +'-'+ df_hour[0]
    df_sentiment_datehour = df_sentiment.groupby(['created_datehour']).mean()
    return df_sentiment_datehour

def groupby_sentiment_day(dataframe):
    df_sentiment = dataframe
    df_day = df_sentiment['created_date'].str.split(' ', n = 1, expand = True)
    df_sentiment['created_dateday'] = df_day[0]
    df_sentiment_dateday = df_sentiment.groupby(['created_dateday']).mean()
    return df_sentiment_dateday

def get_full_filtered_csv(dataframe):
    df_sentiment = get_sorted_by_datetime(dataframe)
    df_sentiment_seconds = groupby_sentiment_seconds(df_sentiment)
    df_sentiment_datehour = groupby_sentiment_hour(df_sentiment)
    df_sentiment_dateday = groupby_sentiment_day(df_sentiment)
    return df_sentiment_seconds,df_sentiment_datehour,df_sentiment_dateday

def reload_sentiment_analysis_ui(df_selected, column):
    df_reset = df_selected.reset_index()
    index = df_reset[column].to_numpy()
    start_selection = st.selectbox('Start datetime', index)
    end_selection = st.selectbox('End datetime', index, index=len(index)-1)

    start_index = df_reset.index[df_reset[column] == start_selection][0]
    end_index = df_reset.index[df_reset[column] == end_selection][0]
    st.subheader("Average Sentiment")
    st.write("Y axis - Sentiment: -1 negative, 0 neutral , 1 positive")
    st.write("X axis - Datetime")
    st.line_chart(df_selected[start_index:end_index])
    st.subheader("Cumulative Sentiment")
    st.write("Y axis - Cumulative Sentiment")
    st.write("X axis - Datetime")
    st.line_chart(df_selected[start_index:end_index].cumsum())

def get_model_path(model_name):
    if model_name == 'TFidf':
        return './models/tfidf_coronatweetmodel.h5','./models/tfidf_tokenizer.pickle'
    if model_name == 'GRU':
        return './models/GRU_coronatweetmodel.h5','./models/GRU_tokenizer.pickle'
    if model_name == 'LSTM':
        return './models/LSTM_coronatweetmodel.h5','./models/LSTM_tokenizer.pickle'
    return 'model not found'