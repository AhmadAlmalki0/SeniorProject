# loading all the libraries necessary

import streamlit as st
import pandas as pd
import numpy as np
import tweepy
import joblib
from bs4 import BeautifulSoup
import re

def classification (text):
    pipeline = joblib.load('../model/pipe.joblib')
    prediction = pipeline.predict([text])
    
    return prediction

#pages will be included to functions
#def setting():
 #   st.write("Settings can be added as an extension for this project like tweaking the model and parameters")
  #  print('setting')
    
def extract():
    def scrape(words, date_since, numtweet):
 
        # Creating DataFrame using pandas
        db = pd.DataFrame(columns=['username',
                                   'description',
                                   'location',
                                   'following',
                                   'followers',
                                   'totaltweets',
                                   'retweetcount',
                                   'text',
                                   'hashtags'])
 
        # We are using .Cursor() to search
        # through twitter for the required tweets.
        # The number of tweets can be
        # restricted using .items(number of tweets)
        tweets = tweepy.Cursor(api.search_tweets,
                               words, lang="ar",
                               since_id=date_since,
                               tweet_mode='extended').items(numtweet)
 
 
        # .Cursor() returns an iterable object. Each item in
        # the iterator has various attributes
        # that you can access to
        # get information about each tweet
        list_tweets = [tweet for tweet in tweets]
 
        # Counter to maintain Tweet Count
        i = 1
 
        # we will iterate over each tweet in the
        # list for extracting information about each tweet
        for tweet in list_tweets:
                username = tweet.user.screen_name
                description = tweet.user.description
                location = tweet.user.location
                following = tweet.user.friends_count
                followers = tweet.user.followers_count
                totaltweets = tweet.user.statuses_count
                retweetcount = tweet.retweet_count
                hashtags = tweet.entities['hashtags']
 
                # Retweets can be distinguished by
                # a retweeted_status attribute,
                # in case it is an invalid reference,
                # except block will be executed
                try:
                        text = tweet.retweeted_status.full_text
                except AttributeError:
                        text = tweet.full_text
                hashtext = list()
                for j in range(0, len(hashtags)):
                        hashtext.append(hashtags[j]['text'])
 
                # Here we are appending all the
                # extracted information in the DataFrame
                ith_tweet = [username, description,
                             location, following,
                             followers, totaltweets,
                             retweetcount, text, hashtext]
                db.loc[len(db)] = ith_tweet
 
                # Function call to print tweet data on screen
                
                i = i+1
        filename = 'scraped_tweets.csv'
 
        # we will save our database as a CSV file.
        db.to_csv(filename)
        
        return(db)
        
    consumer_key = st.secrets["consumer_key"]  
    consumer_secret = st.secrets["consumer_secret"]    
    access_key = st.secrets["access_key"]      
    access_secret = st.secrets["access_secret"]     


    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)

    hashtag = st.text_input('Enter Twitter Hashtag to search for')
    date = st.text_input('Enter Date since The Tweets are required in yyyy-mm--dd')
    number_of_tweet = st.number_input('Enter the number of tweets to classify', value=100)

    #button to extract tweets:
    if st.button('Extract'):
        df = scrape(hashtag, date, number_of_tweet)
        st.text('Extraction was successful')
        
        
        new_df = df[['username','text','retweetcount']]
        new_df["prediction"] = new_df["text"].apply(lambda x: classification(x))
        st.dataframe(new_df)
        print(df.columns)
        total = len(new_df['prediction'])
        ctrue = len(new_df[new_df["prediction"]==1])
        cfalse = len(new_df[new_df["prediction"]==0])
        st.text('Total tweets count')
        st.text(total)
        st.text('Normal tweets count')
        st.text(ctrue)
        st.text('Ads tweets count')
        st.text(cfalse)
    else:
        pass
    
def predict():
    tweet_text = st.text_area(label = 'Enter your tweet to be classified', height=3)
    
    if tweet_text:
        #loading the model
        pipeline = joblib.load('../model/pipe.joblib')

        #predict the text

        prediction = pipeline.predict([tweet_text])

        if prediction[0] == "True":
            prediction_text = '<p style="font-family:Sans-Serif; font-weight: bold; color:Red; font-size: 40px;">Advertisment</p>'
            
        else:
            prediction_text = '<p style="font-family:Sans-Serif; font-weight: bold; color:Green; font-size: 40px;">Not Advertisment</p>'
        
        
       
        
        st.markdown(prediction_text, unsafe_allow_html=True)
        
def enter_url():
    url = st.text_input("Enter the url of the tweet(Private tweet won't work)")
    
    consumer_key = st.secrets["consumer_key"]  
    consumer_secret = st.secrets["consumer_secret"]   
    access_key = st.secrets["access_key"]     
    access_secret = st.secrets["access_secret"]    


    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)
    
    if url:
        text = api.get_oembed(url = url)
        soup = BeautifulSoup(text['html'],'html.parser')
        
        string = str(soup.find_all('p')[0])
        
        patter_start = "ar"
        match_start = re.search(patter_start, string)
        
        pattern_end = "</p>"
        match_end = re.search(pattern_end, string)
        
        start_index = match_start.end() + 2
        end_index = match_end.start()
        
        final_text = string[start_index:end_index]
        
        # st.text(final_text)
        
        #predicting the url
        if final_text:
            #loading the model
            pipeline = joblib.load('../model/pipe.joblib')

            #predict the text

            prediction = pipeline.predict([final_text])

            if prediction[0] == "True":
                prediction_text = '<p style="font-family:Sans-Serif; font-weight: bold; color:Red; font-size: 40px;">Advertisment</p>'
                
            else:
                prediction_text = '<p style="font-family:Sans-Serif; font-weight: bold; color:Green; font-size: 40px;">Not Advertisment</p>'
            
            display_tweet = f'<p style="font-family:Sans-Serif; font-weight: semi-bold; color:White; font-size: 20px;">{final_text}</p>'
        
            st.markdown(display_tweet, unsafe_allow_html=True)
            st.markdown(prediction_text, unsafe_allow_html=True)
    
