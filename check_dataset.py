# check_dataset.py
import os
import json
import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer


dataset_path = "./stocknet-dataset"
price_path = os.path.join(dataset_path, "price/raw/")
tweet_path = os.path.join(dataset_path, "tweet/raw/")

print(f"Dataset path: {dataset_path}")
print(f"Price path: {price_path}")
print(f"Tweet path: {tweet_path}")

def process_and_aggregate_tweet_data(stock_symbol):
    stock_tweet_path = os.path.join(tweet_path, stock_symbol)
    if os.path.exists(stock_tweet_path) and os.path.isdir(stock_tweet_path):
        tweet_files = os.listdir(stock_tweet_path)
        sid = SentimentIntensityAnalyzer()
        daily_sentiments = {}

        for tweet_file in tweet_files:
            tweet_file_path = os.path.join(stock_tweet_path, tweet_file)
            if os.path.isfile(tweet_file_path):
                try:
                    with open(tweet_file_path, 'r') as f:
                        tweet_data = f.readlines()
                        daily_sentiment_sum = 0.0
                        tweet_count = 0

                        for tweet in tweet_data:
                            tweet_json = json.loads(tweet)
                            text = tweet_json.get('text', '')
                            if text.strip():
                                sentiment = sid.polarity_scores(text)
                                daily_sentiment_sum += sentiment['compound']
                                tweet_count += 1

                        if tweet_count > 0:
                            average_sentiment = daily_sentiment_sum / tweet_count
                        else:
                            average_sentiment = 0.0

                        date = tweet_file.split('.')[0]
                        daily_sentiments[date] = average_sentiment
                except Exception as e:
                    print(f"Could not read file {tweet_file_path}. Error: {e}")

        if daily_sentiments:
            sentiment_df = pd.DataFrame(list(daily_sentiments.items()), columns=['Date', 'Sentiment'])
            sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
            sentiment_df = sentiment_df.sort_values(by='Date')
            return sentiment_df
        else:
            return pd.DataFrame()
    else:
        return pd.DataFrame()

#Les inn og kombiner data for alle aksjer
combined_data = []

#Vi fokuserer på perioden 2014-2015
start_date = '2014-01-01'
end_date = '2015-12-31'

for stock_file in os.listdir(price_path):
    if stock_file.endswith('.csv'):
        stock_symbol = stock_file.split('.')[0]
        stock_price_path = os.path.join(price_path, stock_file)

        #Prisdata
        price_data = pd.read_csv(stock_price_path)
        price_data['Date'] = pd.to_datetime(price_data['Date'])

        #Filtrer data til ønsket periode
        price_data = price_data[(price_data['Date'] >= start_date) & (price_data['Date'] <= end_date)]
        if price_data.empty:
            continue

        #Les inn tweet-sentiment for denne aksjen
        sentiment_data = process_and_aggregate_tweet_data(stock_symbol)
        if not sentiment_data.empty:
            sentiment_data = sentiment_data[(sentiment_data['Date'] >= start_date) & (sentiment_data['Date'] <= end_date)]

        #Join pris og sentiment
        combined_df = pd.merge(price_data, sentiment_data, on='Date', how='left')
        combined_df['Sentiment'].fillna(0.0, inplace=True)
        combined_df['Symbol'] = stock_symbol
        combined_data.append(combined_df)

if not combined_data:
    print("Ingen data funnet for angitt periode.")
    exit(0)

combined_all_df = pd.concat(combined_data, ignore_index=True)
combined_all_df = combined_all_df.sort_values(by=['Symbol', 'Date'])

#Split tidsmessig
train_end = '2014-12-31'
val_end = '2015-06-30'
test_end = '2015-12-31'

train_data = combined_all_df[(combined_all_df['Date'] >= '2014-01-01') & (combined_all_df['Date'] <= train_end)]
val_data = combined_all_df[(combined_all_df['Date'] > train_end) & (combined_all_df['Date'] <= val_end)]
test_data = combined_all_df[(combined_all_df['Date'] > val_end) & (combined_all_df['Date'] <= test_end)]

#Skriv ut CSV
train_data.to_csv('train_data.csv', index=False)
val_data.to_csv('validation_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

print("Ferdig med å skrive ut train_data.csv, validation_data.csv og test_data.csv.")