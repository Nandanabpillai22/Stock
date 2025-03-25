import pandas as pd
import numpy as np
import requests
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from typing import List, Dict, Union, Optional
import logging
import datetime
import time
import os
import json
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

class SentimentAnalyzer:
    def __init__(self):
        """
        Sentiment analyzer for financial news and social media data.
        """
        self.sia = SentimentIntensityAnalyzer()
        
        # Financial terms sentiment lexicon - add domain-specific terms
        self.financial_lexicon = {
            # Positive financial terms
            'bullish': 0.7,
            'outperform': 0.6,
            'upgrade': 0.5,
            'beat': 0.5,
            'exceed': 0.5,
            'growth': 0.4,
            'profit': 0.4,
            'gain': 0.4,
            'positive': 0.4,
            'rise': 0.3,
            'up': 0.3,
            'higher': 0.3,
            'rally': 0.5,
            'strong': 0.4,
            'strength': 0.4,
            'opportunity': 0.4,
            'recovery': 0.4,
            'breakthrough': 0.5,
            'innovation': 0.4,
            'dividend': 0.3,
            
            # Negative financial terms
            'bearish': -0.7,
            'underperform': -0.6,
            'downgrade': -0.5,
            'miss': -0.5,
            'below': -0.3,
            'decline': -0.4,
            'loss': -0.5,
            'negative': -0.4,
            'fall': -0.3,
            'down': -0.3,
            'lower': -0.3,
            'selloff': -0.5,
            'weak': -0.4,
            'weakness': -0.4,
            'risk': -0.3,
            'debt': -0.3,
            'bankruptcy': -0.8,
            'recession': -0.6,
            'litigation': -0.5,
            'investigation': -0.4,
            'volatility': -0.3
        }
        
        # Add financial terms to the VADER lexicon
        for term, score in self.financial_lexicon.items():
            self.sia.lexicon[term] = score
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze the sentiment of a text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, float]: Dictionary with sentiment scores
        """
        sentiment = self.sia.polarity_scores(text)
        return sentiment
    
    def analyze_news_headlines(self, headlines: List[str]) -> pd.DataFrame:
        """
        Analyze sentiment for a list of news headlines.
        
        Args:
            headlines (List[str]): List of news headlines
            
        Returns:
            pd.DataFrame: DataFrame with sentiment scores for each headline
        """
        results = []
        
        for headline in headlines:
            sentiment = self.analyze_text(headline)
            
            results.append({
                'headline': headline,
                'compound': sentiment['compound'],
                'positive': sentiment['pos'],
                'neutral': sentiment['neu'],
                'negative': sentiment['neg']
            })
        
        return pd.DataFrame(results)
    
    def analyze_news_df(self, news_df: pd.DataFrame, headline_col: str = 'headline') -> pd.DataFrame:
        """
        Analyze sentiment for news in a DataFrame.
        
        Args:
            news_df (pd.DataFrame): DataFrame with news
            headline_col (str): Column name for headlines
            
        Returns:
            pd.DataFrame: DataFrame with added sentiment scores
        """
        # Create a copy of the input DataFrame
        result_df = news_df.copy()
        
        # Add sentiment columns
        result_df['sentiment_compound'] = np.nan
        result_df['sentiment_positive'] = np.nan
        result_df['sentiment_neutral'] = np.nan
        result_df['sentiment_negative'] = np.nan
        
        # Analyze each headline
        for i, row in result_df.iterrows():
            headline = row[headline_col]
            
            if not isinstance(headline, str) or not headline:
                continue
                
            sentiment = self.analyze_text(headline)
            
            result_df.at[i, 'sentiment_compound'] = sentiment['compound']
            result_df.at[i, 'sentiment_positive'] = sentiment['pos']
            result_df.at[i, 'sentiment_neutral'] = sentiment['neu']
            result_df.at[i, 'sentiment_negative'] = sentiment['neg']
        
        return result_df

def get_daily_sentiment(news_df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """
    Aggregate news sentiment by day.
    
    Args:
        news_df (pd.DataFrame): DataFrame with news and sentiment scores
        date_col (str): Column name for dates
        
    Returns:
        pd.DataFrame: DataFrame with daily sentiment aggregation
    """
    # Ensure date column is datetime
    news_df[date_col] = pd.to_datetime(news_df[date_col])
    
    # Group by date and aggregate sentiment
    daily_sentiment = news_df.groupby(news_df[date_col].dt.date).agg({
        'sentiment_compound': ['mean', 'std', 'count'],
        'sentiment_positive': 'mean',
        'sentiment_negative': 'mean'
    })
    
    # Flatten multi-level columns
    daily_sentiment.columns = ['_'.join(col).strip() for col in daily_sentiment.columns.values]
    
    # Reset index
    daily_sentiment = daily_sentiment.reset_index()
    daily_sentiment.rename(columns={date_col: 'date'}, inplace=True)
    
    return daily_sentiment

def plot_sentiment_trend(daily_sentiment: pd.DataFrame, ticker: str = None):
    """
    Plot the trend of sentiment over time.
    
    Args:
        daily_sentiment (pd.DataFrame): DataFrame with daily sentiment
        ticker (str): Stock ticker for title
    """
    plt.figure(figsize=(12, 8))
    
    # Plot compound sentiment
    plt.subplot(2, 1, 1)
    plt.plot(daily_sentiment['date'], daily_sentiment['sentiment_compound_mean'], 'b-', label='Compound Score')
    
    # Add error bands if std is available
    if 'sentiment_compound_std' in daily_sentiment.columns:
        plt.fill_between(
            daily_sentiment['date'],
            daily_sentiment['sentiment_compound_mean'] - daily_sentiment['sentiment_compound_std'],
            daily_sentiment['sentiment_compound_mean'] + daily_sentiment['sentiment_compound_std'],
            color='b', alpha=0.2
        )
    
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.title(f'News Sentiment Trend{" for " + ticker if ticker else ""}')
    plt.ylabel('Compound Sentiment')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot positive and negative sentiment
    plt.subplot(2, 1, 2)
    plt.plot(daily_sentiment['date'], daily_sentiment['sentiment_positive_mean'], 'g-', label='Positive')
    plt.plot(daily_sentiment['date'], daily_sentiment['sentiment_negative_mean'], 'r-', label='Negative')
    
    plt.title('Positive vs Negative Sentiment')
    plt.ylabel('Sentiment Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def merge_sentiment_with_price(price_df: pd.DataFrame, 
                              sentiment_df: pd.DataFrame,
                              price_date_col: str = 'date',
                              sentiment_date_col: str = 'date') -> pd.DataFrame:
    """
    Merge stock price data with sentiment data.
    
    Args:
        price_df (pd.DataFrame): DataFrame with stock prices
        sentiment_df (pd.DataFrame): DataFrame with sentiment data
        price_date_col (str): Date column in price DataFrame
        sentiment_date_col (str): Date column in sentiment DataFrame
        
    Returns:
        pd.DataFrame: Merged DataFrame
    """
    # Ensure date columns are datetime
    price_df[price_date_col] = pd.to_datetime(price_df[price_date_col])
    sentiment_df[sentiment_date_col] = pd.to_datetime(sentiment_df[sentiment_date_col])
    
    # Merge dataframes on date
    merged_df = pd.merge(
        price_df, 
        sentiment_df, 
        left_on=price_date_col, 
        right_on=sentiment_date_col,
        how='left'
    )
    
    # Forward fill missing sentiment values
    sentiment_cols = [col for col in merged_df.columns if 'sentiment' in col]
    merged_df[sentiment_cols] = merged_df[sentiment_cols].fillna(method='ffill')
    
    # If still missing values at the beginning, backfill
    merged_df[sentiment_cols] = merged_df[sentiment_cols].fillna(method='bfill')
    
    # If still missing, fill with neutral sentiment
    merged_df['sentiment_compound_mean'] = merged_df['sentiment_compound_mean'].fillna(0)
    merged_df['sentiment_positive_mean'] = merged_df['sentiment_positive_mean'].fillna(0.5)
    merged_df['sentiment_negative_mean'] = merged_df['sentiment_negative_mean'].fillna(0.5)
    
    return merged_df

# Function to generate placeholder sentiment data for demo purposes
def generate_placeholder_sentiment(ticker: str, 
                                 start_date: str, 
                                 end_date: str, 
                                 news_per_day: int = 2) -> pd.DataFrame:
    """
    Generate placeholder sentiment data for demo purposes.
    
    Args:
        ticker (str): Stock ticker
        start_date (str): Start date (YYYY-MM-DD)
        end_date (str): End date (YYYY-MM-DD)
        news_per_day (int): Average number of news items per day
        
    Returns:
        pd.DataFrame: DataFrame with placeholder news and sentiment
    """
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate random news items
    news_items = []
    
    # Example headlines
    positive_headlines = [
        f"{ticker} reports strong quarterly results, exceeding expectations",
        f"Analysts upgrade {ticker} to 'Buy', citing growth potential",
        f"{ticker} announces new product line that could boost revenue",
        f"Market reacts positively to {ticker}'s strategic partnership",
        f"{ticker} shows strong performance in emerging markets"
    ]
    
    negative_headlines = [
        f"{ticker} misses quarterly earnings estimates",
        f"Analysts downgrade {ticker} citing competitive pressures",
        f"{ticker} faces regulatory challenges in key markets",
        f"Market concerns over {ticker}'s debt levels",
        f"{ticker} experiences supply chain disruptions"
    ]
    
    neutral_headlines = [
        f"{ticker} appoints new Chief Financial Officer",
        f"{ticker} to present at upcoming industry conference",
        f"{ticker} maintains market position despite industry shifts",
        f"{ticker} announces date for next earnings release",
        f"{ticker} completes scheduled annual maintenance"
    ]
    
    # Create news items
    for date in date_range:
        # Random number of news items for this day
        daily_news_count = np.random.poisson(news_per_day)
        
        for _ in range(daily_news_count):
            # Random sentiment type
            sentiment_type = np.random.choice(['positive', 'negative', 'neutral'], p=[0.4, 0.3, 0.3])
            
            if sentiment_type == 'positive':
                headline = np.random.choice(positive_headlines)
                sentiment = np.random.normal(0.6, 0.2)  # Positive sentiment
            elif sentiment_type == 'negative':
                headline = np.random.choice(negative_headlines)
                sentiment = np.random.normal(-0.5, 0.2)  # Negative sentiment
            else:
                headline = np.random.choice(neutral_headlines)
                sentiment = np.random.normal(0.1, 0.1)  # Neutral sentiment
            
            # Clip sentiment to valid range (-1 to 1)
            sentiment = max(min(sentiment, 1.0), -1.0)
            
            # Create news item
            news_items.append({
                'date': date,
                'headline': headline,
                'source': f"News Source {np.random.randint(1, 6)}",
                'sentiment_compound': sentiment,
                'sentiment_positive': max(0, sentiment),
                'sentiment_negative': max(0, -sentiment),
                'sentiment_neutral': 1 - max(0, sentiment) - max(0, -sentiment)
            })
    
    # Create DataFrame
    news_df = pd.DataFrame(news_items)
    
    return news_df 