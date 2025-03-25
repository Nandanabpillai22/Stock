import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import datetime
import os
import json
from typing import Dict, List, Optional, Tuple, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_stock_data(ticker: str, 
                    period: str = "2y", 
                    interval: str = "1d") -> pd.DataFrame:
    """
    Fetch historical stock data from Yahoo Finance.
    
    Args:
        ticker (str): Stock ticker symbol
        period (str): Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        interval (str): Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        
    Returns:
        pd.DataFrame: DataFrame with historical stock data
    """
    try:
        logger.info(f"Fetching data for {ticker} with period={period} and interval={interval}")
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        
        if df.empty:
            logger.warning(f"No data found for {ticker}. Check if the ticker symbol is correct.")
            return pd.DataFrame()
            
        # Reset index to have Date as a column
        df = df.reset_index()
        df.rename(columns={"Date": "date", 
                           "Open": "open", 
                           "High": "high", 
                           "Low": "low", 
                           "Close": "close", 
                           "Volume": "volume"}, 
                 inplace=True)
        
        # Handle time zone issues
        if isinstance(df['date'].iloc[0], pd.Timestamp):
            df['date'] = df['date'].dt.tz_localize(None)
        
        # Log the data shape and date range
        if not df.empty:
            logger.info(f"Retrieved {len(df)} rows of data for {ticker} from {df['date'].min()} to {df['date'].max()}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame()

def get_stock_info(ticker: str) -> Dict:
    """
    Get company information for a given stock ticker.
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        Dict: Dictionary with company information
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Return the full info dictionary instead of just a subset
        return info
    
    except Exception as e:
        logger.error(f"Error fetching info for {ticker}: {str(e)}")
        return {}

def fetch_multiple_stocks(tickers: List[str], 
                         period: str = "2y", 
                         interval: str = "1d") -> Dict[str, pd.DataFrame]:
    """
    Fetch historical data for multiple stocks.
    
    Args:
        tickers (List[str]): List of stock ticker symbols
        period (str): Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        interval (str): Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary of DataFrames with historical data for each ticker
    """
    result = {}
    for ticker in tickers:
        df = fetch_stock_data(ticker, period, interval)
        if not df.empty:
            result[ticker] = df
    
    return result

def save_stock_data(data: pd.DataFrame, ticker: str, data_dir: str = "../data") -> str:
    """
    Save stock data to CSV file.
    
    Args:
        data (pd.DataFrame): Stock data DataFrame
        ticker (str): Stock ticker symbol
        data_dir (str): Directory to save data
        
    Returns:
        str: Path to saved file
    """
    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate filename
    now = datetime.datetime.now().strftime("%Y%m%d")
    filename = f"{ticker}_{now}.csv"
    filepath = os.path.join(data_dir, filename)
    
    # Save to CSV
    data.to_csv(filepath, index=False)
    logger.info(f"Data saved to {filepath}")
    
    return filepath

def fetch_news_sentiment(ticker: str, days: int = 7) -> List[Dict]:
    """
    Fetch recent news articles for a stock ticker and analyze their sentiment.
    
    Args:
        ticker (str): Stock ticker symbol
        days (int): Number of recent days of news to consider
        
    Returns:
        List[Dict]: List of news items with sentiment scores
    """
    try:
        logger.info(f"Fetching news articles for {ticker}")
        
        # Get news from Yahoo Finance
        stock = yf.Ticker(ticker)
        news_items = stock.news
        
        # Debug log
        logger.info(f"Retrieved {len(news_items) if news_items else 0} news items for {ticker}")
        
        # If no news found, return placeholder message
        if not news_items:
            logger.warning(f"No news found for {ticker}")
            return [{
                "date": datetime.datetime.now().strftime("%Y-%m-%d"),
                "headline": f"No recent news found for {ticker}",
                "source": "Info",
                "sentiment_score": 0.5,  # neutral score
                "url": f"https://finance.yahoo.com/quote/{ticker}"
            }]
        
        # Process news items
        result = []
        for i, item in enumerate(news_items[:min(len(news_items), days * 3)]):  # Limit to ~3 articles per day
            try:
                # Extract content which contains the actual news data
                content = item.get('content', {})
                if not isinstance(content, dict):
                    logger.warning(f"News item {i} content is not a dictionary: {type(content)}")
                    continue
                
                # Extract news ID for logging and debugging
                news_id = item.get('id', f"news-{i}")
                logger.info(f"Processing news item {i+1} with ID: {news_id}")
                
                # Extract title/headline from content
                headline = content.get('title', f"News about {ticker}")
                
                # Extract publication date from content
                pub_date = "Unknown date"
                if 'pubDate' in content and content['pubDate']:
                    # Parse ISO format date
                    try:
                        pub_date = datetime.datetime.strptime(content['pubDate'], "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d")
                    except:
                        # Try alternate format if the first one fails
                        try:
                            pub_date = datetime.datetime.fromisoformat(content['pubDate'].replace('Z', '+00:00')).strftime("%Y-%m-%d")
                        except:
                            logger.warning(f"Could not parse date format: {content['pubDate']}")
                
                # Extract source/provider from content
                source = "Yahoo Finance"
                if 'provider' in content and isinstance(content['provider'], dict) and 'displayName' in content['provider']:
                    source = content['provider']['displayName']
                
                # Extract URL - ensure it's a unique link
                url = None
                # First try canonicalUrl
                if 'canonicalUrl' in content and isinstance(content['canonicalUrl'], dict) and 'url' in content['canonicalUrl']:
                    url = content['canonicalUrl']['url']
                # Then try clickThroughUrl
                elif 'clickThroughUrl' in content and isinstance(content['clickThroughUrl'], dict) and 'url' in content['clickThroughUrl']:
                    url = content['clickThroughUrl']['url']
                # Fallback URL based on the ticker symbol
                else:
                    url = f"https://finance.yahoo.com/quote/{ticker}/news"
                
                logger.info(f"Article URL for '{headline}': {url}")
                
                # Get description and summary for more content to analyze
                description = ""
                if 'summary' in content and content['summary']:
                    description = content['summary']
                elif 'description' in content and content['description']:
                    description = content['description']
                    # Remove HTML tags if present
                    if '<' in description and '>' in description:
                        import re
                        description = re.sub(r'<[^>]+>', ' ', description)
                
                # Combine headline and description for better sentiment analysis
                full_text = (headline + " " + description).lower()
                
                # Simple sentiment analysis based on keywords
                # In a production app, you would use a proper NLP sentiment analyzer
                sentiment_score = 0.5  # neutral by default
                
                # Very basic sentiment analysis with more keywords for better accuracy
                positive_words = ['surge', 'jump', 'rise', 'gain', 'positive', 'growth', 'profit', 
                                 'up', 'higher', 'beat', 'strong', 'success', 'rally', 'bullish', 'opportunity',
                                 'outperform', 'upgrade', 'increase', 'improved', 'exceed', 'better', 'boost',
                                 'promising', 'advantage', 'optimistic', 'favorable', 'win', 'breakthrough']
                negative_words = ['plunge', 'fall', 'drop', 'decline', 'negative', 'loss', 'down', 
                                 'lower', 'miss', 'weak', 'fail', 'sell', 'bearish', 'risk', 'concern',
                                 'downgrade', 'underperform', 'decrease', 'disappointing', 'below', 'worse',
                                 'threat', 'struggle', 'challenging', 'difficult', 'pessimistic', 'warning']
                
                # Check for positive words
                pos_count = sum(1 for word in positive_words if word in full_text)
                # Check for negative words
                neg_count = sum(1 for word in negative_words if word in full_text)
                
                logger.info(f"Sentiment analysis for '{headline}': pos={pos_count}, neg={neg_count}")
                
                # Calculate sentiment score (0 to 1 scale)
                if pos_count > 0 or neg_count > 0:
                    sentiment_score = (0.5 + (pos_count - neg_count) * 0.1)
                    # Clamp to 0.1-0.9 range
                    sentiment_score = max(0.1, min(0.9, sentiment_score))
                
                # Add to result list with unique identifier
                result.append({
                    "date": pub_date,
                    "headline": headline,
                    "source": source,
                    "sentiment_score": sentiment_score,
                    "url": url,
                    "id": news_id  # Add a unique ID from the news item
                })
            except Exception as e:
                logger.error(f"Error processing news item {i}: {str(e)}")
                # Continue processing other items
        
        logger.info(f"Processed {len(result)} news articles for {ticker}")
        return result
        
    except Exception as e:
        logger.error(f"Error fetching news sentiment for {ticker}: {str(e)}")
        # Return a single fallback item in case of error
        return [{
            "date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "headline": f"Error fetching news for {ticker}",
            "source": "Error",
            "sentiment_score": 0.5,
            "url": f"https://finance.yahoo.com/quote/{ticker}"
        }]

def get_market_indices() -> pd.DataFrame:
    """
    Fetch current values of major market indices.
    
    Returns:
        pd.DataFrame: DataFrame with index values
    """
    # List of major indices
    indices = ['^GSPC', '^DJI', '^IXIC', '^RUT', '^VIX']
    names = ['S&P 500', 'Dow Jones', 'NASDAQ', 'Russell 2000', 'VIX']
    
    data = []
    for idx, name in zip(indices, names):
        try:
            index_data = yf.Ticker(idx).history(period='1d')
            if not index_data.empty:
                last_close = index_data['Close'].iloc[-1]
                last_change = index_data['Close'].iloc[-1] - index_data['Close'].iloc[-2] if len(index_data) > 1 else 0
                last_change_pct = (last_change / index_data['Close'].iloc[-2]) * 100 if len(index_data) > 1 else 0
                
                data.append({
                    'Index': name,
                    'Symbol': idx,
                    'Value': last_close,
                    'Change': last_change,
                    'Change%': last_change_pct
                })
        except Exception as e:
            logger.error(f"Error fetching data for index {idx}: {str(e)}")
    
    return pd.DataFrame(data) 

def analyze_article_sentiment(article_url: str) -> Dict:
    """
    Analyze the sentiment of a news article from a provided URL.
    
    Args:
        article_url (str): URL of the news article to analyze
        
    Returns:
        Dict: Analysis results including sentiment score and classification
    """
    try:
        logger.info(f"Analyzing sentiment for article: {article_url}")
        
        # Initialize sentiment analysis result
        result = {
            "url": article_url,
            "headline": "Could not parse article",
            "sentiment_score": 0.5,  # neutral by default
            "sentiment_label": "Neutral",
            "date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "source": "Unknown source",
            "success": False,
            "error": None
        }
        
        # Import necessary libraries
        try:
            import requests
            from bs4 import BeautifulSoup
            from urllib.parse import urlparse
        except ImportError as e:
            logger.error(f"Required library not installed: {str(e)}")
            result["error"] = f"Required library not installed: {str(e)}"
            return result
        
        # Get the article content
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(article_url, headers=headers, timeout=10)
            response.raise_for_status()  # Raise exception for 4XX/5XX status codes
        except Exception as e:
            logger.error(f"Error fetching article URL: {str(e)}")
            result["error"] = f"Error fetching article: {str(e)}"
            return result
        
        # Parse the content
        try:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Get article title - usually in the title tag or h1
            title = soup.title.string if soup.title else None
            if not title:
                title_tag = soup.find('h1')
                title = title_tag.text.strip() if title_tag else "Unknown title"
            
            # Clean up the title
            title = title.replace(" | Yahoo Finance", "").replace(" - Yahoo Finance", "")
            result["headline"] = title
            
            # Extract article text from common article element tags
            article_tags = soup.find_all(['p', 'article', 'div.article-body', 'div.content'])
            article_text = " ".join([tag.text for tag in article_tags])
            
            # Get source domain from URL
            parsed_url = urlparse(article_url)
            source = parsed_url.netloc.replace("www.", "")
            result["source"] = source
            
            # Combine title and article text for sentiment analysis
            full_text = (title + " " + article_text).lower()
            
            # Use the same sentiment analysis logic as in fetch_news_sentiment
            positive_words = ['surge', 'jump', 'rise', 'gain', 'positive', 'growth', 'profit', 
                             'up', 'higher', 'beat', 'strong', 'success', 'rally', 'bullish', 'opportunity',
                             'outperform', 'upgrade', 'increase', 'improved', 'exceed', 'better', 'boost',
                             'promising', 'advantage', 'optimistic', 'favorable', 'win', 'breakthrough']
            negative_words = ['plunge', 'fall', 'drop', 'decline', 'negative', 'loss', 'down', 
                             'lower', 'miss', 'weak', 'fail', 'sell', 'bearish', 'risk', 'concern',
                             'downgrade', 'underperform', 'decrease', 'disappointing', 'below', 'worse',
                             'threat', 'struggle', 'challenging', 'difficult', 'pessimistic', 'warning']
            
            # Check for positive and negative words
            pos_count = sum(1 for word in positive_words if word in full_text)
            neg_count = sum(1 for word in negative_words if word in full_text)
            
            logger.info(f"Sentiment analysis for article: pos={pos_count}, neg={neg_count}")
            
            # Calculate sentiment score (0 to 1 scale)
            if pos_count > 0 or neg_count > 0:
                sentiment_score = (0.5 + (pos_count - neg_count) * 0.1)
                # Clamp to 0.1-0.9 range
                sentiment_score = max(0.1, min(0.9, sentiment_score))
            else:
                sentiment_score = 0.5  # neutral
            
            result["sentiment_score"] = sentiment_score
            
            # Set sentiment label
            if sentiment_score > 0.6:
                result["sentiment_label"] = "Bullish"
            elif sentiment_score < 0.4:
                result["sentiment_label"] = "Bearish"
            else:
                result["sentiment_label"] = "Neutral"
            
            # Set success flag
            result["success"] = True
            result["error"] = None
            
            logger.info(f"Successfully analyzed article: {title} with sentiment score: {sentiment_score}")
            return result
            
        except Exception as e:
            logger.error(f"Error parsing article content: {str(e)}")
            result["error"] = f"Error parsing article content: {str(e)}"
            return result
    
    except Exception as e:
        logger.error(f"Unexpected error analyzing article sentiment: {str(e)}")
        return {
            "url": article_url,
            "headline": "Error analyzing article",
            "sentiment_score": 0.5,  # neutral by default
            "sentiment_label": "Neutral",
            "date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "source": "Error",
            "success": False,
            "error": str(e)
        } 