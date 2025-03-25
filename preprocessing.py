import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Dict, Union
import logging
import ta  # Technical Analysis library

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the dataframe.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        
    Returns:
        pd.DataFrame: DataFrame with technical indicators
    """
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Ensure the column names are correct
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_columns):
        # Try to find alternate capitalization
        column_map = {}
        for req_col in required_columns:
            for col in df.columns:
                if col.lower() == req_col:
                    column_map[col] = req_col
        
        if column_map:
            df = df.rename(columns=column_map)
        else:
            logger.error("Required columns not found in dataframe")
            return df

    try:
        # Add Moving Averages
        df['ma7'] = df['close'].rolling(window=7).mean()
        df['ma21'] = df['close'].rolling(window=21).mean()
        df['ma50'] = df['close'].rolling(window=50).mean()
        
        # Add RSI (Relative Strength Index)
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        
        # Add MACD (Moving Average Convergence Divergence)
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        # Add Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['bollinger_high'] = bollinger.bollinger_hband()
        df['bollinger_low'] = bollinger.bollinger_lband()
        df['bollinger_pct'] = bollinger.bollinger_pband()
        
        # Add ATR (Average True Range) - volatility indicator
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        
        # Add OBV (On-Balance Volume) - volume indicator
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        
        # Add price changes
        df['price_change'] = df['close'].pct_change()
        df['price_change_1d'] = df['close'].pct_change(periods=1)
        df['price_change_5d'] = df['close'].pct_change(periods=5)
        df['price_change_21d'] = df['close'].pct_change(periods=21)
        
        # Calculate volatility over different windows
        df['volatility_7d'] = df['price_change'].rolling(window=7).std()
        df['volatility_21d'] = df['price_change'].rolling(window=21).std()
        
        # Add day of week as a cyclical feature
        if 'date' in df.columns:
            df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
            df['day_sin'] = np.sin(df['day_of_week'] * (2 * np.pi / 7))
            df['day_cos'] = np.cos(df['day_of_week'] * (2 * np.pi / 7))
            
            # Add month as a cyclical feature
            df['month'] = pd.to_datetime(df['date']).dt.month
            df['month_sin'] = np.sin(df['month'] * (2 * np.pi / 12))
            df['month_cos'] = np.cos(df['month'] * (2 * np.pi / 12))
        
        # Drop NaN values created by the indicators
        return df
    
    except Exception as e:
        logger.error(f"Error adding technical indicators: {str(e)}")
        return df

def prepare_stock_data(df: pd.DataFrame, 
                     seq_length: int = 60,
                     target_column: str = 'close',
                     feature_columns: List[str] = None,
                     forecast_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, MinMaxScaler]:
    """
    Prepare stock data for LSTM model by creating sequences and scaling.
    
    Args:
        df (pd.DataFrame): DataFrame with stock data and indicators
        seq_length (int): Length of input sequences (lookback period)
        target_column (str): Column name to predict
        feature_columns (List[str]): List of feature columns to use
        forecast_horizon (int): Number of days to forecast ahead
        
    Returns:
        Tuple: (X, y, forecast_df, scaler)
            X: Scaled input sequences
            y: Scaled target values
            forecast_df: DataFrame with dates for forecasting
            scaler: Fitted MinMaxScaler for inverse transformations
    """
    # Add technical indicators
    df = add_technical_indicators(df)
    
    # Default feature columns if not specified
    if feature_columns is None:
        feature_columns = ['close', 'volume', 'ma7', 'ma21', 'rsi', 'macd', 'bollinger_pct', 'volatility_7d']
    
    # Prepare the feature columns - remove NaN values
    df_features = df[feature_columns].copy()
    df_features = df_features.dropna()
    
    # If date is present in the DataFrame, save it for later reference
    if 'date' in df.columns:
        df_dates = df['date'].iloc[len(df) - len(df_features):]
        df_dates = df_dates.reset_index(drop=True)
    else:
        df_dates = pd.Series(range(len(df_features)))
    
    # Scale the features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_features)
    
    # Create X and y for time series forecasting
    X = []
    y = []
    
    for i in range(seq_length, len(scaled_data) - forecast_horizon + 1):
        X.append(scaled_data[i - seq_length:i])
        # Get the target column index
        target_idx = feature_columns.index(target_column)
        y.append(scaled_data[i + forecast_horizon - 1, target_idx])
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Create a DataFrame for forecasting (dates and true values)
    forecast_df = pd.DataFrame({
        'date': df_dates.iloc[seq_length:len(df_dates) - forecast_horizon + 1].values,
        'true_value': df_features[target_column].iloc[seq_length + forecast_horizon - 1:].values
    })
    
    return X, y, forecast_df, scaler

def prepare_multi_output_model(df: pd.DataFrame,
                              seq_length: int = 60,
                              target_column: str = 'close',
                              feature_columns: List[str] = None,
                              forecast_horizons: List[int] = [1, 5, 21]) -> Dict:
    """
    Prepare stock data for multi-output LSTM model (different forecast horizons).
    
    Args:
        df (pd.DataFrame): DataFrame with stock data and indicators
        seq_length (int): Length of input sequences (lookback period)
        target_column (str): Column name to predict
        feature_columns (List[str]): List of feature columns to use
        forecast_horizons (List[int]): List of forecast horizons (days ahead to predict)
        
    Returns:
        Dict: Dictionary with data for different forecast horizons
    """
    # Add technical indicators
    df = add_technical_indicators(df)
    
    # Default feature columns if not specified
    if feature_columns is None:
        feature_columns = ['close', 'volume', 'ma7', 'ma21', 'rsi', 'macd', 'bollinger_pct', 'volatility_7d']
    
    # Prepare the feature columns - remove NaN values
    df_features = df[feature_columns].copy()
    df_features = df_features.dropna()
    
    # Scale the features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_features)
    
    # If date is present in the DataFrame, save it for later reference
    if 'date' in df.columns:
        df_dates = df['date'].iloc[len(df) - len(df_features):]
        df_dates = df_dates.reset_index(drop=True)
    else:
        df_dates = pd.Series(range(len(df_features)))
    
    # Get the target column index
    target_idx = feature_columns.index(target_column)
    
    result = {
        'scaler': scaler,
        'feature_columns': feature_columns,
        'target_column': target_column,
        'target_idx': target_idx,
        'horizons': {}
    }
    
    # For each forecast horizon, create X and y
    max_horizon = max(forecast_horizons)
    
    for horizon in forecast_horizons:
        X = []
        y = []
        
        for i in range(seq_length, len(scaled_data) - max_horizon + 1):
            X.append(scaled_data[i - seq_length:i])
            y.append(scaled_data[i + horizon - 1, target_idx])
        
        X_horizon = np.array(X)
        y_horizon = np.array(y)
        
        # Create a DataFrame for forecasting (dates and true values)
        forecast_df = pd.DataFrame({
            'date': df_dates.iloc[seq_length:len(df_dates) - max_horizon + 1].values,
            f'true_h{horizon}': df_features[target_column].iloc[seq_length + horizon - 1:len(df_features) - max_horizon + horizon].values
        })
        
        result['horizons'][horizon] = {
            'X': X_horizon,
            'y': y_horizon,
            'forecast_df': forecast_df
        }
    
    return result

def split_train_test(X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the data into training and testing sets.
    
    Args:
        X (np.ndarray): Input sequences
        y (np.ndarray): Target values
        test_size (float): Proportion of data to use for testing (0 to 1)
        
    Returns:
        Tuple: (X_train, X_test, y_train, y_test)
    """
    # Calculate the split point
    split_idx = int(len(X) * (1 - test_size))
    
    # Split the data
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test

def inverse_transform_predictions(predictions: np.ndarray, 
                               scaler: MinMaxScaler, 
                               feature_columns: List[str],
                               target_column: str) -> np.ndarray:
    """
    Inverse transform scaled predictions back to original scale.
    
    Args:
        predictions (np.ndarray): Scaled predictions
        scaler (MinMaxScaler): Fitted scaler used for scaling
        feature_columns (List[str]): List of feature column names
        target_column (str): Name of the target column
        
    Returns:
        np.ndarray: Predictions in original scale
    """
    # Get the target column index
    target_idx = feature_columns.index(target_column)
    
    # Create a placeholder array
    dummy = np.zeros((len(predictions), len(feature_columns)))
    # Put the predictions in the target column index
    dummy[:, target_idx] = predictions
    
    # Inverse transform
    dummy_inverse = scaler.inverse_transform(dummy)
    
    # Return the target column
    return dummy_inverse[:, target_idx] 