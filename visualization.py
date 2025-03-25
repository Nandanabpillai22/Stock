import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.dates as mdates
from typing import List, Dict, Tuple, Union, Optional
import logging
from sklearn.preprocessing import MinMaxScaler
import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_stock_price(df: pd.DataFrame, 
                    ticker: str,
                    date_col: str = 'date',
                    price_col: str = 'close',
                    ma_cols: List[str] = None,
                    volume_col: str = 'volume',
                    figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot stock price with moving averages and volume.
    
    Args:
        df (pd.DataFrame): DataFrame with stock data
        ticker (str): Stock ticker
        date_col (str): Column name for dates
        price_col (str): Column name for price
        ma_cols (List[str]): List of moving average column names
        volume_col (str): Column name for volume
        figsize (Tuple[int, int]): Figure size
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # Plot price
    ax1.plot(df[date_col], df[price_col], label=price_col, color='blue')
    
    # Plot moving averages
    if ma_cols:
        colors = ['red', 'green', 'orange', 'purple']
        for i, ma in enumerate(ma_cols):
            if ma in df.columns:
                ax1.plot(df[date_col], df[ma], label=ma, color=colors[i % len(colors)])
    
    # Format x-axis
    if pd.api.types.is_datetime64_any_dtype(df[date_col]):
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
    
    # Add legend and labels
    ax1.set_title(f"{ticker} Stock Price")
    ax1.set_ylabel("Price")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot volume
    if volume_col in df.columns:
        ax2.bar(df[date_col], df[volume_col], color='gray', alpha=0.7)
        ax2.set_ylabel("Volume")
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig

def plot_interactive_candlestick(df: pd.DataFrame, 
                               ticker: str,
                               date_col: str = 'date',
                               open_col: str = 'open',
                               high_col: str = 'high',
                               low_col: str = 'low',
                               close_col: str = 'close',
                               volume_col: str = 'volume',
                               ma_cols: List[str] = None,
                               bollinger_cols: List[str] = None) -> go.Figure:
    """
    Create an interactive candlestick chart with Plotly.
    
    Args:
        df (pd.DataFrame): DataFrame with stock data
        ticker (str): Stock ticker
        date_col (str): Column name for dates
        open_col (str): Column name for open price
        high_col (str): Column name for high price
        low_col (str): Column name for low price
        close_col (str): Column name for close price
        volume_col (str): Column name for volume
        ma_cols (List[str]): List of moving average column names
        bollinger_cols (List[str]): List of Bollinger Band column names [upper, lower]
        
    Returns:
        go.Figure: Plotly figure
    """
    # Create subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.05, 
                       row_heights=[0.7, 0.3],
                       subplot_titles=(f"{ticker} Stock Price", "Volume"))
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df[date_col],
            open=df[open_col],
            high=df[high_col],
            low=df[low_col],
            close=df[close_col],
            name="Price"
        ),
        row=1, col=1
    )
    
    # Add moving averages
    if ma_cols:
        colors = ['rgba(255, 0, 0, 0.7)', 'rgba(0, 255, 0, 0.7)', 'rgba(255, 165, 0, 0.7)']
        for i, ma in enumerate(ma_cols):
            if ma in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df[date_col],
                        y=df[ma],
                        name=ma,
                        line=dict(color=colors[i % len(colors)], width=1.5)
                    ),
                    row=1, col=1
                )
    
    # Add Bollinger Bands
    if bollinger_cols and len(bollinger_cols) == 2:
        upper_col, lower_col = bollinger_cols
        if upper_col in df.columns and lower_col in df.columns:
            # Upper band
            fig.add_trace(
                go.Scatter(
                    x=df[date_col],
                    y=df[upper_col],
                    name="Upper Bollinger",
                    line=dict(color='rgba(173, 216, 230, 0.7)', width=1)
                ),
                row=1, col=1
            )
            
            # Lower band
            fig.add_trace(
                go.Scatter(
                    x=df[date_col],
                    y=df[lower_col],
                    name="Lower Bollinger",
                    line=dict(color='rgba(173, 216, 230, 0.7)', width=1),
                    fill='tonexty',
                    fillcolor='rgba(173, 216, 230, 0.1)'
                ),
                row=1, col=1
            )
    
    # Add volume
    if volume_col in df.columns:
        # Color volume bars based on price movement
        colors = ['green' if close > open else 'red' 
                 for close, open in zip(df[close_col], df[open_col])]
        
        fig.add_trace(
            go.Bar(
                x=df[date_col],
                y=df[volume_col],
                name="Volume",
                marker_color=colors,
                marker_line_width=0,
                opacity=0.7
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=800,
        title=f"{ticker} Stock Analysis",
        yaxis_title="Price",
        yaxis2_title="Volume",
        xaxis_title="Date",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white"
    )
    
    # Hide weekend gaps in x-axis
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"])  # hide weekends
        ]
    )
    
    return fig

def plot_prediction_vs_actual(date: List, 
                            actual: List, 
                            predictions: List, 
                            ticker: str,
                            confidence_intervals: Optional[Tuple[List, List]] = None) -> go.Figure:
    """
    Plot actual vs predicted values with Plotly.
    
    Args:
        date (List): List of dates
        actual (List): List of actual values
        predictions (List): List of predicted values
        ticker (str): Stock ticker
        confidence_intervals (Optional[Tuple[List, List]]): Upper and lower confidence intervals
        
    Returns:
        go.Figure: Plotly figure
    """
    fig = go.Figure()
    
    # Add actual values if provided
    if actual is not None:
        fig.add_trace(
            go.Scatter(
                x=date,
                y=actual,
                mode='lines',
                name='Actual',
                line=dict(color='blue', width=2)
            )
        )
    
    # Add predicted values
    fig.add_trace(
        go.Scatter(
            x=date,
            y=predictions,
            mode='lines',
            name='Predicted',
            line=dict(color='red', width=2)
        )
    )
    
    # Add confidence intervals if provided
    if confidence_intervals:
        upper, lower = confidence_intervals
        
        fig.add_trace(
            go.Scatter(
                x=date,
                y=upper,
                mode='lines',
                name='Upper CI',
                line=dict(width=0),
                showlegend=False
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=date,
                y=lower,
                mode='lines',
                name='Lower CI',
                line=dict(width=0),
                fillcolor='rgba(255, 0, 0, 0.1)',
                fill='tonexty',
                showlegend=False
            )
        )
    
    # Determine title based on whether we're showing actual values
    title_text = f"{ticker} - Actual vs Predicted Price" if actual is not None else f"{ticker} - Future Price Forecast"
    
    # Update layout
    fig.update_layout(
        title=title_text,
        xaxis_title="Date",
        yaxis_title="Price",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=600
    )
    
    # Hide weekend gaps in x-axis
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"])  # hide weekends
        ]
    )
    
    return fig

def plot_multi_horizon_predictions(forecasts: Dict[int, Dict], ticker: str) -> go.Figure:
    """
    Plot predictions for multiple time horizons.
    
    Args:
        forecasts (Dict[int, Dict]): Dictionary with forecasts for different horizons
        ticker (str): Stock ticker
        
    Returns:
        go.Figure: Plotly figure
    """
    fig = go.Figure()
    
    # Add actual values (from the shortest horizon forecast data)
    min_horizon = min(forecasts.keys())
    dates = forecasts[min_horizon]['dates']
    actual = forecasts[min_horizon]['actual']
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=actual,
            mode='lines',
            name='Actual',
            line=dict(color='blue', width=2)
        )
    )
    
    # Add predictions for each horizon
    colors = ['red', 'green', 'purple', 'orange', 'brown']
    
    for i, (horizon, data) in enumerate(forecasts.items()):
        # Add predicted values
        color = colors[i % len(colors)]
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=data['predictions'],
                mode='lines',
                name=f'{horizon}-Day Forecast',
                line=dict(color=color, width=2, dash='dot')
            )
        )
    
    # Update layout
    fig.update_layout(
        title=f"{ticker} - Multi-horizon Forecasts",
        xaxis_title="Date",
        yaxis_title="Price",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=600
    )
    
    # Hide weekend gaps in x-axis
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"])  # hide weekends
        ]
    )
    
    return fig

def plot_feature_importance(features: List[str], importance: List[float]) -> go.Figure:
    """
    Plot feature importance.
    
    Args:
        features (List[str]): List of feature names
        importance (List[float]): List of importance values
        
    Returns:
        go.Figure: Plotly figure
    """
    # Sort by importance (descending)
    sorted_idx = np.argsort(importance)[::-1]
    sorted_features = [features[i] for i in sorted_idx]
    sorted_importance = [importance[i] for i in sorted_idx]
    
    # Color gradient based on importance
    max_importance = max(sorted_importance)
    colors = [f'rgba(55, 83, 109, {max(0.4, min(1.0, imp/max_importance))})' 
              if max_importance > 0 else 'rgba(55, 83, 109, 0.7)' 
              for imp in sorted_importance]
    
    fig = go.Figure()
    
    # Create the importance bars
    fig.add_trace(
        go.Bar(
            y=sorted_features,
            x=sorted_importance,
            orientation='h',
            marker_color=colors,
            marker=dict(
                line=dict(width=1, color='rgba(0, 0, 0, 0.2)')
            ),
            text=[f"{imp:.3f}" for imp in sorted_importance],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
        )
    )
    
    # Update layout with better styling
    fig.update_layout(
        title={
            'text': 'Feature Importance for Stock Prediction',
            'font': {'size': 20, 'color': '#333333'},
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title={
            'text': 'Relative Importance',
            'font': {'size': 16, 'color': '#333333'}
        },
        yaxis_title={
            'text': 'Feature',
            'font': {'size': 16, 'color': '#333333'}
        },
        template='plotly_white',
        height=max(500, 100 + len(features) * 30),  # Dynamic height based on number of features
        margin=dict(l=150, r=40, t=100, b=80),
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Roboto"
        ),
        plot_bgcolor='rgba(240, 240, 240, 0.3)'
    )
    
    # Add annotations for feature ranking
    for i, (feature, imp) in enumerate(zip(sorted_features, sorted_importance)):
        fig.add_annotation(
            x=0, 
            y=feature,
            text=f"{i+1}",
            showarrow=False,
            xshift=-40,
            align='right',
            font=dict(size=12, color='gray'),
            xanchor='center'
        )
    
    return fig

def plot_technical_indicators(df: pd.DataFrame, 
                            ticker: str,
                            date_col: str = 'date',
                            price_col: str = 'close',
                            indicators: Dict[str, List[str]] = None,
                            show_volume: bool = True) -> go.Figure:
    """
    Plot technical indicators with Plotly.
    
    Args:
        df (pd.DataFrame): DataFrame with stock data
        ticker (str): Stock ticker
        date_col (str): Column name for dates
        price_col (str): Column name for price
        indicators (Dict[str, List[str]]): Dictionary mapping indicator types to column names
        show_volume (bool): Whether to show the volume chart
        
    Returns:
        go.Figure: Plotly figure
    """
    if indicators is None:
        indicators = {
            'momentum': ['rsi'],
            'trend': ['macd', 'macd_signal', 'macd_histogram'],
            'volatility': ['bollinger_high', 'bollinger_low', 'atr'],
            'volume': ['obv']
        }
    
    # Filter out empty indicator categories
    indicators = {k: v for k, v in indicators.items() if v}
    
    # Prepare dates
    dates = df[date_col] if date_col in df.columns else df.index
    
    # Count the number of subplots needed
    has_volume = 'volume' in df.columns and show_volume
    active_indicators = sum(1 for inds in indicators.values() if inds)
    n_subplots = 1 + (1 if has_volume else 0) + active_indicators
    
    # Create readable subplot titles
    title_map = {
        'momentum': 'RSI (Momentum)',
        'trend': 'MACD (Trend)',
        'volatility': 'Volatility (BB/ATR)',
        'volume': 'OBV (Volume)'
    }
    
    # Create more prominent subplot titles with spacing
    subplot_titles = [f"<b>{ticker} Price Chart</b>"]
    if has_volume:
        subplot_titles.append(f"<b>{ticker} Volume</b>")
    
    # Add titles for each active indicator category
    for indicator_type in indicators:
        if indicators[indicator_type]:
            subplot_titles.append(f"<b>{title_map.get(indicator_type, indicator_type.title())}</b>")
    
    # Create subplots with appropriate heights
    row_heights = [0.4]  # Price chart
    if has_volume:
        row_heights.append(0.2)  # Volume chart
    
    # Equal height for remaining indicator charts
    if active_indicators > 0:
        indicator_height = 0.4 / active_indicators
        row_heights.extend([indicator_height] * active_indicators)
    
    fig = make_subplots(
        rows=n_subplots, 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.08,  # Increased spacing between subplots
        row_heights=row_heights,
        subplot_titles=subplot_titles
    )
    
    # Add price chart
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=df[price_col],
            name=price_col.title(),
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # Add volume chart if data is available
    current_row = 2
    if has_volume:
        volume_colors = ['green' if df[price_col].iloc[i] >= df[price_col].iloc[i-1] 
                        else 'red' for i in range(1, len(df))]
        volume_colors.insert(0, 'green')  # For the first bar
        
        fig.add_trace(
            go.Bar(
                x=dates,
                y=df['volume'],
                name='Volume',
                marker_color=volume_colors
            ),
            row=current_row, col=1
        )
        current_row += 1
    
    # Add indicators in specific order: momentum, trend, volatility, volume indicators
    for indicator_type in ['momentum', 'trend', 'volatility', 'volume']:
        if indicator_type not in indicators or not indicators[indicator_type]:
            continue
            
        if indicator_type == 'momentum' and 'rsi' in indicators[indicator_type]:
            # RSI chart
            if 'rsi' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=df['rsi'],
                        name='RSI',
                        line=dict(color='purple', width=1.5)
                    ),
                    row=current_row, col=1
                )
                # Add overbought/oversold lines
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1,
                             annotation_text="Overbought", annotation_position="top right")
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1,
                             annotation_text="Oversold", annotation_position="bottom right")
                fig.update_yaxes(title_text="RSI", row=current_row, col=1)
                current_row += 1
                
        elif indicator_type == 'trend' and set(['macd', 'macd_signal', 'macd_histogram']).intersection(set(indicators[indicator_type])):
            # MACD chart
            if all(col in df.columns for col in ['macd', 'macd_signal']):
                # MACD line
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=df['macd'],
                        name='MACD',
                        line=dict(color='blue', width=1.5)
                    ),
                    row=current_row, col=1
                )
                # Signal line
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=df['macd_signal'],
                        name='Signal',
                        line=dict(color='red', width=1.5)
                    ),
                    row=current_row, col=1
                )
                # Histogram
                if 'macd_histogram' in df.columns:
                    fig.add_trace(
                        go.Bar(
                            x=dates,
                            y=df['macd_histogram'],
                            name='Histogram',
                            marker_color=['green' if val >= 0 else 'red' for val in df['macd_histogram']]
                        ),
                        row=current_row, col=1
                    )
                fig.update_yaxes(title_text="MACD", row=current_row, col=1)
                current_row += 1
                
        elif indicator_type == 'volatility':
            has_bb = set(['bollinger_high', 'bollinger_low']).intersection(set(indicators[indicator_type]))
            has_atr = 'atr' in indicators[indicator_type]
            
            # Bollinger Bands
            if has_bb and all(col in df.columns for col in ['bollinger_high', 'bollinger_low']):
                # Price line
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=df[price_col],
                        name=price_col.title(),
                        line=dict(color='blue', width=1)
                    ),
                    row=current_row, col=1
                )
                # Upper band
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=df['bollinger_high'],
                        name='BB Upper',
                        line=dict(color='gray', width=1),
                        legendgroup='bollinger'
                    ),
                    row=current_row, col=1
                )
                # Lower band
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=df['bollinger_low'],
                        name='BB Lower',
                        line=dict(color='gray', width=1),
                        fill='tonexty',
                        fillcolor='rgba(173, 216, 230, 0.2)',
                        legendgroup='bollinger'
                    ),
                    row=current_row, col=1
                )
                
            # ATR (Average True Range)
            if has_atr and 'atr' in df.columns:
                if has_bb:  # If we already have BB bands, add ATR to the same subplot
                    fig.add_trace(
                        go.Scatter(
                            x=dates,
                            y=df['atr'],
                            name='ATR',
                            line=dict(color='orange', width=1.5),
                            yaxis="y2"
                        ),
                        row=current_row, col=1
                    )
                    # Add secondary y-axis for ATR
                    fig.update_layout(
                        yaxis2=dict(
                            title="ATR",
                            overlaying="y",
                            side="right",
                            showgrid=False
                        )
                    )
                else:  # Standalone ATR chart
                    fig.add_trace(
                        go.Scatter(
                            x=dates,
                            y=df['atr'],
                            name='ATR',
                            line=dict(color='orange', width=1.5)
                        ),
                        row=current_row, col=1
                    )
                    
            fig.update_yaxes(title_text="Volatility", row=current_row, col=1)
            current_row += 1
                
        elif indicator_type == 'volume' and 'obv' in indicators[indicator_type]:
            # On-Balance Volume
            if 'obv' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=df['obv'],
                        name='OBV',
                        line=dict(color='green', width=1.5)
                    ),
                    row=current_row, col=1
                )
                fig.update_yaxes(title_text="OBV", row=current_row, col=1)
                current_row += 1
    
    # Update layout
    fig.update_layout(
        height=300 * n_subplots,  # Increased height per subplot
        title=f"{ticker} Technical Analysis",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        margin=dict(t=100, b=50, l=50, r=50)  # Add more margin
    )
    
    # Hide weekend gaps in x-axis
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"])  # hide weekends
        ]
    )
    
    return fig

def plot_portfolio_weights(weights: Dict[str, float], title: str = "Portfolio Allocation") -> go.Figure:
    """
    Plot portfolio weights as a pie chart.
    
    Args:
        weights (Dict[str, float]): Dictionary mapping tickers to weights
        title (str): Chart title
        
    Returns:
        go.Figure: Plotly figure
    """
    # Sort weights by value
    sorted_weights = dict(sorted(weights.items(), key=lambda x: x[1], reverse=True))
    
    # Create pie chart
    fig = go.Figure(
        data=[
            go.Pie(
                labels=list(sorted_weights.keys()),
                values=list(sorted_weights.values()),
                hole=0.4,
                textinfo='label+percent',
                insidetextorientation='radial'
            )
        ]
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=600
    )
    
    return fig

def plot_efficient_frontier_interactive(portfolios_df: pd.DataFrame, 
                                      optimized_portfolios: Dict[str, Dict] = None) -> go.Figure:
    """
    Plot the efficient frontier interactively with Plotly.
    
    Args:
        portfolios_df (pd.DataFrame): DataFrame with portfolio simulations
        optimized_portfolios (Dict[str, Dict]): Dictionary of optimized portfolios
        
    Returns:
        go.Figure: Plotly figure
    """
    fig = go.Figure()
    
    # Add random portfolios
    fig.add_trace(
        go.Scatter(
            x=portfolios_df['Volatility'],
            y=portfolios_df['Return'],
            mode='markers',
            name='Simulated Portfolios',
            marker=dict(
                color=portfolios_df['Sharpe'],
                colorscale='Viridis',
                size=5,
                colorbar=dict(title='Sharpe Ratio'),
                opacity=0.5
            ),
            text=[f"Sharpe: {s:.2f}" for s in portfolios_df['Sharpe']],
            hovertemplate='Return: %{y:.2%}<br>Volatility: %{x:.2%}<br>%{text}'
        )
    )
    
    # Add optimized portfolios
    if optimized_portfolios:
        markers = {'low': 'square', 'medium': 'circle', 'high': 'triangle-up'}
        colors = {'low': 'blue', 'medium': 'green', 'high': 'red'}
        
        for risk_level, portfolio in optimized_portfolios.items():
            fig.add_trace(
                go.Scatter(
                    x=[portfolio['annualized_volatility']],
                    y=[portfolio['annualized_return']],
                    mode='markers',
                    name=f"{risk_level.capitalize()} Risk",
                    marker=dict(
                        color=colors.get(risk_level, 'black'),
                        size=15,
                        symbol=markers.get(risk_level, 'circle')
                    ),
                    text=[f"Sharpe: {portfolio['sharpe_ratio']:.2f}"],
                    hovertemplate='Return: %{y:.2%}<br>Volatility: %{x:.2%}<br>%{text}'
                )
            )
    
    # Update layout
    fig.update_layout(
        title='Portfolio Efficient Frontier',
        xaxis_title='Annualized Volatility',
        yaxis_title='Annualized Return',
        template="plotly_white",
        height=600,
        xaxis=dict(tickformat='.1%'),
        yaxis=dict(tickformat='.1%')
    )
    
    return fig

def plot_correlation_heatmap(returns: pd.DataFrame) -> go.Figure:
    """
    Plot correlation heatmap between stocks.
    
    Args:
        returns (pd.DataFrame): DataFrame with stock returns
        
    Returns:
        go.Figure: Plotly figure
    """
    # Calculate correlation matrix
    corr_matrix = returns.corr()
    
    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu_r',
            zmin=-1,
            zmax=1,
            text=corr_matrix.round(2).values,
            texttemplate='%{text:.2f}',
            colorbar=dict(title='Correlation')
        )
    )
    
    # Update layout
    fig.update_layout(
        title='Stock Returns Correlation Heatmap',
        template="plotly_white",
        height=700,
        width=700
    )
    
    return fig

def generate_animation_data(df: pd.DataFrame, 
                          date_col: str, 
                          price_col: str, 
                          ma_cols: List[str],
                          window_size: int = 90,
                          step: int = 1) -> Dict:
    """
    Generate data for animated stock charts.
    
    Args:
        df (pd.DataFrame): DataFrame with stock data
        date_col (str): Column name for dates
        price_col (str): Column name for price
        ma_cols (List[str]): List of moving average column names
        window_size (int): Size of the sliding window
        step (int): Step size for the animation
        
    Returns:
        Dict: Dictionary with animation data
    """
    if len(df) <= window_size:
        return {'dates': df[date_col], 'prices': df[price_col], 'mas': {ma: df[ma] for ma in ma_cols if ma in df}}
    
    # Create frames for animation
    frames = []
    
    for i in range(window_size, len(df), step):
        frame_df = df.iloc[max(0, i - window_size):i]
        
        frame = {
            'dates': frame_df[date_col],
            'prices': frame_df[price_col],
            'mas': {ma: frame_df[ma] for ma in ma_cols if ma in frame_df}
        }
        
        frames.append(frame)
    
    return frames

def plot_ensemble_predictions(ensemble_result: Dict, ticker: str) -> go.Figure:
    """
    Plot ensemble predictions with confidence intervals.
    
    Args:
        ensemble_result (Dict): Result from ensemble model prediction
        ticker (str): Stock ticker symbol
        
    Returns:
        go.Figure: Plotly figure
    """
    # Extract data from ensemble result
    dates = ensemble_result['dates']
    ensemble_pred = ensemble_result['predictions']
    
    # Create figure
    fig = go.Figure()
    
    # Add ensemble prediction
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=ensemble_pred,
            name='Forecast',
            line=dict(color='blue', width=3),
            hovertemplate='<b>Date</b>: %{x}<br><b>Price</b>: $%{y:.2f}<extra></extra>'
        )
    )
    
    # Add confidence intervals if available
    if 'confidence_intervals' in ensemble_result:
        ci = ensemble_result['confidence_intervals']
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=ci['upper'],
                name='Upper Bound (95% CI)',
                line=dict(color='rgba(0, 0, 255, 0.2)', width=0),
                hoverinfo='skip'
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=ci['lower'],
                name='Lower Bound (95% CI)',
                line=dict(color='rgba(0, 0, 255, 0.2)', width=0),
                fill='tonexty',
                hoverinfo='skip'
            )
        )
    
    # Update layout
    fig.update_layout(
        title=f"{ticker} Price Forecast",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def plot_returns_comparison(df: pd.DataFrame, 
                         ticker: str,
                         date_col: str = 'date',
                         price_col: str = 'close') -> go.Figure:
    """
    Plot stock returns for 1-year, 3-year, and 5-year periods.
    
    Args:
        df (pd.DataFrame): DataFrame with stock data
        ticker (str): Stock ticker
        date_col (str): Column name for dates
        price_col (str): Column name for price
        
    Returns:
        go.Figure: Plotly figure showing return comparisons
    """
    # Create a copy of the dataframe
    data = df.copy()
    
    # Ensure the date column is a datetime
    if date_col in data.columns and not pd.api.types.is_datetime64_any_dtype(data[date_col]):
        data[date_col] = pd.to_datetime(data[date_col])
    
    # Use index if date_col not in dataframe
    if date_col not in data.columns and isinstance(data.index, pd.DatetimeIndex):
        date_series = data.index
    else:
        date_series = data[date_col]
    
    # Get current date and calculate lookback dates
    latest_date = date_series.max()
    
    # Define the time periods to calculate returns
    periods = {
        "1 Year": 365,
        "3 Years": 365 * 3,
        "5 Years": 365 * 5
    }
    
    fig = go.Figure()
    
    # Calculate and plot returns for each period
    for period_name, days in periods.items():
        # Calculate start date for this period
        start_date = latest_date - pd.Timedelta(days=days)
        
        # Filter data for this period
        period_data = data[date_series >= start_date].copy()
        
        if len(period_data) == 0:
            continue
            
        # Calculate cumulative returns (normalized to 100)
        start_price = period_data[price_col].iloc[0]
        period_data['return'] = period_data[price_col] / start_price * 100 - 100
        
        # Add trace for this period
        fig.add_trace(
            go.Scatter(
                x=period_data[date_col] if date_col in period_data.columns else period_data.index,
                y=period_data['return'],
                mode='lines',
                name=f"{period_name} Return",
                line=dict(width=2)
            )
        )
    
    # Calculate total return for each period for annotations
    annotations = []
    
    for period_name, days in periods.items():
        start_date = latest_date - pd.Timedelta(days=days)
        period_data = data[date_series >= start_date]
        
        if len(period_data) > 0:
            start_price = period_data[price_col].iloc[0]
            end_price = period_data[price_col].iloc[-1]
            total_return = (end_price / start_price - 1) * 100
            
            annotations.append(
                dict(
                    x=0.02,
                    y=0.95 - (list(periods.keys()).index(period_name) * 0.07),
                    xref="paper",
                    yref="paper",
                    text=f"{period_name}: {total_return:.2f}%",
                    showarrow=False,
                    font=dict(
                        family="Arial",
                        size=14,
                        color="black"
                    ),
                    align="left",
                    bgcolor="rgba(255, 255, 255, 0.7)",
                    bordercolor="rgba(0, 0, 0, 0.2)",
                    borderwidth=1,
                    borderpad=4
                )
            )
    
    # Update layout
    fig.update_layout(
        title=f"{ticker} Returns Comparison",
        xaxis_title="Date",
        yaxis_title="Return (%)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white",
        annotations=annotations,
        height=600
    )
    
    # Add a horizontal line at y=0
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=1,
        y1=0,
        xref="paper",
        line=dict(
            color="grey",
            width=1,
            dash="dash",
        )
    )
    
    # Hide weekend gaps in x-axis
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"])  # hide weekends
        ]
    )
    
    return fig 