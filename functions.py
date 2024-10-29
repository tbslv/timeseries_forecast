import ta
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
from sklearn.preprocessing import MinMaxScaler

''' DATA FETCHING FUNCTIONS '''

def fetch_data(name, start, end, scale=None):
	data = yf.download(name, start=start, end=end)
	
	if scale:
		scaler = MinMaxScaler()
		data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)
	
	return data




''' MISSING VALUE FUNCTIONS '''


def visualize_nans(data, date_column=None):
	"""
	Investigates and visualizes NaN counts in the DataFrame, including weekday distribution if a date column is provided.
	
	Parameters:
	- data (pd.DataFrame): The DataFrame to analyze.
	- date_column (str, optional): The name of the date column to use for weekday analysis.
	
	Returns:
	- nan_counts (pd.Series): Total NaN counts per column.
	- nan_weekday_counts (pd.Series): NaN counts by weekday (if date_column is provided).
	"""
	
	# Count total NaNs in each column
	nan_counts = data.isnull().sum()
	
	# Check NaN counts by weekday if date column is provided
	if date_column:
		data[date_column] = pd.to_datetime(data[date_column], errors='coerce')
		data['Weekday'] = data[date_column].dt.day_name()
		nan_weekday_counts = data[data.isnull().any(axis=1)]['Weekday'].value_counts()
	else:
		nan_weekday_counts = None
	
	# Plot the total NaN counts for each column
	plt.figure(figsize=(12, 6))
	nan_counts.plot(kind='bar', color='skyblue')
	plt.title('Total NaN Counts per Column')
	plt.xlabel('Columns')
	plt.ylabel('Number of NaNs')
	plt.xticks(rotation=90)
	plt.tight_layout()
	plt.show()
	
	# Plot the weekday distribution of NaNs if weekday data is available
	if nan_weekday_counts is not None:
		plt.figure(figsize=(8, 5))
		nan_weekday_counts.plot(kind='bar', color='orange')
		plt.title('Weekday Distribution of NaNs')
		plt.xlabel('Weekday')
		plt.ylabel('Number of NaNs')
		plt.xticks(rotation=90)
		plt.tight_layout()
		plt.show()
	
	return nan_counts, nan_weekday_counts


''' DATA ENGINEERING FUNCTIONS '''
 
def add_target_column (data, name1,name2,variable):
	tmp = data[variable].diff().shift(-1).values
	data.loc[:,name1] = [ 1 if tmp[i] > 0 else 0 for i in range(tmp.shape[0])]
	data.loc[:,name2] = tmp
	return data


def add_ta_features(data, suffixes=['_btc', '_oil', '_gold', '_sp500'], window_sizes=[10]):
	"""
	Adds technical analysis features (OBV, RSI, ROC, SMA, WMA, KAMA) for each specified asset type 
	in the DataFrame.

	Parameters:
	- data (pd.DataFrame): The DataFrame containing asset price columns.
	- suffixes (list of str): List of suffixes representing different asset types.
	- window_sizes (list of int): List of window sizes for calculating RSI, ROC, SMA, WMA, and KAMA.

	Returns:
	- pd.DataFrame: The DataFrame with new technical analysis feature columns.
	"""
	new_columns = {}  # Dictionary to store new columns for efficient addition

	for suffix in suffixes:
		# Define column names based on the suffix
		close_col = f'Close{suffix}'
		volume_col = f'Volume{suffix}'
		
		# Ensure required columns exist in the DataFrame
		if close_col in data.columns and volume_col in data.columns:
			# OBV (On Balance Volume)
			obv = ta.volume.OnBalanceVolumeIndicator(close=data[close_col], volume=data[volume_col])
			new_columns[f'OBV{suffix}'] = obv.on_balance_volume()
			
			for window in window_sizes:
				# RSI
				rsi = ta.momentum.RSIIndicator(close=data[close_col], window=window, fillna=True)
				new_columns[f'RSI_{window}{suffix}'] = rsi.rsi()
				
				# ROC
				roc = ta.momentum.ROCIndicator(close=data[close_col], window=window, fillna=True)
				new_columns[f'ROC_{window}{suffix}'] = roc.roc()
				
				# SMA (Simple Moving Average)
				sma = ta.trend.SMAIndicator(close=data[close_col], window=window, fillna=True)
				new_columns[f'SMA_{window}{suffix}'] = sma.sma_indicator()
				
				# WMA (Weighted Moving Average)
				wma = ta.trend.WMAIndicator(close=data[close_col], window=window, fillna=True)
				new_columns[f'WMA_{window}{suffix}'] = wma.wma()
				
				# KAMA (Kaufman's Adaptive Moving Average)
				kama = ta.momentum.KAMAIndicator(close=data[close_col], window=window, fillna=True)
				new_columns[f'KAMA_{window}{suffix}'] = kama.kama()

	# Concatenate all new columns at once to the original DataFrame
	data = pd.concat([data, pd.DataFrame(new_columns)], axis=1)
	
	return data

def add_ratio_trend_features(data, target,variable, window_sizes = [2,5,10,20,40,60]):
	
	
	for hor in window_sizes:
		
		rolling_average = data[variable].rolling(hor).mean()
		
		ratio_col = 'ratio_{}'.format(hor)
		data.loc[:,ratio_col] = data[variable]/rolling_average
		
		if target in data.columns:
			
			trend_col = 'rTrend_{}'.format(hor)
			data.loc[:,trend_col] = data[target].shift(1).rolling(hor).sum(
					)
			
	return data
	

def add_custom_features(data, suffixes=['_btc', '_oil', '_gold', '_sp500']):
	"""
	Adds custom features (volatility, high-close change, low-close change, and percent price change)
	to each specified asset type in the DataFrame.

	Parameters:
	- data (pd.DataFrame): The DataFrame containing asset price columns.
	- suffixes (list of str): List of suffixes representing different asset types.

	Returns:
	- pd.DataFrame: The DataFrame with new custom feature columns.
	"""
	# Dictionary to store new columns to be added later
	new_columns = {}

	for suffix in suffixes:
		# Define column names based on the suffix
		high_col = f'High{suffix}'
		low_col = f'Low{suffix}'
		close_col = f'Close{suffix}'
		
		# Ensure columns exist in the DataFrame before applying calculations
		if all(col in data.columns for col in [high_col, low_col, close_col]):
			# Calculate new feature columns and store in the dictionary
			new_columns[f'volatility{suffix}'] = (data[high_col] - data[low_col]) / data[low_col] * 100
			new_columns[f'hc_change{suffix}'] = (data[close_col] - data[high_col]) / data[high_col] * 100
			new_columns[f'lc_change{suffix}'] = (data[close_col] - data[low_col]) / data[low_col] * 100
			new_columns[f'price_change_percent{suffix}'] = (data[close_col].pct_change() * 100).round(2)

	# Concatenate new columns to the DataFrame at once
	data = pd.concat([data, pd.DataFrame(new_columns)], axis=1)
	
	return data

def log_norm_scale(df, log_norms=None, scaler = None,non_scalables=None):
    """
    Performs log-normal transformation on specified columns and Min-Max scaling on the rest.
    
    Parameters:
    - df: pandas DataFrame to process.
    - log_norms: list of columns to apply log-normal transformation (default: None).
    - non_scalables: list of columns to exclude from Min-Max scaling (default: None).
    
    Returns:
    - Processed DataFrame with log-normal transformation and Min-Max scaling applied.
    """
    
    # Handle default values
    log_norms = log_norms or []
    non_scalables = non_scalables or []
    
    # Copy the DataFrame to avoid modifying the original data
    df_processed = df.copy()

    # Apply log-normal transformation
    for col in log_norms:
        if col in df_processed.columns:
            # Add a small constant to avoid log(0)
            df_processed[col] = np.log1p(df_processed[col])
    
    if scaler:
	    # Define columns to scale
	    columns_to_scale = df_processed.columns.difference(non_scalables)
	    
	    # Initialize MinMaxScaler
	    scaler = MinMaxScaler()
	    
	    # Apply Min-Max scaling
	    df_processed[columns_to_scale] = scaler.fit_transform(df_processed[columns_to_scale])
    
    	return df_processed

    else:
    	return df_processed

def add_technical_indicators(df, suffixes=['_btc', '_oil', '_gold', '_sp500'], rsi_periods=[10, 14, 20]):
	"""
	Adds RSI, ROC, and OBV technical indicators to the DataFrame for each specified asset type and RSI period.
	
	Parameters:
	- df (pd.DataFrame): The DataFrame containing price and volume data.
	- suffixes (list of str): List of suffixes for different asset types.
	- rsi_periods (list of int): List of periods for RSI calculations.
	
	Returns:
	- pd.DataFrame: The DataFrame with new columns for RSI, ROC, and OBV indicators for each asset type and RSI period.
	"""
	for suffix in suffixes:
		# Define column names based on the suffix
		close_col = f'Close{suffix}'
		volume_col = f'Volume{suffix}'
		
		# Ensure required columns exist in the DataFrame before applying calculations
		if close_col in df.columns and volume_col in df.columns:
			# OBV
			obv = ta.volume.OnBalanceVolumeIndicator(df[close_col], df[volume_col])
			df.loc[:, f'OBV{suffix}'] = obv.on_balance_volume()
			
			for period in rsi_periods:
				# RSI
				rsi = ta.momentum.RSIIndicator(df[close_col], window=period, fillna=True)
				df.loc[:, f'RSI_{period}{suffix}'] = rsi.rsi()
				
				# ROC with each RSI period
				roc = ta.momentum.ROCIndicator(df[close_col], window=period, fillna=True)
				df.loc[:, f'ROC_{period}{suffix}'] = roc.roc()
	
	return df


def detect_outliers(df, method='iqr', threshold=1.5):
    """
    Detects outliers in each column of a DataFrame based on the specified method.

    Parameters:
    - df: pandas DataFrame to analyze.
    - method: Method to use for outlier detection ('iqr' or 'z-score'). Default is 'iqr'.
    - threshold: Threshold to define outliers. For 'iqr', it's the multiplier of IQR; 
                 for 'z-score', it's the z-score threshold. Default is 1.5 for IQR, 3 for Z-score.

    Returns:
    - Dictionary with column names as keys and lists of row indices with outliers as values.
    """
    outliers = {}

    # Loop through each column in the DataFrame
    for col in df.select_dtypes(include=[np.number]).columns:  # Only process numeric columns
        if method == 'iqr':
            # Calculate IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outliers as values outside the range [Q1 - threshold*IQR, Q3 + threshold*IQR]
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outlier_indices = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
        
        elif method == 'z-score':
            # Calculate Z-scores
            mean = df[col].mean()
            std_dev = df[col].std()
            z_scores = (df[col] - mean) / std_dev
            
            # Define outliers as values with Z-scores greater than the threshold
            outlier_indices = df[(np.abs(z_scores) > threshold)].index.tolist()
        
        else:
            raise ValueError("Invalid method. Choose 'iqr' or 'z-score'.")
        
        # Add to the dictionary if outliers are found
        if outlier_indices:
            outliers[col] = outlier_indices
    
    return outliers

