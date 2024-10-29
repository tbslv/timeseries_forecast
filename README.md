
# WORK IN PROGRESS

## timeseries_forecast

pyenv local 3.11.3
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt


Project that developed different models (Random Forest, LSTM - Neural Networks) to forecast the next day's Bitcoin price movement (rise or fall) by incorporating data from gold, oil, S&P 500, and market sentiment.


# feature engineering

1) impute high volume btc trading day - done
2) oil close, low and adj colde outlier day impute with ffill - done
3) Calculate features
    - gold volume outlier - high volume days - done
    - oil volume outlier  - low volume days - done
4) calcuate ta, technical, moving average and custom features
5) get the derivatives of the columns 
6) define high and low golld and sp500 regime defined by bimodel dist
7) log transform dists
8) scale -1,1


