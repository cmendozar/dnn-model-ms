import os
import time 
import pandas as pd
import numpy as np
import pandas_datareader.data as web
from datetime import datetime,timedelta

def create_dataframe(stocks,start_date,final_date):
    """ Download the stock data from yahoo finance and create a dataframe.
    Parametres: Stocks: list a stocks, the others are self-explanatory.
    """
    print("The process has started")
    df = pd.DataFrame()
    s = time.time()
    for stock in stocks: 
        df[stock] = web.DataReader(stock, 
                                   'yahoo', 
                                   start=start_date, 
                                   end=final_date).Close  #download the Close price for stocks 
        print(f"The {stock} stock is already downloaded")
    f = time.time()
    print("Elapsed time: {} secs".format(round(f-s,1)))
    return df

def get_returns(stock_data):
    return np.log(stock_data/stock_data.shift(1))

def create_features(stock_returns):
    """ Features of each stock to add to neural network model"""
    features = pd.DataFrame()
    features['rt1'] = stock_returns.shift(1)
    features['rt5'] = stock_returns.shift(5)
    features['rt22'] = stock_returns.shift(22)
    features['MM5'] = stock_returns.shift(1).rolling(window = 5).mean()
    features['MM22'] = stock_returns.shift(1).rolling(window = 22).mean()
    #features.index = stock_returns.index.values
    return features.dropna()

def create_features_ahead(stock_returns):
    """ Features of each stock to add to neural network model"""
    features = pd.DataFrame()
    features['rt1'] = stock_returns
    features['rt5'] = stock_returns.shift(4)
    features['rt22'] = stock_returns.shift(21)
    features['MM5'] = stock_returns.rolling(window = 5).mean()
    features['MM22'] = stock_returns.rolling(window = 22).mean()
    #features.index = stock_returns.index.values
    return features.dropna()

def split_data(data):
    """ Split data between training, validation and test set"""
    training   =  data.loc['2010-03-01':'2019-06-30']
    validation = data.loc['2019-07-01':'2019-12-31']
    test       = data.loc['2020-01-01':'2021-12-31']
    return training,validation,test

def create_input(body):
    delta = str(datetime.strptime(body["day_to_predict"],"%Y-%m-%d") + timedelta(days = -50))
    df = create_dataframe(['GE'],delta,body["day_to_predict"])
    returns = get_returns(df.GE)#.dropna()
    features = create_features_ahead(returns)
    last_day = features.tail(2).index.values[0]
    print(f"The input is setted at {last_day}")
    input_model = features[features.index == datetime.strptime(body["day_to_predict"],"%Y-%m-%d") + timedelta(days = -1)]
    if input_model.empty is True:
        return features.tail(1).values
    return input_model.values


if __name__ == "__main__":
    body = { "day_to_predict":  "2022-06-09"}
    print(create_input(body))
    df = create_dataframe(['GE','GOOG','IBM'],'2010-01-01','2021-12-31')
    print(os.path.isdir('data'))
    df.to_excel('../../../data/data2.xlsx')
    