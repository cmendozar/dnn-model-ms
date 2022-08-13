import pandas as pd
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt 


def get_returns(stock_data):
    return np.log(stock_data/stock_data.shift(1))

def create_features(stock_returns):
    """ features of each stock to add to neural network model"""
    features = pd.DataFrame()
    features['rt1']  = stock_returns.shift(1)
    features['rt5']  = stock_returns.shift(5)
    features['rt22'] = stock_returns.shift(22)
    features['MM5']  = stock_returns.shift(1).rolling(window = 5).mean()
    features['MM22'] = stock_returns.shift(1).rolling(window = 22).mean()
    features.index = stock_returns.index.values
    return features.dropna()

def split_data(data):
    training =  data.loc['2010-03-01':'2019-06-30']
    validation= data.loc['2019-07-01':'2019-12-31']
    test  = data.loc['2020-01-01':'2021-12-31']
    return training,validation,test


df = pd.read_excel('data/data.xlsx',index_col='Date')
returns  = get_returns(df.GE)
features = create_features(returns)
target   = returns 

_,_,test_features = split_data(features)
_,_,test_target   = split_data(target)


model = tf.keras.models.load_model('src/model/modelo_ge.h5')

forecast = pd.DataFrame(model.predict(test_features.values))

forecast.index = test_target.index

plt.style.use('ggplot')
graph = plt.figure()
plt.plot(test_target.index,test_target.values,label = 'real')
plt.plot(forecast.index,forecast.values,label = 'forecast')
plt.legend()
graph.savefig('results.png')


results = pd.concat([test_target,forecast],axis = 1,ignore_index=True)
results.to_excel('resutls.xlsx')
