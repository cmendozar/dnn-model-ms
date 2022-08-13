import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 

def get_returns(stock_data):
    return np.log(stock_data/stock_data.shift(1))

def create_features(stock_returns):
    """ features of each stock to add to neural network model"""
    features = pd.DataFrame()
    features['rt1'] = stock_returns.shift(1)
    features['rt5'] = stock_returns.shift(5)
    features['rt22'] = stock_returns.shift(22)
    features['MM5'] = stock_returns.shift(1).rolling(window = 5).mean()
    features['MM22'] = stock_returns.shift(1).rolling(window = 22).mean()
    features.index = stock_returns.index.values
    print(stock_returns.index.values)
    return features.dropna()


df = pd.read_excel('data/data.xlsx',index_col='Date')

returns_ge = get_returns(df.GE)
features_ge = create_features(returns_ge)
print(features_ge)
target_ge = pd.DataFrame(returns_ge)


training_features =  features_ge.loc['2010-03-01':'2019-06-30']
training_target   =  target_ge.loc['2010-03-01':'2019-06-30']

validation_features = features_ge.loc['2019-07-01':'2019-12-31']
validation_target   = target_ge.loc['2019-07-01':'2019-12-31']

test_features = features_ge.loc['2020-01-01':'2021-12-31']
test_target   = target_ge.loc['2020-01-01':'2021-12-31']

plt.style.use('ggplot')
fig = test_target.plot()
fig.plot(training_target)
fig.plot(validation_target)


model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(50,
                            activation = 'relu',
                            input_shape = (5,)
                            ))
model.add(tf.keras.layers.Dropout(0.2))
#model.add(tf.keras.layers.LSTM(cell))
#model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(10))
model.add(tf.keras.layers.Dense(1))
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),loss = "mse")


model.fit(training_features,training_target,
        epochs = 100,
        verbose = 1,
        batch_size = 128,
        validation_data = (validation_features,validation_target)
)

model.save('model/modelo_ge.h5')

