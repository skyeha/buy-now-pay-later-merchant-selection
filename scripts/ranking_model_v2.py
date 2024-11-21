from pyspark.sql import functions as F, SparkSession, Window
from pyspark.ml.functions import vector_to_array
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, StandardScaler,Normalizer
from pyspark import SparkContext
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns

def forecast_revenue(df):
    """
        This function use a LSTM Neural Network to forecast revnenue 3 period ahead.
    """
    # Define the features
    features = ["revenue_lag_1", "revenue_lag_2", "revenue_lag_3", "revenue_growth_lag_1", "revenue_growth_lag_2"]

    # Split the data into train and test
    train_df, test_df = df.randomSplit([0.8, 0.2], seed = 30032)

    # Transform into numpy array to reshape for inpu
    X_train = np.array(train_df.select(features).collect())
    y_train = np.array(train_df.select("revenue").collect())
    
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    # Define our LSTM model
    forecaster = Sequential()
    
    # Add the first layer
    forecaster.add(LSTM(units = 50, activation = 'relu', return_sequences = True, input_shape = (X_train.shape[1],1)))
    # forecaster.add(Dropout(rate = 0.15, seed = 30032))

    # Add our second layer
    forecaster.add(LSTM(units = 50, return_sequences = True, activation = 'relu'))
    # forecaster.add(Dropout(rate = 0.15, seed= 30032))

    # Add third layer
    forecaster.add(LSTM(units = 32, activation = 'relu'))

    # Add our ouput layer
    forecaster.add(Dense(1))

    # optimizer = Adam(learning_rate=0.0005)
    forecaster.compile(optimizer='rmsprop', loss='mse')
    forecaster.fit(X_train, y_train, epochs = 200, verbose=0, shuffle=False)

    last_values = X_train[-1].reshape(1, X_train.shape[1], 1)  
    input_values = generate_input(inputCol=last_values, prediction=None, period=0, df = df)
    
    predictions = []

    print(f"Initial input values: {input_values}")
    for i in range(1,4):  # Predict 3 periods ahead
        prediction = forecaster.predict(input_values)
        predictions.append(prediction[0][0])  # Append predicted value
        input_values = generate_input(inputCol=input_values, prediction=prediction, period = i, df = df)
        
    return predictions


def generate_input(inputCol, prediction, period, df):
    """"
        This functions prepare the input for the LSTM to forecast revenue
    """
    new_input = inputCol.copy()
    # Shifting existing lags
    new_input[0][2] = new_input[0][1]
    new_input[0][1] = new_input[0][0]

    if period == 0:
        new_input[0][0] = df.select('revenue').tail(1)[0][0] 
    else:
        new_input[0][0] = prediction[0][0] # use predicted revenue in period 1 to predict revenue in period 2
        
    new_input[0][4] = new_input[0][3]
    new_input[0][3] = (new_input[0][0] - new_input[0][1]) / new_input[0][1] # compute new revenue growth lag

    return new_input

def generate_num_order_weight(mean, observed):
    return (1/(1+np.exp(-(observed-mean))))




