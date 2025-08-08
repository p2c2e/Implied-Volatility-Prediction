# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 21:53:29 2024

@author: he_98
"""

import numpy as np
import pandas as pd
import os
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF C++ logs (INFO/WARNING)
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Reduce TensorFlow Python-side logs
warnings.filterwarnings("ignore", message=".*tf.function retracing.*")  # Silence retracing warnings
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense, Dropout
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# Loading the dataset
csv_file_path = './CSVs used for Code/main_columns_filtered.csv'

df = pd.read_csv(csv_file_path)
df = df.drop(columns=['Exchange Date'])

# Grouping the dataframe by 'Strike Price' and creating a separate dataset for each group
grouped = df.groupby('Strike Price')

# Creating a dictionary to hold the datasets
datasets = {}
for name, group in grouped:
    datasets[name] = group

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_datasets = {}

for strike_price, dataset in datasets.items():
    # Keeping all columns as features except for the target column
    X = dataset.drop(dataset.columns[1], axis=1)
    y = dataset.iloc[:, 1].values
    
    # Scaling the features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)
    scaled_datasets[strike_price] = (X_scaled, y)


look_back = 5  # Number of days to look back to make a prediction
look_forward = 5  # Number of days ahead to predict

prepared_datasets = {}

for strike_price, (X_scaled, y) in scaled_datasets.items():
    X_prepared = []
    y_prepared = []

    # We stop at -look_forward to avoid going out of bounds
    for i in range(len(X_scaled) - look_back - look_forward):
        X_prepared.append(X_scaled[i:(i + look_back), :])
        y_prepared.append(y[i + look_back + look_forward - 1])
    
    # Reshaping the input to be [samples, time steps, features]
    X_prepared = np.array(X_prepared, dtype=np.float32).reshape(-1, look_back, X_scaled.shape[1])
    y_prepared = np.array(y_prepared, dtype=np.float32)
    
    prepared_datasets[strike_price] = (X_prepared, y_prepared)


# Using as an example 'prepared_datasets[5000]', which contains the time-ordered sequences for strike price 5000
X_5000, y_5000 = prepared_datasets[5000]

# Defining the train-test split index
train_size = int(len(X_5000) * 0.8)

# Splitting the data while maintaining the time series order
X_train_5000, X_test_5000 = X_5000[:train_size], X_5000[train_size:]
y_train_5000, y_test_5000 = y_5000[:train_size], y_5000[train_size:]

# Defining the LSTM model in a function to use with KerasRegressor
def build_lstm_model(input_shape, lstm_units=64, dropout_rate=0.2):
    # Use an explicit Input layer to avoid passing input_shape directly to RNN layers,
    # which triggers the UserWarning from Keras.
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units=lstm_units, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(units=int(lstm_units/2)),
        Dense(int(lstm_units/4), activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Wrapping the model into KerasRegressor (scikeras)
# scikeras expects the model callable via the `model` argument and model build parameters
# are passed with the `model__` prefix (used below in the grid). Also set verbose=0 for silent runs.
model_wrapper = KerasRegressor(
    model=build_lstm_model,
    model__input_shape=(X_train_5000.shape[1], X_train_5000.shape[2]),
    verbose=0,
    fit__shuffle=False  # keep batch structure stable to reduce retracing
)

# Defining the grid search parameters
# Note: model hyperparameters must be prefixed with 'model__' for scikeras so they are forwarded
# into the model-building callable.
param_grid = {
    'model__lstm_units': [32, 64, 128],
    'model__dropout_rate': [0.1, 0.2, 0.3],
    'epochs': [50, 100],
    'batch_size': [16, 32]
}

# Creating GridSearchCV
grid = GridSearchCV(estimator=model_wrapper, param_grid=param_grid, n_jobs=1, cv=3, scoring='neg_mean_squared_error')

# Fitting the grid search
grid_result = grid.fit(X_train_5000, y_train_5000)

# Printing the best configuration and its score
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# Evaluating the best model on the test set
# scikeras stores the underlying Keras model in the `model_` attribute after fitting
best_model = grid_result.best_estimator_.model_
predictions = best_model.predict(X_test_5000)
mse = mean_squared_error(y_test_5000, predictions)
print(f"Test MSE: {mse}")
