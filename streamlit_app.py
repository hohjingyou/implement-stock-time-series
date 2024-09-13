
import streamlit as st

# Title

st.title("Predicting Financial Market Stock Trends by Implementing Time Series Forecasting Techniques")
st.title("Made By: HOH JING YOU     TP067120")





# USER SELECTION ON DATASET

st.header("Select the Dataset You Want to Analyze")
choice = st.selectbox("Choose a company:", ["IBM", "Coca-Cola"])





# DATA PRE-PROCESSING

st.header("DATA PRE-PROCESSING")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Load the dataset
if choice == 'IBM':
    file_path = 'ibm.us.txt'
elif choice == 'Coca-Cola':
    file_path = 'ko.us.txt'
ibm_data = pd.read_csv(file_path)

# Preview of the dataset
st.subheader("Preview of the Dataset")
st.write(ibm_data.head())

# Display the structure of the dataset
st.subheader("Dataset Information")
buffer = io.StringIO()
ibm_data.info(buf=buffer)
s = buffer.getvalue()
st.text(s)

# Check for missing values
st.subheader("Missing Values in Each Column")
st.write(ibm_data.isnull().sum())

# Descriptive statistics
st.subheader("Descriptive Statistics")
st.write(ibm_data.describe())

# Convert 'Date' to datetime format and set it as the index
ibm_data['Date'] = pd.to_datetime(ibm_data['Date'])
ibm_data.set_index('Date', inplace=True)

# Plotting the closing price over time
st.subheader(f'Closing Price of {choice} Stock Over Time')
st.line_chart(ibm_data['Close'])

# Plotting the volume traded over time
st.subheader(f'Volume of {choice} Stock Traded Over Time')
st.line_chart(ibm_data['Volume'])

# Pairplot to see the relationships between different numerical variables
st.subheader(f'Pairplot of {choice} Stock Data')
sns.pairplot(ibm_data)
st.pyplot()

# Boxplot for outlier detection
st.subheader('Boxplot for Numerical Columns')
plt.figure(figsize=(14, 7))
sns.boxplot(data=ibm_data[['Open', 'High', 'Low', 'Close', 'Volume']])
st.pyplot()

# Correlation matrix
st.subheader(f'Correlation Matrix of {choice} Stock Data')
plt.figure(figsize=(10, 8))
sns.heatmap(ibm_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
st.pyplot()





# DATA UNDERSTANDING

st.header("DATA UNDERSTANDING")

# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the IBM dataset
if choice == 'IBM':
    ibm_df = pd.read_csv('ibm.us.txt')
elif choice == 'Coca-Cola':
    ibm_df = pd.read_csv('ko.us.txt')

# Convert the Date column to datetime format
ibm_df['Date'] = pd.to_datetime(ibm_df['Date'])

# Ensure the dataframe is sorted by date
ibm_df = ibm_df.sort_values('Date')

# Display the first few rows to confirm the data
st.subheader("First Few Rows of the Dataset")
st.write(ibm_df.head())

# Plot history of the close price of IBM
st.subheader(f'{choice} Adjusted Close Price History')
st.line_chart(ibm_df['Close'])

# Validate there are no duplicate dates
st.subheader("Is Date column unique?")
st.write(ibm_df['Date'].is_unique)

# Exclude specific periods if needed (e.g., exclude 2020 data)
# Uncomment the following line if you want to exclude 2020 data
# ibm_df = ibm_df.loc[ibm_df['Date'].dt.year < 2020]

# Function to calculate daily returns
def calculate_returns(prices):
    return prices.pct_change()

# Compute daily return
ibm_df['day_return'] = calculate_returns(ibm_df['Close'])

# Display the first few rows with returns
st.subheader("Daily Returns")
st.write(ibm_df[['Date', 'Close', 'day_return']].head())

# Plot the distribution of daily returns
st.subheader(f'Distribution of Daily Returns - {choice}')
sns.histplot(ibm_df['day_return'].dropna(), kde=True)
st.pyplot()

# Add a column for the day of the week
ibm_df['weekday'] = ibm_df['Date'].dt.day_name()

# Create a barplot showing average daily return by weekday
st.subheader(f'{choice} Daily Return by Day of the Week')
sns.barplot(x='weekday', y='day_return', data=ibm_df)
st.pyplot()

# Set Date as index for time series decomposition
ibm_ts = ibm_df.set_index('Date')

# Decompose the time series (adjust the frequency as needed), Plot the decomposition results
st.subheader(f'Time Series Decomposition of {choice} Stock Data')
result = seasonal_decompose(ibm_ts['Close'], model='multiplicative', period=365)
result.plot()
st.pyplot()

# Plot autocorrelation for IBM
st.subheader(f'Autocorrelation of {choice} Adjusted Close Price')
autocorrelation_plot(ibm_ts['Close'])
st.pyplot()





# ARIMA MODEL BUILDING

st.header("ARIMA MODEL BUILDING")

# Step 1: Import the Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 2: Load the Dataset
if choice == 'IBM':
    file_path = 'ibm.us.txt'
elif choice == 'Coca-Cola':
    file_path = 'ko.us.txt'
df = pd.read_csv(file_path)

# Step 3: Parse dates and set date as the index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Step 4: Focus on the 'Close' price for prediction
data = df['Close']

# Step 5: Visualize the Data
st.subheader(f"{choice} Stock Close Price Over Time")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(data, label=f'{choice} Close Price')
ax.set_xlabel('Date')
ax.set_ylabel('Close Price')
ax.set_title(f'{choice} Stock Prices Over Time')
ax.legend()
st.pyplot(fig)

# Step 6: Fit the ARIMA Model
# Define the ARIMA model (p, d, q) order can be adjusted
model = ARIMA(data, order=(5, 1, 0))  # (p, d, q)
fitted_model = model.fit()

# Print model summary
st.subheader("ARIMA Model Summary")
st.write(fitted_model.summary())

# Step 7: Make Predictions for the Next 10 Days
forecast = fitted_model.forecast(steps=10)

# Print the forecasted values
st.subheader("Forecasted Prices for the Next 10 Days:")
st.write(forecast)

# Step 8: Plot the Forecasted Prices along with the Actual Data
st.subheader("Forecasted vs Actual Prices")
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(data[-100:], label='Actual Prices') # Plotting the last 100 days for comparison
ax2.plot(pd.date_range(start=data.index[-1], periods=11, freq='B')[1:], forecast, label='Forecasted Prices', color='red')
ax2.set_xlabel('Date')
ax2.set_ylabel('Close Price')
ax2.set_title(f'{choice} Stock Price Forecast')
ax2.legend()
st.pyplot(fig2)

# Step 9: Generate predictions on the training data (in-sample predictions)

st.subheader("Model Evaluation Metrics")

train_predictions = fitted_model.predict(start=0, end=len(data)-1)

# Step 10: Calculate the evaluation metrics

# Mean Absolute Error (MAE)
mae = mean_absolute_error(data, train_predictions)

# Mean Squared Error (MSE)
mse = mean_squared_error(data, train_predictions)

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

# Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((data - train_predictions) / data)) * 100

# R-squared
r2 = r2_score(data, train_predictions)

# Akaike Information Criterion (AIC)
aic = fitted_model.aic

# Print the evaluation metrics
st.write(f"Mean Absolute Error (MAE): {mae}")
st.write(f"Mean Squared Error (MSE): {mse}")
st.write(f"Root Mean Squared Error (RMSE): {rmse}")
st.write(f"Mean Absolute Percentage Error (MAPE): {mape}")
st.write(f"R-squared: {r2}")
st.write(f"Akaike Information Criterion (AIC): {aic}")





# CNN MODEL BUILDING

st.header("CNN MODEL BUILDING")

# Step 1: Import the Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 2: Load the Dataset
if choice == 'IBM':
    file_path = 'ibm.us.txt'
elif choice == 'Coca-Cola':
    file_path = 'ko.us.txt'
df = pd.read_csv(file_path)

# Step 3: Parse dates and set date as the index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Step 4: Focus on the 'Close' price for prediction
data = df['Close']

# Step 5: Normalize the Data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data.values.reshape(-1, 1))

# Step 6: Create Sequences for Training
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 60  # Using the last 60 days to predict the next day's price
X, y = create_sequences(data_scaled, seq_length)

# Step 7: Split the Data into Training and Testing Sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Reshape X for the CNN model (samples, time steps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Step 8: Build the CNN Model
model = Sequential()

# Add convolutional layers
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(seq_length, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))

# Add more convolutional layers
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))

# Flatten the output
model.add(Flatten())

# Add dense layers
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))  # Output layer for predicting the next price

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Print model summary
st.subheader("CNN Model Summary")
st.write(model.summary())

# Step 9: Train the Model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Step 10: Evaluate and Visualize the Results
# Predicting on the test set
predicted_prices = model.predict(X_test)

# Inverse transform the predicted prices to the original scale
predicted_prices = scaler.inverse_transform(predicted_prices)

# Inverse transform the actual prices to the original scale
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plotting the results
st.subheader("Actual vs Predicted Prices")
plt.figure(figsize=(10, 6))
plt.plot(actual_prices, color='blue', label='Actual Prices')
plt.plot(predicted_prices, color='red', label='Predicted Prices')
plt.title(f'{choice} Stock Price Prediction - CNN')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
st.pyplot(plt.gcf())

# Step 11: Making Future Predictions
# Example: Predicting the next 10 days
last_sequence = data_scaled[-seq_length:]
next_10_days = []

for _ in range(10):
    next_price = model.predict(last_sequence.reshape((1, seq_length, 1)))[0, 0]
    next_10_days.append(next_price)
    last_sequence = np.append(last_sequence[1:], next_price).reshape(-1, 1)

# Inverse transform the future prices
next_10_days = scaler.inverse_transform(np.array(next_10_days).reshape(-1, 1))

# Print the forecasted values
st.subheader("Forecasted Prices for the Next 10 Days:")
st.write(next_10_days)

# Plot the future predictions
st.subheader("Forecasted vs Actual Prices")
plt.figure(figsize=(10, 6))
plt.plot(data.index[-100:], data.values[-100:], color='blue', label='Actual Prices')
plt.plot(pd.date_range(start=data.index[-1], periods=11, freq='B')[1:], next_10_days, color='red', label='Forecasted Prices')
plt.title(f'{choice} Stock Price Forecast - Next 10 Days')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
st.pyplot(plt.gcf())

# Step 12: Evaluation Metrics

st.subheader("Model Evaluation Metrics")

# Mean Absolute Error (MAE)
mae = mean_absolute_error(actual_prices, predicted_prices)
st.write(f"Mean Absolute Error (MAE): {mae}")

# Mean Squared Error (MSE)
mse = mean_squared_error(actual_prices, predicted_prices)
st.write(f"Mean Squared Error (MSE): {mse}")

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
st.write(f"Root Mean Squared Error (RMSE): {rmse}")

# Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
st.write(f"Mean Absolute Percentage Error (MAPE): {mape}%")

# R-squared (R²)
r_squared = r2_score(actual_prices, predicted_prices)
st.write(f"R-squared (R²): {r_squared}")

# Akaike Information Criterion (AIC)
n = len(y_test)  # number of observations
k = model.count_params()  # number of model parameters
aic = n * np.log(mse) + 2 * k
st.write(f"Akaike Information Criterion (AIC): {aic}")

# Display all results together
evaluation_results = {
    "Mean Absolute Error (MAE)": mae,
    "Mean Squared Error (MSE)": mse,
    "Root Mean Squared Error (RMSE)": rmse,
    "Mean Absolute Percentage Error (MAPE)": mape,
    "R-squared (R²)": r_squared,
    "Akaike Information Criterion (AIC)": aic
}

st.write("\nEvaluation Results:")
for metric, value in evaluation_results.items():
    st.write(f"{metric}: {value}")





# RNN MODEL BUILDING

st.header("RNN MODEL BUILDING")

# Step 1: Import the Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 2: Load the Dataset
if choice == 'IBM':
    file_path = 'ibm.us.txt'
elif choice == 'Coca-Cola':
    file_path = 'ko.us.txt'
df = pd.read_csv(file_path)

# Step 3: Parse dates and set date as the index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Step 4: Focus on the 'Close' price for prediction
data = df['Close'].values
data = data.reshape(-1, 1)

# Step 5: Scale the Data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Step 6: Create Training and Test Datasets
train_size = int(len(scaled_data) * 0.80)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Create the training dataset
def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 60  # Look back 60 days
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Step 7: Build the LSTM Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Step 8: Compile the Model
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 9: Train the Model
model.fit(X_train, y_train, batch_size=1, epochs=1)

# Step 10: Make Predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform to get actual prices
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Step 11: Plot the Results

st.subheader("Actual vs Training vs Test Prices")

fig, ax = plt.subplots(figsize=(10, 6))
# Plot the actual prices
ax.plot(data, label='Actual Prices')

# Plot the training predictions
train_plot = np.empty_like(data)
train_plot[:, :] = np.nan
train_plot[time_step:len(train_predict) + time_step, :] = train_predict
ax.plot(train_plot, label='Training Predictions')

# Plot the test predictions
test_plot = np.empty_like(data)
test_plot[:, :] = np.nan
test_plot[len(train_predict) + (time_step * 2) + 1:len(data) - 1, :] = test_predict
ax.plot(test_plot, label='Test Predictions')

plt.title(f'{choice} Stock Price Prediction using LSTM')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
st.pyplot(fig)

# Step 12: Make Future Predictions
x_input = test_data[-time_step:].reshape(1, -1)
temp_input = list(x_input)
temp_input = temp_input[0].tolist()

# Predicting the next 10 days
lst_output = []
n_steps = time_step
i = 0
while i < 10:
    if len(temp_input) > n_steps:
        x_input = np.array(temp_input[1:])
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.append(yhat[0][0])
        temp_input = temp_input[1:]
        lst_output.append(yhat[0][0])
        i += 1
    else:
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.append(yhat[0][0])
        lst_output.append(yhat[0][0])
        i += 1

# Inverse transform to get actual prices
lst_output = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1))

# Plot the Future Predictions

st.subheader("Forecasted vs Actual Prices")

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(np.arange(len(data)), data, label='Actual Prices')
ax2.plot(np.arange(len(data), len(data) + 10), lst_output, label='Future Predictions', color='red')
plt.title(f'{choice} Stock Price Future Prediction')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
st.pyplot(fig2)

# Step 13: Evaluate the Model

st.subheader("Model Evaluation Metrics")

# Mean Absolute Error (MAE)
mae_train = mean_absolute_error(y_train, train_predict)
mae_test = mean_absolute_error(y_test, test_predict)

# Mean Squared Error (MSE)
mse_train = mean_squared_error(y_train, train_predict)
mse_test = mean_squared_error(y_test, test_predict)

# Root Mean Squared Error (RMSE)
rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)

# Mean Absolute Percentage Error (MAPE)
mape_train = np.mean(np.abs((y_train - train_predict) / y_train)) * 100
mape_test = np.mean(np.abs((y_test - test_predict) / y_test)) * 100

# R-squared (R²)
r2_train = r2_score(y_train, train_predict)
r2_test = r2_score(y_test, test_predict)

# Akaike Information Criterion (AIC)
def calculate_aic(y_true, y_pred, num_params):
    resid = y_true - y_pred
    sse = np.sum(np.square(resid))
    aic = len(y_true) * np.log(sse / len(y_true)) + 2 * num_params
    return aic

# Number of model parameters (e.g., 50 LSTM neurons in each layer + biases + Dense layers)
num_params = 50 * 50 * 2 + 50 * 25 + 25 + 1 + 50 + 25 + 1 + 50 + 1  # Example calculation, adjust as needed

# AIC for training data
aic_train = calculate_aic(y_train, train_predict, num_params)

# AIC for testing data
aic_test = calculate_aic(y_test, test_predict, num_params)

# Step 14: Display the Evaluation Metrics
st.write("Evaluation Metrics for Training Data")
st.write(f"MAE: {mae_train}")
st.write(f"MSE: {mse_train}")
st.write(f"RMSE: {rmse_train}")
st.write(f"MAPE: {mape_train}")
st.write(f"R-squared: {r2_train}")
st.write(f"AIC: {aic_train}\n")

st.write("Evaluation Metrics for Testing Data")
st.write(f"MAE: {mae_test}")
st.write(f"MSE: {mse_test}")
st.write(f"RMSE: {rmse_test}")
st.write(f"MAPE: {mape_test}")
st.write(f"R-squared: {r2_test}")
st.write(f"AIC: {aic_test}")





# DENSE MODEL BUILDING

st.header("DENSE MODEL BUILDING")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
if choice == 'IBM':
    file_path = 'ibm.us.txt'
elif choice == 'Coca-Cola':
    file_path = 'ko.us.txt'
df = pd.read_csv(file_path)

# Parse dates and set date as the index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Focus on the 'Close' price for prediction
data = df['Close']

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

# Prepare the data for the Dense model
look_back = 10  # Number of previous days to use for predicting the next day

X, y = [], []
for i in range(len(scaled_data) - look_back):
    X.append(scaled_data[i:i + look_back, 0])
    y.append(scaled_data[i + look_back, 0])

X, y = np.array(X), np.array(y)

# Build the model
model = Sequential()
model.add(Dense(50, activation='relu', input_shape=(look_back,)))
model.add(Dense(25, activation='relu'))
model.add(Dense(1))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=20, batch_size=32)

# Make predictions for the next 10 days
last_10_days = scaled_data[-look_back:]
forecast = []

for _ in range(10):
    pred = model.predict(last_10_days.reshape(1, look_back))
    forecast.append(pred[0, 0])
    last_10_days = np.append(last_10_days[1:], pred[0, 0])

# Inverse transform the forecast back to original values
forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

# Plot the forecasted prices along with the actual data

st.subheader(f'{choice} Stock Price Forecast')

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(data[-100:], label='Actual Prices')  # Plotting the last 100 days for comparison
ax.plot(pd.date_range(start=data.index[-1], periods=11, freq='B')[1:], forecast, label='Forecasted Prices', color='red')
ax.set_xlabel('Date')
ax.set_ylabel('Close Price')
ax.set_title(f'{choice} Stock Price Forecast using Dense Model')
ax.legend()
st.pyplot(fig)

# Evaluation of the model
# Make predictions for the training data to evaluate the model
train_predictions = model.predict(X)

# Inverse transform the predictions back to original values
train_predictions_inverse = scaler.inverse_transform(train_predictions.reshape(-1, 1))
y_inverse = scaler.inverse_transform(y.reshape(-1, 1))

# Calculate evaluation metrics

st.subheader('Model Evaluation Metrics')

mae = mean_absolute_error(y_inverse, train_predictions_inverse)
mse = mean_squared_error(y_inverse, train_predictions_inverse)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_inverse - train_predictions_inverse) / y_inverse)) * 100
r2 = r2_score(y_inverse, train_predictions_inverse)

# Akaike Information Criterion (AIC) - for simplicity, let's use a basic approximation
n = len(y_inverse)
k = model.count_params()
aic = n * np.log(mse) + 2 * k

# Print the evaluation metrics
st.write(f"Mean Absolute Error (MAE): {mae}")
st.write(f"Mean Squared Error (MSE): {mse}")
st.write(f"Root Mean Squared Error (RMSE): {rmse}")
st.write(f"Mean Absolute Percentage Error (MAPE): {mape}")
st.write(f"R-squared (R²): {r2}")
st.write(f"Akaike Information Criterion (AIC): {aic}")
