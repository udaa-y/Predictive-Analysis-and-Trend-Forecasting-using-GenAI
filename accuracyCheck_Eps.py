import yfinance as yf
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error

# Define the ticker symbol
ticker = "AAPL"  # Replace with the desired company's ticker
stock = yf.Ticker(ticker)

# Get the quarterly financial data and inspect columns
financials = stock.quarterly_financials.T  # Transpose to get quarters as rows
print("Available columns in financials:", financials.columns)  # Check available columns

# Check if "Basic EPS" and "Total Revenue" are in the columns
if 'Basic EPS' not in financials.columns or 'Total Revenue' not in financials.columns:
    raise KeyError("Basic EPS or additional feature data is not available for this company in Yahoo Finance.")

# Prepare the main data (Basic EPS)
data = financials[['Basic EPS']].reset_index()
data.columns = ['ds', 'y']  # Rename columns for Prophet (ds = date, y = value)

# Add additional regressor (e.g., Total Revenue)
data['Total Revenue'] = financials['Total Revenue'].values
data.dropna(inplace=True)  # Drop rows with NaN values
data['ds'] = pd.to_datetime(data['ds'])

# Split data into training (all but last quarter) and test (last quarter)
train_data = data.iloc[:-1]
test_data = data.iloc[-1:]  # This will be the last quarter

# Initialize the Prophet model with additional seasonality and custom parameters
model = Prophet(
    changepoint_prior_scale=0.1,      # Controls trend flexibility
    seasonality_prior_scale=8.0      # Controls seasonality flexibility
)

# Add quarterly seasonality (since EPS can show quarterly patterns)
model.add_seasonality(name='quarterly', period=1, fourier_order=5)
model.add_seasonality(name='yearly', period=4, fourier_order=10)

# Add the extra regressor
model.add_regressor('Total Revenue')

# Fit the model with additional regressor
model.fit(train_data)

# Prepare future data and include the additional regressor
future = model.make_future_dataframe(periods=1, freq='Q')
future['Total Revenue'] = data['Total Revenue'].iloc[-1]  # Use last known revenue as a proxy

# Predict the last quarter's EPS
forecast = model.predict(future)

# Extract the forecasted EPS for the last quarter
predicted_last_quarter_eps = forecast.iloc[-1]['yhat']
actual_last_quarter_eps = test_data['y'].values[0]

# Calculate accuracy score
accuracy = 100 * (1 - mean_absolute_percentage_error([actual_last_quarter_eps], [predicted_last_quarter_eps]))

print(f"Predicted Last Quarter Earnings Per Share: {predicted_last_quarter_eps}")
print(f"Actual Last Quarter Earnings Per Share: {actual_last_quarter_eps}")
print(f"Accuracy: {accuracy:.2f}%")