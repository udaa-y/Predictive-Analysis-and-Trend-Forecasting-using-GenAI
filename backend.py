from flask import Flask, request, jsonify
import yfinance as yf
from flask_cors import CORS
import pandas as pd
from prophet import Prophet  # For time series forecasting
import numpy as np
import os
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from twilio.rest import Client
import plotly.express as px  # Import plotly express for visualization
import json  # Import json for JSON encoding the plot
import plotly

app = Flask(__name__)
CORS(app)
#########
TWILIO_ACCOUNT_SID = 'AC53a04391f80e7ebffff9d16259100b49'
TWILIO_AUTH_TOKEN = '96510b4331deb1595b598b340b24737e'
TWILIO_PHONE_NUMBER = '+15746336538'
RECIPIENT_PHONE_NUMBER = '+918078851881'
#########
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

#######

EXCEL_FILE = r"C:\Users\Student\Downloads\project_7sem.xlsx"

# Create the file if it doesn't exist, with an additional signup_time column
if not os.path.exists(EXCEL_FILE):
    df = pd.DataFrame(columns=['email', 'password', 'signup_time'])
    df.to_excel(EXCEL_FILE, index=False)


@app.route('/authenticate', methods=['POST'])
def authenticate():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    is_signup = data.get('isSignup')

    # Read current users data from Excel
    df = pd.read_excel(EXCEL_FILE)

    if is_signup:
        # Signup process
        if email in df['email'].values:
            return jsonify({'success': False, 'message': 'User already exists. Please log in.'})

        # Get the current time for signup
        signup_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Add new user to Excel with signup time
        new_user = pd.DataFrame([[email, password, signup_time]], columns=['email', 'password', 'signup_time'])
        df = pd.concat([df, new_user], ignore_index=True)
        df.to_excel(EXCEL_FILE, index=False)
        return jsonify({'success': True, 'message': 'Signup successful! Please log in.'})
    else:
        # Login process
        user = df[(df['email'] == email) & (df['password'] == password)]

        if not user.empty:
            return jsonify({'success': True, 'message': 'Login successful!'})
        else:
            return jsonify({'success': False, 'message': 'Invalid email or password.'})



######

def send_sms_alert(message):
    try:
        sms = client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=RECIPIENT_PHONE_NUMBER
        )
        print(f"SMS sent: {sms.sid}")
    except Exception as e:
        print(f"Failed to send SMS: {str(e)}")

# Function to check for drastic price changes
def check_price_change(history, ticker, threshold=0.00001):
    # Check the daily price change (percentage)
    current_price = history['Close'].iloc[-1]
    previous_price = history['Close'].iloc[-2]

    price_change_percentage = ((current_price - previous_price) / previous_price) * 100

    if abs(price_change_percentage) > threshold:
        # Drastic change detected, send SMS
        if price_change_percentage<0:
            message = f"Alert! {ticker} stock price changed by {price_change_percentage:.2f}%."
            send_sms_alert(message)
        else:
            message = f"Alert! {ticker} stock price changed by +{price_change_percentage:.2f}%."
            send_sms_alert(message)


def suggest_improvement(metric_name):
    suggestions = {
        "Total Revenue": "Consider diversifying product offerings, improving marketing efforts, or targeting new markets to increase sales.",
        "Net Income": "Focus on cutting unnecessary expenses, optimizing operational efficiency, or renegotiating contracts with suppliers.",
        "Cash Flow": "Review payment ter"
                     "ms with customers and suppliers, reduce inventory levels, or seek additional financing.",
        "Stock Price": "Enhance investor relations, share strategic growth plans, or issue positive financial guidance to boost investor confidence."
    }
    return suggestions.get(metric_name, "Consider consulting with a financial advisor for tailored strategies.")
def forecast_and_evaluate(df, periods, metric_name):
    """
    Forecast future values for a given financial metric and evaluate performance on past data.
    df: A dataframe with columns ['ds', 'y'] where 'ds' is date and 'y' is the value to forecast.
    periods: Number of future periods to forecast.
    metric_name: Name of the metric (for display in logging/debug).
    """
    # Split the data into training and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)

    # Train the model with hyperparameter tuning
    model = Prophet(
        growth='linear',
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.1,  # Adjust to control changepoint flexibility
        seasonality_prior_scale=10.0,  # Adjust seasonality effect
    )
    model.fit(train_df)

    # Make future dataframe and forecast
    future = model.make_future_dataframe(periods=len(test_df), freq='Q')  # Forecast for the test period
    forecast = model.predict(future)

    # Evaluate the model by comparing the forecast with actual test data
    test_forecast = forecast.iloc[-len(test_df):]['yhat']  # Forecasted values for the test period
    actual_values = test_df['y'].values  # Actual values from the test set

    # Calculate evaluation metrics
    mae = mean_absolute_error(actual_values, test_forecast)
    mse = mean_squared_error(actual_values, test_forecast)
    mape = np.mean(np.abs((actual_values - test_forecast) / actual_values)) * 100

    evaluation_results = {
        'mae': mae,
        'mse': mse,
        'mape': mape,
        'test_forecast': test_forecast.tolist(),  # Convert to list for JSON serialization
        'actual_values': actual_values.tolist()     # Convert to list for JSON serialization
    }

    # Forecast future periods beyond the test set
    future_forecast = model.make_future_dataframe(periods=periods, freq='Q')
    full_forecast = model.predict(future_forecast)

    return full_forecast, evaluation_results
# Updated Function: Forecasting using Prophet (New Addition)

@app.route('/dashboard/revenue', methods=['GET'])
def revenue_dashboard():
    # Get the ticker from query parameters
    ticker = request.args.get('ticker', '').upper()

    if not ticker:
        return jsonify({'error': 'Please provide a valid stock ticker.'})

    # Fetch stock data
    stock_data = yf.Ticker(ticker)

    # Get revenue data
    try:
        # Retrieve revenue data
        revenue_data = stock_data.financials.loc['Total Revenue']
        revenue_data.index = pd.to_datetime(revenue_data.index)
        revenue_df = pd.DataFrame({'Date': revenue_data.index, 'Revenue': revenue_data.values})

        # Plot using plotly
        fig = px.line(revenue_df, x='Date', y='Revenue', title=f"{ticker} Revenue Over Time")
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return jsonify({'plot': graphJSON})
    except Exception as e:
        return jsonify({'error': f'Error retrieving data for {ticker}: {str(e)}'})

#####
@app.route('/dashboard/revenue_with_forecast', methods=['GET'])
@app.route('/dashboard/revenue_with_forecast', methods=['GET'])
def revenue_dashboard_with_forecast():
    # Get the ticker from query parameters
    ticker = request.args.get('ticker', '').upper()

    if not ticker:
        return jsonify({'error': 'Please provide a valid stock ticker.'})

    # Fetch stock data
    stock_data = yf.Ticker(ticker)

    try:
        # Retrieve historical revenue data
        revenue_data = stock_data.financials.loc['Total Revenue']
        revenue_data.index = pd.to_datetime(revenue_data.index)
        revenue_df = pd.DataFrame({'Date': revenue_data.index, 'Revenue': revenue_data.values})

        # Prepare data for forecasting
        forecast_df = revenue_df.rename(columns={'Date': 'ds', 'Revenue': 'y'})

        # Forecast for the next quarter
        forecast, evaluation = forecast_and_evaluate(forecast_df, periods=1, metric_name="Total Revenue")
        forecasted_date = forecast['ds'].iloc[0]
        forecasted_revenue = forecast['yhat'].iloc[0]

        # Append forecasted revenue as a new data point
        revenue_df = revenue_df.append({'Date': forecasted_date, 'Revenue': forecasted_revenue}, ignore_index=True)

        # If there's a previous forecast from the "query" endpoint, we can plot that as well.
        if 'forecasted_revenue' in request.args:
            # Get forecasted revenue from the query if it exists
            additional_forecasted_value = float(request.args.get('forecasted_revenue'))
            revenue_df = revenue_df.append({'Date': forecasted_date, 'Revenue': additional_forecasted_value}, ignore_index=True)

        # Plot the combined chart
        fig = px.line(revenue_df, x='Date', y='Revenue', title=f"{ticker} Revenue with Forecast for Next Quarter")

        # Add the forecasted point with a distinct style
        fig.add_scatter(
            x=[revenue_df['Date'].iloc[-2], forecasted_date],
            y=[revenue_df['Revenue'].iloc[-2], forecasted_revenue],
            mode="lines+markers+text",
            name="Forecasted Revenue",
            line=dict(color="red", dash="dot"),
            marker=dict(color="red", size=10),
            text=["", f"Forecasted: ${forecasted_revenue:,.2f}"],
            textposition="top right"
        )

        # Add the additional forecasted revenue point if it exists
        if 'forecasted_revenue' in request.args:
            fig.add_scatter(
                x=[forecasted_date],
                y=[additional_forecasted_value],
                mode="markers+text",
                name="Additional Forecasted Revenue",
                marker=dict(color="blue", size=10),
                text=[f"Forecasted: ${additional_forecasted_value:,.2f}"],
                textposition="top right"
            )

        # Convert to JSON for frontend display
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return jsonify({'plot': graphJSON})

    except Exception as e:
        return jsonify({'error': f'Error retrieving or forecasting data for {ticker}: {str(e)}'})

#####

@app.route('/query', methods=['POST'])
def query():
    user_query = request.json.get('query', '').lower()
    ticker = request.json.get('ticker', '').upper()

    if not ticker:
        return jsonify({'response': 'Please provide a valid stock ticker.'})

    # Fetch stock data
    stock_data = yf.Ticker(ticker)

    # Retrieve the financial data
    try:
        financials = stock_data.financials
        balance_sheet = stock_data.balance_sheet
        cashflow = stock_data.cashflow
        stock_info = stock_data.info
        history = stock_data.history(period="5y")
        check_price_change(history, ticker, threshold=0.00001)
    except Exception as e:
        return jsonify({'response': f'Error retrieving data for {ticker}: {str(e)}'})

    response = ""

    # Process queries
    try:
        if "forecast next quarter with trend" in user_query:
            # Extract last three quarters' revenue
            revenue_data = financials.loc['Total Revenue'].dropna()
            revenue_data = revenue_data.iloc[:3]
            revenue_data.index = pd.to_datetime(revenue_data.index)
            revenue_df = pd.DataFrame({'ds': revenue_data.index, 'y': revenue_data.values})

            # Forecast for next quarter
            forecast_revenue, _ = forecast_and_evaluate(revenue_df, periods=1, metric_name="Total Revenue")
            next_quarter_revenue = forecast_revenue.iloc[-1]['yhat']

            # Combine last three quarters and forecasted value
            revenue_df.loc['Forecast'] = [forecast_revenue.iloc[-1]['ds'], next_quarter_revenue]

            # Plot with Plotly
            fig = px.line(revenue_df, x='ds', y='y', title=f"{ticker} Revenue Trend and Forecast for Next Quarter")
            graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

            response = jsonify({
                'response': f"Forecasted revenue for {ticker} next quarter is ${next_quarter_revenue:,.2f}.",
                'plot': graphJSON
            })
            return response

            #except Exception as e:
            #   response = f"An error occurred: {str(e)}"

    #return jsonify({'response': response})
        elif "forecast revenue" in user_query or "revenue next quarter" in user_query:
            try:
                revenue_data = financials.loc['Total Revenue'].dropna()
                revenue_data.index = pd.to_datetime(revenue_data.index)  # Convert index to datetime
                revenue_df = pd.DataFrame({'ds': revenue_data.index, 'y': revenue_data.values})
                forecast_revenue, evaluation = forecast_and_evaluate(revenue_df, periods=1, metric_name="Total Revenue")
                next_quarter_revenue = forecast_revenue.iloc[-1]['yhat']

                if next_quarter_revenue < 1000000000:
                    suggestion = suggest_improvement("Total Revenue")
                    response = f"Forecasted revenue for {ticker} next quarter is (${abs(next_quarter_revenue):,.2f}). {suggestion}"
                else:
                    response = f"Forecasted revenue for {ticker} next quarter is ${next_quarter_revenue:,.2f}."

                # Updated evaluation metrics in a readable format
                response += (
                    f"\nModel Evaluation:\n"
                    f" - Mean Absolute Error (MAE): ${evaluation['mae']:,.2f}\n"
                    f" - Mean Squared Error (MSE): ${evaluation['mse']:,.2f}\n"
                    f" - Test Forecast Values: {evaluation['test_forecast']}\n"
                    f" - Actual Values: {evaluation['actual_values']}"
                )
            except Exception as e:
                response = f"Sorry, I couldn't retrieve or forecast the revenue data: {str(e)}."

            # Stock Price Forecasting for Next Year
        elif "forecast stock price" in user_query or "stock price next year" in user_query:
            try:
                stock_prices = history['Close'].resample('M').mean().dropna()  # Monthly stock prices
                stock_df = pd.DataFrame({'ds': stock_prices.index, 'y': stock_prices.values})
                forecast_stock_price, evaluation = forecast_and_evaluate(stock_df, periods=12,
                                                                         metric_name="Stock Price")  # 12 months forecast
                next_year_stock_price = forecast_stock_price.iloc[-1]['yhat']

                if next_year_stock_price < 0:
                    suggestion = suggest_improvement("Stock Price")
                    response = f"Forecasted stock price for {ticker} next year is negative (${abs(next_year_stock_price):,.2f}). {suggestion}"
                else:
                    response = f"Forecasted stock price for {ticker} next year is ${next_year_stock_price:,.2f}."

                # Updated evaluation metrics in a readable format
                response += (
                    f"\nModel Evaluation:\n"
                    f" - Mean Absolute Error (MAE): ${evaluation['mae']:,.2f}\n"
                    f" - Mean Squared Error (MSE): ${evaluation['mse']:,.2f}\n"
                    f" - Mean Absolute Percentage Error (MAPE): {evaluation['mape']:.2f}%\n"
                    f" - Test Forecast Values: {evaluation['test_forecast']}\n"
                    f" - Actual Values: {evaluation['actual_values']}"
                )
            except Exception as e:
                response = f"Sorry, I couldn't retrieve or forecast the stock price data: {str(e)}."

            # Net Income Forecasting
        elif "forecast net income" in user_query or "net income next year" in user_query:
            try:
                income_data = financials.loc['Net Income'].dropna()
                income_data.index = pd.to_datetime(income_data.index)
                income_df = pd.DataFrame({'ds': income_data.index, 'y': income_data.values})
                forecast_income, evaluation = forecast_and_evaluate(income_df, periods=4,
                                                                    metric_name="Net Income")  # 4 periods (quarters) forecast
                next_year_income = forecast_income.iloc[-1]['yhat']

                if next_year_income < 0:
                    suggestion = suggest_improvement("Net Income")
                    response = f"Forecasted net income for {ticker} next year is negative (${abs(next_year_income):,.2f}). {suggestion}"
                else:
                    response = f"Forecasted net income for {ticker} next year is ${next_year_income:,.2f}."

                # Updated evaluation metrics in a readable format
                response += (
                    f"\nModel Evaluation:\n"
                    f" - Mean Absolute Error (MAE): ${evaluation['mae']:,.2f}\n"
                    f" - Mean Squared Error (MSE): ${evaluation['mse']:,.2f}\n"
                    f" - Mean Absolute Percentage Error (MAPE): {evaluation['mape']:.2f}%\n"
                    f" - Test Forecast Values: {evaluation['test_forecast']}\n"
                    f" - Actual Values: {evaluation['actual_values']}"
                )
            except Exception as e:
                response = f"Sorry, I couldn't retrieve or forecast the net income data: {str(e)}."


            # Cash Flow Forecasting
        elif "forecast cash flow" in user_query or "cash flow next year" in user_query:
            try:
                cashflow_data = cashflow.loc['Net Cash Flow'].dropna()
                cashflow_data.index = pd.to_datetime(cashflow_data.index)
                flow_df = pd.DataFrame({'ds': cashflow_data.index, 'y': cashflow_data.values})
                forecast_cashflow, evaluation = forecast_and_evaluate(flow_df, periods=4, metric_name="Cash Flow")
                next_year_cashflow = forecast_cashflow.iloc[-1]['yhat']

                if next_year_cashflow < 0:
                    suggestion = suggest_improvement("Cash Flow")
                    response = f"Forecasted cash flow for {ticker} next year is negative (${abs(next_year_cashflow):,.2f}). {suggestion}"
                else:
                    response = f"Forecasted cash flow for {ticker} next year is ${next_year_cashflow:,.2f}."

                # Updated evaluation metrics in a readable format
                response += (
                    f"\nModel Evaluation:\n"
                    f" - Mean Absolute Error (MAE): ${evaluation['mae']:,.2f}\n"
                    f" - Mean Squared Error (MSE): ${evaluation['mse']:,.2f}\n"
                    f" - Mean Absolute Percentage Error (MAPE): {evaluation['mape']:.2f}%\n"
                    f" - Test Forecast Values: {evaluation['test_forecast']}\n"
                    f" - Actual Values: {evaluation['actual_values']}"
                )
            except Exception as e:
                response = f"Sorry, I couldn't retrieve or forecast the cash flow data: {str(e)}."
        # Unchanged Block: Revenue Comparison
        elif "compare revenue" in user_query:
            try:
                last_quarter_revenue = financials.loc['Total Revenue'].iloc[0]
                prev_quarter_revenue = financials.loc['Total Revenue'].iloc[1]
                revenue_change = ((last_quarter_revenue - prev_quarter_revenue) / prev_quarter_revenue) * 100
                response = f"{ticker}'s revenue last quarter was ${last_quarter_revenue:,.2f}, " \
                           f"which is a {revenue_change:.2f}% change compared to the previous quarter."
            except:
                response = "Sorry, I couldn't retrieve the revenue comparison information."

        # Unchanged Block: Stock Price Comparison Over a Year
        elif "compare stock price last year" in user_query or "yearly stock price" in user_query:
            try:
                current_price = stock_info['currentPrice']
                last_year_price = history['Close'].iloc[-252]  # Assuming 252 trading days in a year
                price_change = ((current_price - last_year_price) / last_year_price) * 100
                response = f"{ticker}'s stock price is currently ${current_price:,.2f}, " \
                           f"which is a {price_change:.2f}% change from ${last_year_price:,.2f} a year ago."
            except:
                response = "Sorry, I couldn't retrieve the stock price comparison."

        # Unchanged Block: Net Income Comparison (YoY)
        elif "compare net income" in user_query or "year over year" in user_query:
            try:
                current_income = financials.loc['Net Income'].iloc[0]
                previous_year_income = financials.loc['Net Income'].iloc[4]
                income_change = ((current_income - previous_year_income) / previous_year_income) * 100
                response = f"{ticker}'s net income this year is ${current_income:,.2f}, " \
                           f"a {income_change:.2f}% change compared to ${previous_year_income:,.2f} last year."
            except:
                response = "Sorry, I couldn't retrieve the net income comparison."
        ###
        elif "compare revenue" in user_query:
            try:
                last_quarter_revenue = financials.loc['Total Revenue'].iloc[0]
                prev_quarter_revenue = financials.loc['Total Revenue'].iloc[1]
                revenue_change = ((last_quarter_revenue - prev_quarter_revenue) / prev_quarter_revenue) * 100
                response = f"{ticker}'s revenue last quarter was ${last_quarter_revenue:,.2f}, " \
                           f"which is a {revenue_change:.2f}% change compared to the previous quarter."
            except:
                response = "Sorry, I couldn't retrieve the revenue comparison information."

            # Stock price comparison over a year
        elif "compare stock price last year" in user_query or "yearly stock price" in user_query:
            try:
                current_price = stock_info['currentPrice']
                last_year_price = history['Close'].iloc[-252]  # Assuming 252 trading days in a year
                price_change = ((current_price - last_year_price) / last_year_price) * 100
                response = f"{ticker}'s stock price is currently ${current_price:,.2f}, " \
                           f"which is a {price_change:.2f}% change from ${last_year_price:,.2f} a year ago."
            except:
                response = "Sorry, I couldn't retrieve the stock price comparison."

            # Net income comparison (YoY)
        elif "compare net income" in user_query or "year over year" in user_query:
            try:
                current_income = financials.loc['Net Income'].iloc[0]
                previous_year_income = financials.loc['Net Income'].iloc[4]
                income_change = ((current_income - previous_year_income) / previous_year_income) * 100
                response = f"{ticker}'s net income this year is ${current_income:,.2f}, " \
                           f"a {income_change:.2f}% change compared to ${previous_year_income:,.2f} last year."
            except:
                response = "Sorry, I couldn't retrieve the net income comparison."

            # Total assets comparison (YoY)
        elif "compare total assets" in user_query:
            try:
                current_assets = balance_sheet.loc['Total Assets'].iloc[0]
                previous_year_assets = balance_sheet.loc['Total Assets'].iloc[4]
                asset_change = ((current_assets - previous_year_assets) / previous_year_assets) * 100
                response = f"{ticker}'s total assets this year are ${current_assets:,.2f}, " \
                           f"which is a {asset_change:.2f}% increase from ${previous_year_assets:,.2f} last year."
            except:
                response = "Sorry, I couldn't retrieve the total assets comparison."

            # Cash flow comparison (YoY)
        elif "compare cash flow" in user_query:
            try:
                current_cash_flow = cashflow.loc['Total Cash From Operating Activities'].iloc[0]
                previous_year_cash_flow = cashflow.loc['Total Cash From Operating Activities'].iloc[4]
                cash_flow_change = ((current_cash_flow - previous_year_cash_flow) / previous_year_cash_flow) * 100
                response = f"{ticker}'s operating cash flow this year is ${current_cash_flow:,.2f}, " \
                           f"which is a {cash_flow_change:.2f}% change compared to ${previous_year_cash_flow:,.2f} last year."
            except:
                response = "Sorry, I couldn't retrieve the cash flow comparison."
        ###
        elif "revenue last quarter" in user_query:
            last_quarter_revenue = financials.loc['Total Revenue'].iloc[0]
            response = f"The revenue for {ticker} in the last quarter was ${last_quarter_revenue:,.2f}."

        elif "total assets" in user_query:
            total_assets = balance_sheet.loc['Total Assets'].iloc[0]
            response = f"{ticker}'s total assets are ${total_assets:,.2f}."

        elif "cash flow" in user_query:
            cash_flow = cashflow.loc['Total Cash From Operating Activities'].iloc[0]
            response = f"{ticker}'s total cash flow from operating activities is ${cash_flow:,.2f}."

        elif "current stock price" in user_query:
            stock_price = stock_info['currentPrice']
            response = f"The current stock price of {ticker} is ${stock_price:,.2f}."

        elif "market capitalization" in user_query:
            market_cap = stock_info['marketCap']
            response = f"{ticker}'s market capitalization is ${market_cap:,.2f}."

        elif "net income last quarter" in user_query:
            net_income = financials.loc['Net Income'].iloc[0]
            response = f"{ticker}'s net income in the last quarter was ${net_income:,.2f}."

        elif "earnings per share" in user_query:
            eps = stock_info['trailingEps']
            response = f"{ticker}'s earnings per share (EPS) is ${eps:,.2f}."

        elif "gross profit margin" in user_query:
            gross_profit_margin = stock_info['grossMargins']
            response = f"{ticker}'s gross profit margin is {gross_profit_margin:.2%}."

        elif "total liabilities" in user_query:
            total_liabilities = balance_sheet.loc['Total Liabilities Net Minority Interest'].iloc[0]
            response = f"{ticker}'s total liabilities are ${total_liabilities:,.2f}."

        elif "debt-to-equity ratio" in user_query:
            debt_to_equity = stock_info['debtToEquity']
            response = f"{ticker}'s debt-to-equity ratio is {debt_to_equity:.2f}."

        elif "dividend yield" in user_query:
            dividend_yield = stock_info['dividendYield']
            response = f"{ticker}'s dividend yield is {dividend_yield:.2%}."

        elif "price-to-earnings ratio" in user_query:
            pe_ratio = stock_info['trailingPE']
            response = f"{ticker}'s price-to-earnings (P/E) ratio is {pe_ratio:.2f}."

        elif "price-to-book ratio" in user_query:
            pb_ratio = stock_info['priceToBook']
            response = f"{ticker}'s price-to-book (P/B) ratio is {pb_ratio:.2f}."

        elif "current ratio" in user_query:
            current_ratio = stock_info['currentRatio']
            response = f"{ticker}'s current ratio is {current_ratio:.2f}."

        elif "return on equity" in user_query:
            roe = stock_info['returnOnEquity']
            response = f"{ticker}'s return on equity (ROE) is {roe:.2%}."

        elif "return on assets" in user_query:
            roa = stock_info['returnOnAssets']
            response = f"{ticker}'s return on assets (ROA) is {roa:.2%}."

        elif "free cash flow" in user_query:
            free_cash_flow = cashflow.loc['Total Cash From Operating Activities'].iloc[0]
            response = f"{ticker}'s free cash flow is ${free_cash_flow:,.2f}."

        elif "revenue growth year-over-year" in user_query:
            revenue_growth = stock_info['revenueGrowth']
            response = f"{ticker}'s revenue growth year-over-year (YoY) is {revenue_growth:.2%}."

        elif "profit margin" in user_query:
            profit_margin = stock_info['profitMargins']
            response = f"{ticker}'s profit margin is {profit_margin:.2%}."

        elif "total debt" in user_query:
            total_debt = balance_sheet.loc['Total Debt'].iloc[0]
            response = f"{ticker}'s total debt is ${total_debt:,.2f}."

        elif "book value per share" in user_query:
            book_value_per_share = stock_info['bookValue']
            response = f"{ticker}'s book value per share is ${book_value_per_share:,.2f}."

        elif "ebitda" in user_query:
            ebitda = stock_info['ebitda']
            response = f"{ticker}'s EBITDA is ${ebitda:,.2f}."

        elif "interest coverage ratio" in user_query:
            interest_coverage = stock_info['ebitda'] / stock_info['totalDebt']
            response = f"{ticker}'s interest coverage ratio is {interest_coverage:.2f}."

        # Handle other unchanged queries...
        else:
            response = "Sorry, I couldn't understand your query. Please ask about stock price, revenue, assets, or other financial metrics."

    except KeyError:
        response = "Sorry, some financial data is missing for this company."

    from transformers import pipeline
    ########
    import nltk
    from transformers import pipeline

    # Ensure nltk has the necessary data for tokenization
    nltk.download('punkt')
    nltk.download('punkt_tab')

    # Load the text generation model
    generator = pipeline('text-generation', model='gpt2')

    # Function to tokenize sentences and generate text for each sentence
    def generate_text_with_sentence_tokenization(user_query_output):
        # Tokenize the input into sentences
        sentences = nltk.sent_tokenize(user_query_output)

        generated_sentences = []

        # For each sentence, generate text
        for sentence in sentences:
            generated_result = generator(sentence, max_length=150, num_return_sequences=1)
            generated_sentences.append(generated_result[0]['generated_text'])

        # Join all generated sentences
        final_output = ' '.join(generated_sentences)

        return final_output

    # Example usage with a user query output
    #user_query_output = "Once upon a time, there was a knight. He went on an adventure."
    response = generate_text_with_sentence_tokenization(response)

    # Print the final output
    #print(final_output)

    #########
    # Load the text generation model
    #generator = pipeline('text-generation', model='gpt2')

    # Transformer function that generates text based on user query output
    #def generate_text(user_query_output):
        # Generate text based on the user query output
        #result = generator(user_query_output, max_length=150, num_return_sequences=1)
        #result = generator(response, max_length=150, num_return_sequences=1)
        # Display the final generated text
        #return result[0]['generated_text']

    # Example usage with a user query output
    #user_query_output = response
    #final_output = generate_text(response)
    #response = generate_text(response)
    # Print the final output
    #print(final_output)
    #########
    #response = final_output
    return jsonify({'response': response})


if __name__ == '__main__':
    app.run(debug=True)

