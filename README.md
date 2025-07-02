# Predictive-Analysis-and-Trend-Forecasting-using-GenAI
Description:
This project is a Flask-based financial forecasting API that leverages machine learning, natural language processing, and financial data APIs to predict company metrics such as revenue, stock price, net income, and cash flow. It supports authentication, alert systems, and AI-generated textual insights.

Features:

User authentication (signup/login) using Excel sheet as database
Revenue forecasting using Facebook Prophet
Stock price prediction for the next year
Net income and cash flow forecasting
SMS alert system using Twilio for drastic stock price changes
Plotly-based financial charts returned as JSON
Query-based financial insights using HuggingFace GPT-2 (text generation)
Forecast model evaluation using MAE, MSE, and MAPE

Technologies Stack:

Flask & Flask-CORS
YFinance for real-time financial data
Facebook Prophet for time series forecasting
Scikit-learn for model evaluation
Twilio for SMS alerts
Plotly for chart visualization
HuggingFace Transformers (GPT-2) for generating textual responses
NLTK for sentence tokenization
Pandas & OpenPyXL for Excel data handling

This project is a Flask-based backend application that leverages real-time financial data, machine learning, and natural language processing to deliver intelligent financial insights. It allows users to sign up and log in securely, with credentials stored in an Excel sheet alongside signup timestamps. Once authenticated, users can query financial data for any public company by its stock ticker symbol. The application fetches data such as revenue, stock price, net income, and cash flow using the Yahoo Finance API through the yfinance library. For predictive analytics, it utilizes Facebook’s Prophet model to forecast key metrics like revenue for the next quarter, stock price trends for the next year, and other financial indicators, while also evaluating the model’s performance using MAE, MSE, and MAPE.

Additionally, the system detects drastic changes in stock price and sends real-time SMS alerts using the Twilio API. It also includes a dashboard endpoint that provides visualizations of revenue trends and forecasts using Plotly. Beyond numbers, the application processes natural language queries and responds with relevant financial insights. For more nuanced and coherent answers, it uses HuggingFace's GPT-2 model to expand or refine responses, effectively turning structured outputs into fluent, human-like explanations. Overall, this project functions as an intelligent financial assistant capable of analysis, forecasting, alerting, and interactive Q&A.
