import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Regression
from sklearn.linear_model import LinearRegression, Ridge

# Tree-based
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Time series
from prophet import Prophet
from prophet.plot import plot_plotly
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA

# Deep learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

st.set_page_config(layout="wide")
st.title("📦 Unified Demand Forecasting System")

uploaded_file = st.file_uploader("Upload your training_data.csv", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.dropna(subset=['units_sold'], inplace=True)

    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df = df.sort_values('date')
    ts_df = df[['date', 'units_sold']].groupby('date').sum().reset_index()
    ts_df.columns = ['ds', 'y']

    st.sidebar.header("Select Forecasting Model")
    model_choice = st.sidebar.radio("Choose a model:", [
        'Linear Regression', 'Ridge Regression',
        'Random Forest', 'XGBoost', 'LightGBM',
        'ARIMA', 'SARIMA', 'Prophet',
        'LSTM', 'GRU'])

    def show_recommendations(mae=None, r2=None, trend_delta=None):
        st.subheader("💡 Business Recommendations")
        if trend_delta is not None:
            if trend_delta > 0:
                st.success("Demand is trending upward. Consider increasing inventory and marketing.")
            else:
                st.info("Demand is flat or declining. Optimize logistics, reduce inventory risk.")
        elif r2 is not None:
            if r2 > 0.85:
                st.success("Excellent accuracy! You can rely on this model for operational planning.")
            elif r2 > 0.6:
                st.warning("Moderate accuracy. Review feature importance and consider model tuning.")
            else:
                st.error("Low accuracy. Consider additional data, feature engineering, or another model type.")

    if model_choice in ['Linear Regression', 'Ridge Regression', 'Random Forest', 'XGBoost', 'LightGBM']:
        df = df.join(pd.get_dummies(df['store_id'], prefix='store')).drop('store_id', axis=1)
        df = df.join(pd.get_dummies(df['sku_id'], prefix='sku')).drop('sku_id', axis=1)
        X = df.drop(['units_sold', 'date', 'day', 'month', 'year'], axis=1)
        y = df['units_sold']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model_choice == 'Linear Regression':
            model = LinearRegression()
        elif model_choice == 'Ridge Regression':
            model = Ridge()
        elif model_choice == 'Random Forest':
            model = RandomForestRegressor(n_jobs=-1)
        elif model_choice == 'XGBoost':
            model = XGBRegressor()
        elif model_choice == 'LightGBM':
            model = LGBMRegressor()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.metric("MAE", f"{mae:.2f}")
        st.metric("R² Score", f"{r2:.2f}")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.6)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        st.pyplot(fig)

        show_recommendations(mae=mae, r2=r2)

    elif model_choice in ['ARIMA', 'SARIMA']:
        ts_df.set_index('ds', inplace=True)
        if model_choice == 'ARIMA':
            model = ARIMA(ts_df['y'], order=(5,1,0)).fit()
        elif model_choice == 'SARIMA':
            model = SARIMAX(ts_df['y'], order=(1,1,1), seasonal_order=(1,1,1,12)).fit()
        forecast = model.forecast(steps=30)
        st.line_chart(forecast)

        trend_delta = forecast.mean() - ts_df['y'].iloc[-30:].mean()
        show_recommendations(trend_delta=trend_delta)

    elif model_choice == 'Prophet':
        m = Prophet()
        m.fit(ts_df)
        future = m.make_future_dataframe(periods=30)
        forecast = m.predict(future)
        st.plotly_chart(plot_plotly(m, forecast))

        trend_delta = forecast['yhat'].tail(30).mean() - ts_df['y'].iloc[-30:].mean()
        show_recommendations(trend_delta=trend_delta)

    elif model_choice in ['LSTM', 'GRU']:
        ts = ts_df.set_index('ds')['y'].values.reshape(-1, 1)
        generator = TimeseriesGenerator(ts, ts, length=10, batch_size=1)
        model = Sequential()
        if model_choice == 'LSTM':
            model.add(LSTM(50, activation='relu', input_shape=(10, 1)))
        else:
            model.add(GRU(50, activation='relu', input_shape=(10, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(generator, epochs=5, verbose=0)

        pred_input = ts[-10:].reshape((1, 10, 1))
        preds = []
        for _ in range(30):
            pred = model.predict(pred_input, verbose=0)[0]
            preds.append(pred[0])
            pred_input = np.append(pred_input[:,1:,:], [[pred]], axis=1)

        st.line_chart(preds)

        trend_delta = np.mean(preds) - np.mean(ts[-30:])
        show_recommendations(trend_delta=trend_delta)
else:
    st.info("Upload the dataset to begin.")
