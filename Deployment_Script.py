import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import itertools
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from arch import arch_model
import statsmodels.api as sm
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

def electricity_price_forecast(
    data_path='electricityConsumptionAndProductioction.csv',
    forecast_date='2025-03-19 23:00:00',
    training_window_days=365,
    n_steps=24,
    n_forecast_hours=24,
    random_seed=42,
    lstm_units=[64, 32],
    dropout_rates=[0.2, 0.2],
    epochs=50,
    batch_size=32,
    patience=5,
    plot_results=True
):
    
    #Set random seeds for reproducibility
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    
    #Load and preprocess data
    df = (pd.read_csv(data_path, parse_dates=['DateTime'])
        .set_index('DateTime')
        .sort_index())
    
    #Create full hourly index to handle missing values
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df[~df.index.duplicated()]
    df = df.reindex(full_idx).ffill()
    
    #Create time features
    df["Hour"] = df.index.hour
    df["DayOfWeek"] = df.index.dayofweek
    df["DayOfYear"] = df.index.dayofyear
    
    #Convert forecast date to timestamp
    current_date = pd.Timestamp(forecast_date)
    training_window_size = pd.Timedelta(days=training_window_days)
    
    #Ensure we have enough data
    if current_date < df.index.min() + training_window_size:
        print(f"Warning: Not enough historical data for the specified training window. Adjusting forecast date.")
        current_date = df.index.min() + training_window_size
    
    #Calculate training window
    train_start = current_date - training_window_size
    training_df = df.loc[train_start:current_date].copy()
    
    print(f"Training period: {train_start} to {current_date}")
    print(f"Training data points: {len(training_df)}")
    
    #Calculate returns for EGARCH
    returns = np.log(training_df['Consumption']).diff().dropna()
    
    #EGARCH model selection
    print("\nSelecting EGARCH model...")
    rolling_best_ic = np.inf
    rolling_best_egarch_order = None
    rolling_ic_table = []
    
    for p, o, q in itertools.product(range(1, 4), range(1, 3), range(1, 4)):
        try:
            mod = arch_model(returns, vol='EGARCH', p=p, o=o, q=q, 
                            mean='Zero', dist='skewt').fit(disp='off')
            aic = mod.aic
            bic = mod.bic
            min_ic = min(aic, bic)
            rolling_ic_table.append({'p': p, 'o': o, 'q': q, 'aic': aic, 'bic': bic, 'min_ic': min_ic})
            if min_ic < rolling_best_ic:
                rolling_best_ic = min_ic
                rolling_best_egarch_order = (p, o, q)
        except:
            continue
    
    print(f"Selected EGARCH order: {rolling_best_egarch_order}")
    
    #Fit final EGARCH model
    rolling_egarch_model = arch_model(
        returns, 
        vol="EGARCH", 
        p=rolling_best_egarch_order[0], 
        o=rolling_best_egarch_order[1], 
        q=rolling_best_egarch_order[2], 
        mean='Zero', 
        dist="skewt"
    ).fit(disp="off")
    
    forecast_dates = pd.date_range(
        start=current_date + pd.Timedelta(hours=1),
        periods=n_forecast_hours, 
        freq='H'
    )

    #Generate volatility forecast
    rolling_forecast_result = rolling_egarch_model.forecast(
        method="simulation", 
        simulations=1000, 
        horizon=n_forecast_hours, 
        reindex=False
    )
    sigma_values = np.sqrt(rolling_forecast_result.variance.values[-1])
    forecast_sigma = pd.Series(sigma_values, index = forecast_dates)

    #Prepare data for LSTM
    in_sample_sigma = pd.Series(rolling_egarch_model.conditional_volatility, index = returns.index)
    sigma_train = in_sample_sigma.reindex(training_df.index).shift(1)
    sigma_train.iloc[0] = sigma_train.dropna().iloc[0]


    features = ['Consumption', 'Nuclear', 'Wind', 'Hydroelectric', 'Oil and Gas', 'Coal', 'Solar', 'Biomass', 'Sigma', 'Hour', 'DayOfWeek', 'DayOfYear']

    rolling_df_features = training_df.copy()
    rolling_df_features['Sigma'] = sigma_train.ffill().bfill()

    #Scale features
    rolling_scaler = MinMaxScaler()
    scaled_data = rolling_scaler.fit_transform(rolling_df_features[features])
    
    #Create sequences for LSTM
    X, y = [], []
    for i in range(len(scaled_data) - n_steps):
        X.append(scaled_data[i:i+n_steps])
        y.append(scaled_data[i+n_steps, 0])
    X, y = np.array(X), np.array(y)

    #Validation for EarlyStopping
    split = int(len(X)*0.9)
    X_tr, y_tr, X_val, y_val = X[:split], y[:split], X[split:], y[split:]
    
    #Build LSTM model
    rolling_lstm_model = Sequential()
    for i, (units, dr) in enumerate(zip(lstm_units, dropout_rates)):
        rolling_lstm_model.add(
            LSTM(units,
                return_sequences=(i < len(lstm_units)-1),
                input_shape=(n_steps, len(features)) if i == 0 else None)
    )
    rolling_lstm_model.add(Dropout(dr))
    rolling_lstm_model.add(Dense(1))
    rolling_lstm_model.compile(optimizer='adam', loss='mse')

    early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    rolling_lstm_model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                       epochs=epochs, batch_size=batch_size, shuffle=False,
                       callbacks=[early_stop], verbose=1)
    
    #Generate forecasts
    print("\nGenerating forecasts...")
    idx = {name: i for i, name in enumerate(features)}
    last_seq = scaled_data[-n_steps:].copy().reshape(1, n_steps, len(features))
    forecast_prices = []

    def scale_val(val, fidx):
        mn, mx = rolling_scaler.data_min_[fidx], rolling_scaler.data_max_[fidx]
        return (val - mn) / (mx - mn + 1e-12)

    def inv_target(y_s):
        mn, mx = rolling_scaler.data_min_[idx['Consumption']], rolling_scaler.data_max_[idx['Consumption']]
        return y_s * (mx - mn) + mn

    for i, ts in enumerate(forecast_dates):
    #predict next hour (scaled)
        y_hat_s = rolling_lstm_model.predict(last_seq, verbose=0)[0, 0]
        forecast_prices.append(float(inv_target(y_hat_s)))

    #next sigma (unscaled -> scaled using Sigma’s scaler)
        sig_s = scale_val(forecast_sigma.iloc[i], idx['Sigma'])

    #build next row: start from last row, then overwrite changing fields
        new_row = last_seq[:, -1, :].copy()
        new_row[0, idx['Consumption']] = y_hat_s
        new_row[0, idx['Sigma']]       = sig_s
        new_row[0, idx['Hour']]        = scale_val(ts.hour,      idx['Hour'])
        new_row[0, idx['DayOfWeek']]   = scale_val(ts.dayofweek, idx['DayOfWeek'])
        new_row[0, idx['DayOfYear']]   = scale_val(ts.dayofyear, idx['DayOfYear'])

        if not (6 <= ts.hour <= 19):
            new_row[0, idx['Solar']] = scale_val(0, idx['Solar'])
    
        last_seq = np.concatenate([last_seq[:, 1:, :], new_row.reshape(1,1,-1)], axis=1)

    forecast_df = pd.DataFrame({'Predicted_Consumption': forecast_prices}, index=forecast_dates)
    
    print("\nForecast Results:")
    print(forecast_df)
    
    #Plot results if requested
    if plot_results:
        plt.figure(figsize=(15, 6))
        hist = df.loc[current_date - pd.Timedelta(days=7):current_date, 'Consumption']
        plt.plot(hist.index, hist, label='Historical Consumption', color='blue', alpha=0.6)
        plt.plot(forecast_df.index, forecast_df['Predicted_Consumption'], 'r--o', label='Forecast')
        plt.title(f'{n_forecast_hours}-Hour Consumption Forecast (generated at {current_date})')
        plt.xlabel('Date'); plt.ylabel('Consumption'); plt.legend(); plt.grid(True); plt.tight_layout()
        plt.show()

    return forecast_df, rolling_egarch_model, rolling_lstm_model, rolling_scaler

forecast_df, egarch_model, lstm_model, scaler = electricity_price_forecast()

forecast_df.to_csv('Egarch_LSTM.csv')