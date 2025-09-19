import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from pandas.plotting import autocorrelation_plot
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import itertools
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

#load the data
df = pd.read_csv('electricityConsumptionAndProductioction.csv')
df.head()

#Define the energy sources of interest
energy_sources = ['Nuclear', 'Wind', 'Hydroelectric', 'Oil and Gas', 'Coal', 'Solar', 'Biomass']

#Create subplots for the selected energy sources
fig, axes = plt.subplots(nrows=len(energy_sources), ncols=1, figsize=(14, 7), sharex=True)

#Plot each energy source in its own subplot
for i, source in enumerate(energy_sources):
    axes[i].plot(df.index, df[source], label=source)
    axes[i].set_ylabel('energy sources')
    axes[i].set_title(f'{source} energy sources')
    axes[i].grid(True)

#Set common x-axis label and adjust layout
plt.xlabel('Date Time')
plt.tight_layout()
plt.show()

#Electricity generation by source
plt.figure(figsize=(14, 7))
plt.hist(df['Nuclear'], bins=50, alpha=0.5, label='Nuclear', color='blue')
plt.hist(df['Wind'], bins=50, alpha=0.5, label='Wind', color='green')
plt.hist(df['Hydroelectric'], bins=50, alpha=0.5, label='Hydroelectric', color='orange')
plt.xlabel('Electricity (MW)')
plt.ylabel('Frequency')
plt.title('Distribution of Electricity Generation by Source')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 7))
plt.hist(df['Oil and Gas'], bins=50, alpha=0.5,label='Oil and Gas', color='black')
plt.hist(df['Coal'], bins=50, alpha=0.5,label='Coal', color='red')
plt.hist(df['Solar'], bins=50, alpha=0.5,label='Solar', color='purple')
plt.hist(df['Biomass'], bins=50, alpha=0.5,label='Biomass', color='brown')
plt.xlabel('Electricity (MW)')
plt.ylabel('Frequency')
plt.title('Distribution of Electricity Generation by Source')
plt.legend()
plt.grid(True)
plt.show()

#Hourly electricity production by source
plt.figure(figsize=(14, 7))
plt.stackplot(df.index, [df[source] for source in energy_sources], labels=energy_sources)
plt.xlabel('Date Time')
plt.ylabel('Electricity Production (MW)')
plt.title('Hourly Electricity Production by Source (Stacked Area Plot)')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

#proportion of total electricity produced
total_production_by_source = df[energy_sources].sum()
plt.figure(figsize=(10, 10))
total_production_by_source.plot(kind='pie', autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0','#ffb3e6','#c4e17f'])
plt.ylabel('')
plt.title('Proportion of Total Electricity Production by Source')
plt.show()

#correlation
df['DateTime'] = pd.to_datetime(df['DateTime'])
plt.figure(figsize=(12, 6))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='viridis', linewidths=0.5)
plt.title('Correlation Matrix of Electricity Data')
plt.show()

#set time index
df = df.set_index("DateTime")

print(df.index.duplicated().sum())

expected_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='H')
missing = expected_range.difference(df.index)
print(f"Missing timestamps: {len(missing)}")

full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h")
df = df[~df.index.duplicated()]
df = df.reindex(full_idx).ffill()

freq = pd.infer_freq(df.index)
print(freq)


#plot of consumption
plt.figure(figsize=(10, 4))
plt.plot(df['Consumption'])
plt.title('Electricity Consumption Over Time')
plt.show()

#Create time features
df["Hour"] = df.index.hour
df["DayOfWeek"] = df.index.dayofweek
df["DayOfYear"] = df.index.dayofyear

#Train-test split (pre-2025 vs 2025+)
train = df[df.index.year < 2025]
test = df[df.index.year >= 2025]

#calculate returns
train_returns = np.log(train['Consumption']).diff().dropna()
test_returns = np.log(test['Consumption']).diff().dropna()

#ADF test - stationarity
from statsmodels.tsa.stattools import adfuller, acf, pacf
adf_df = adfuller(train_returns, autolag="AIC")
print(f"ADF Statistic: {adf_df[0]}, p-value={adf_df[1]}")
if adf_df[1] < 0.05:
    print("Reject null hypothesis, data is stationary")
else:
    print("Fail to reject null hypothesis, data is not stationary")

#Checking ARCH Effect
from statsmodels.stats.diagnostic import het_arch
arch_test = het_arch(train_returns)
print(f'ARCH test statistic: {arch_test[0]}, p-value: {arch_test[1]}')
if arch_test[1] < 0.05:
    print("Reject null hypothesis, returns exhibit ARCH effect.")
else:
    print("Fail to reject null hypothesis, No ARCH effect in returns.")

#Leverage effect check
from arch import arch_model
import statsmodels.api as sm
import scipy.stats as stats
garch = arch_model(train_returns, mean='Zero', vol='GARCH', p=1, q=1,
                    dist='Normal', rescale=True).fit(disp="off")
std_resid = garch.resid / garch.conditional_volatility
#Engle–Ng test
def engle_ng(train_res, lags=30):
    eps = pd.Series(train_res)
    S   = (eps < 0).astype(int)

    Z = pd.concat([(S.shift(i) * eps.shift(i)).fillna(0)
                   for i in range(1, lags+1)], axis=1)
    X = sm.add_constant(Z)
    y = (eps**2).iloc[lags:]
    X = X.iloc[lags:]

    aux = sm.OLS(y, X).fit()
    lm_stat = len(eps) * aux.rsquared
    p_val   = 1 - stats.chi2.cdf(lm_stat, df=lags)
    return lm_stat, p_val, aux

LM, p, _ = engle_ng(std_resid, lags=10)
print(f"Engle–Ng sign-bias LM = {LM}, p = {p}")
if p < 0.05:
    print("Leverage effect detected, switch to eGARCH")
else:
    print("No leverage effect, symmetric GARCH is adequate")

#EGARCH
rolling_best_ic = np.inf
rolling_best_egarch_order = None
rolling_ic_table = []    
for p, o, q in itertools.product(range(1, 4), range(1, 3), range(1, 4)):
    try:
        mod = arch_model(train_returns, vol='EGARCH', p=p, o=o, q=q, 
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
rolling_egarch_model = arch_model(train_returns, 
                                  vol="EGARCH", 
                                  p=rolling_best_egarch_order[0], 
                                  o=rolling_best_egarch_order[1], 
                                  q=rolling_best_egarch_order[2], 
                                  mean='Zero', 
                                  dist="skewt").fit(disp="off")
    
print("\nEGARCH Model Summary:")
print(rolling_egarch_model.summary())

#Generate volatility forecast
h = len(test_returns)
forecast_result = rolling_egarch_model.forecast(
        method="simulation", 
        simulations=1000, 
        horizon=h, 
        reindex=False
    )

forecast_sigma = np.sqrt(forecast_result.variance.iloc[0, :h]) 
forecast_sigma = pd.Series(forecast_sigma.values, index=test_returns.index)
scale_factor = 1 / forecast_sigma.mean()
forecast_sigma_scaled = forecast_sigma * scale_factor

#Evaluate volatility forecast
r2 = test_returns**2
sig2 = forecast_sigma_scaled**2 
qlike = np.mean(np.log(sig2) + r2 / sig2)
llh = (-0.5*np.log(2*np.pi) - np.log(forecast_sigma_scaled) - 0.5*(test_returns/forecast_sigma_scaled)**2).mean()
print(f"QLIKE: {qlike:.6f}")
print(f"Log-likelihood: {llh:.6f}")

#Prepare data for LSTM
in_sample_sigma = pd.Series(rolling_egarch_model.conditional_volatility, index=train_returns.index)
out_sample_sigma = forecast_sigma 
sigma_full = pd.concat([in_sample_sigma, out_sample_sigma]).reindex(df.index)
df_copy = df.copy()
df_copy['Sigma'] = sigma_full.interpolate().fillna(method='bfill').fillna(method='ffill')

train_updated = df_copy[df_copy.index.year < 2025].copy()
test_updated = df_copy[df_copy.index.year >= 2025].copy()
    
features = ['Consumption', 'Nuclear', 'Wind', 'Hydroelectric', 'Oil and Gas', 'Coal', 'Solar', 'Biomass', 'Sigma', 'Hour', 'DayOfWeek', 'DayOfYear']
    
#Scale features
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_updated[features])
test_scaled = scaler.transform(test_updated[features])

n_steps = 24
X_train, y_train = [], []
for i in range(len(train_scaled) - n_steps):
    X_train.append(train_scaled[i:i+n_steps])
    y_train.append(train_scaled[i+n_steps, 0])

X_test, y_test = [], []
for i in range(len(test_scaled) - n_steps):
    X_test.append(test_scaled[i:i+n_steps])
    y_test.append(test_scaled[i+n_steps, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test = np.array(X_test), np.array(y_test)

print(f"Training sequences: {X_train.shape}")
print(f"Test sequences: {X_test.shape}")

#Build and train LSTM model
lstm_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(n_steps, len(features))),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

#split training sequences into train/val without shuffling
split = int(len(X_train) * 0.8)
X_tr, y_tr = X_train[:split], y_train[:split]
X_val, y_val = X_train[split:], y_train[split:]

history = lstm_model.fit(X_tr, y_tr, epochs=50, batch_size=32, shuffle=False,
                        validation_data=(X_val, y_val), callbacks=[early_stop], verbose=1)

#Generate predictions
train_pred = lstm_model.predict(X_tr)
val_pred = lstm_model.predict(X_val)
test_pred = lstm_model.predict(X_test)

#Evaluate model
mse = mean_squared_error(y_test, test_pred)
mae = mean_absolute_error(y_test, test_pred)
rmse = np.sqrt(mse)
actual_direction = np.diff(y_test) > 0
pred_direction = np.diff(test_pred.flatten()) > 0
directional_accuracy = np.mean(actual_direction == pred_direction) * 100

print(f"Model Performance Metrics:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"Directional Accuracy: {directional_accuracy:.4f}%")

#Price Forecast Evaluation Plot (Test Set)
plt.figure(figsize=(15, 6))
plt.plot(test_updated.index[n_steps:], y_test, label='Actual Prices', color='blue', alpha=0.7)
plt.plot(test_updated.index[n_steps:], test_pred, label='Predicted Prices', color='red', linestyle='--')
plt.title('Test Set: Actual vs Predicted Electricity Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#Plot training & validation loss values
plt.figure(figsize=(14, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
