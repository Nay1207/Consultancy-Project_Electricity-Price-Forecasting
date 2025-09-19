# Hybrid EGARCH-LSTM for Electricity Price Forecasting

## 📖 Overview

This repository contains the code and resources for a consultancy project with **Prosumer Energy** to forecast hourly electricity prices. 
The project implements hybrid model that combines the **Exponential GARCH (EGARCH)** model with a **Long Short-Term Memory (LSTM)** neural network.

The repository includes two main Python scripts:
1.  `model_evaluation.py`: An analytical script for model development, training on a historical dataset, and out-of-sample performance evaluation.
2.  `real_time_forecaster.py`: An operational script that simulates a real-time application, using a rolling window to retrain the model and generate future forecasts.

