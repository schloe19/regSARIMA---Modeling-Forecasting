# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 16:55:25 2025

@author: Chloe
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime, timedelta
from scipy import stats
from scipy.io import savemat
from scipy.optimize import minimize

from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.graphics.tsaplots as tsa

import statsmodels.api as sm
import statsmodels.tsa.api as api
from statsmodels.formula.api import ols
from statsmodels.regression import linear_model
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import (
    adfuller, acf, pacf, ccf
)
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera

from patsy import dmatrices



class BirthsForecastPipelinenew:
    def __init__(self, train, test):
        self.train = train
        self.test = test
        self.models = []
        self.sarimax_result = None
        self.naive_forecast = None

    def fit_ols_model(self, formula, label):
        y_train, X_train = dmatrices(formula, self.train, return_type='dataframe')
        y_test, X_test = dmatrices(formula, self.test, return_type='dataframe')
        model = sm.OLS(y_train, X_train).fit()
        
        print(f"\n=== OLS Summary: {label} ===")
        print(model.summary())
    
        pred_train = pd.Series(model.fittedvalues, index=y_train.index)
        pred_test = pd.Series(model.predict(X_test), index=y_test.index)
        resid_train = pd.Series(y_train.squeeze() - model.fittedvalues, index=y_train.index)
    
        self.models.append({
            'label': label,
            'model': model,
            'y_train': y_train.squeeze(),
            'y_test': y_test.squeeze(),
            'pred_train': pred_train,
            'pred_test': pred_test,
            'resid_train': resid_train,
            'X_train': X_train,
            'X_test': X_test
        })
    
    def plot_all_ols_predictions(self):
        if len(self.models) < 1:
            print("No OLS models fitted yet.")
            return
    
        # Plot 1: Train Predictions
        plt.figure(figsize=(10, 6))
        labels_used = set()
        for model in self.models:
            plt.plot(model['pred_train'].index, model['pred_train'], label=f"{model['label']} (Train)", linestyle='--')
        for model in self.models:
            if 'Real Births (Train)' not in labels_used:
                plt.plot(model['y_train'].index, model['y_train'], label='Real Births (Train)', color='black', alpha=0.4)
                labels_used.add('Real Births (Train)')
        plt.xlabel('Time')
        plt.ylabel('Births')
        plt.title('Train Predictions from All OLS Models')
        plt.legend(loc='lower left')
        plt.show()
    
        # Plot 2: Test Predictions
        plt.figure(figsize=(10, 6))
        labels_used = set()
        for model in self.models:
            plt.plot(model['pred_test'].index, model['pred_test'], label=f"{model['label']} (Test)", linestyle='--')
        for model in self.models:
            if 'Real Births (Test)' not in labels_used:
                plt.plot(model['y_test'].index, model['y_test'], label='Real Births (Test)', color='black', alpha=0.4)
                labels_used.add('Real Births (Test)')
        plt.xlabel('Time')
        plt.ylabel('Births')
        plt.title('Test Predictions from All OLS Models')
        plt.legend(loc='lower left')
        plt.show()
    
        # Plot 3: Combined Train + Test
        plt.figure(figsize=(12, 6))
        for model in self.models:
            plt.plot(model['y_train'].index, model['y_train'], label='Real Births (Train)', color='black', alpha=0.4)
            plt.plot(model['y_test'].index, model['y_test'], label='Real Births (Test)', color='purple', alpha=0.4)
            break
        for model in self.models:
            plt.plot(model['pred_train'].index, model['pred_train'], label=f"{model['label']} (Train)", linestyle='--')
            plt.plot(model['pred_test'].index, model['pred_test'], label=f"{model['label']} (Test)", linestyle='-')
        plt.xlabel('Time')
        plt.ylabel('Births')
        plt.title('Predictions from All OLS Models (Train & Test)')
        plt.legend(loc='lower left')
        plt.show()
    
        # Plot 4: Residuals Over Time
        plt.figure(figsize=(10, 6))
        for model in self.models:
            plt.plot(model['resid_train'].index, model['resid_train'], label=f"Residuals {model['label']}", linewidth=0.7)
        plt.xlabel('Time')
        plt.ylabel('Residuals')
        plt.title('Residuals Over Time (Train)')
        plt.legend(loc='lower left')
        plt.show()



    def plot_residual_diagnostics(self):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
        # Scatter: Residuals vs Fitted
        for model in self.models:
            axes[0, 0].scatter(model['pred_train'], model['resid_train'], s=10, alpha=0.6, label=model['label'])
        axes[0, 0].axhline(0, color='black', linestyle='--')
        axes[0, 0].set_title("Residuals vs Fitted")
        axes[0, 0].legend()
    
        # ACF
        for model in self.models:
            tsa.plot_acf(model['resid_train'], lags=60, ax=axes[1, 0], alpha=0.05, title=None)
        axes[1, 0].set_title('ACF of Residuals')
    
        # Q-Q Plot
        for model in self.models:
            stats.probplot(model['resid_train'], dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title("Q-Q Plot")
    
        # PACF
        for model in self.models:
            sm.graphics.tsa.plot_pacf(model['resid_train'], lags=60, ax=axes[1, 1], alpha=0.05, title=None)
        axes[1, 1].set_title('PACF of Residuals')
    
        plt.tight_layout()
        plt.show()
        
    def plot_seasonal_residual_acf_pacf(self, seasonal_lag=52, lags=60):
        if len(self.models) < 1:
            print("No OLS models fitted.")
            return
    
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for model in self.models:
            resid_diff = model['resid_train'].diff(periods=seasonal_lag).dropna()
            tsa.plot_acf(resid_diff, lags=lags, alpha=0.05, ax=axes[0], title=None)
            tsa.plot_pacf(resid_diff, lags=lags, alpha=0.05, ax=axes[1], title=None)
            axes[0].plot([], [], label=model['label'])  # Add label manually to legend
            axes[1].plot([], [], label=model['label'])  # Same here
    
        axes[0].set_title(f'ACF of Residuals (Seasonal Diff={seasonal_lag})')
        axes[1].set_title(f'PACF of Residuals (Seasonal Diff={seasonal_lag})')
        axes[0].legend()
        axes[1].legend()
        plt.tight_layout()
        plt.show()

    def plot_seasonal_differenced_residuals_over_time(self, seasonal_lag=52):
        if len(self.models) < 1:
            print("No OLS models fitted.")
            return
    
        plt.figure(figsize=(10, 6))
        for model in self.models:
            resid_diff = model['resid_train'].diff(periods=seasonal_lag)
            # Align with the time index after seasonal differencing
            index = model['resid_train'].index[seasonal_lag:]
            plt.plot(index, resid_diff[seasonal_lag:], label=f'{model["label"]}', linewidth=0.8, alpha=0.8)
        
        plt.xlabel('Time')
        plt.ylabel('Seasonally Differenced Residuals')
        plt.title(f'Seasonally Differenced Residuals Over Time (Lag={seasonal_lag})')
        plt.legend(loc='lower left')
        plt.tight_layout()
        plt.show()
    def fit_custom_sarimax_model(self, label, endog_col='count', exog_pred=None,
                             order=(11, 0, 0), seasonal_order=(0, 0, 0, 0),
                             estimate_params=None,
                             enforce_stationarity=False,
                             enforce_invertibility=False,
                             custom_endog=None,
                             custom_exog=None):

        if not hasattr(self, 'sarimax_models'):
            self.sarimax_models = []
    
        endog_series = custom_endog if custom_endog is not None else self.train[endog_col]
        exog_input = custom_exog if custom_exog is not None else (
            exog_pred.to_frame() if isinstance(exog_pred, pd.Series) else exog_pred
        )
    
        model = SARIMAX(endog_series,
                        exog=exog_input,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=enforce_stationarity,
                        enforce_invertibility=enforce_invertibility)
    
        param_names = model.param_names
    
        if estimate_params is None:
            results = model.fit()
        else:
            fixed_params = {name: 0 for name in param_names if name not in estimate_params}
            results = model.fit_constrained(constraints=fixed_params)
    
        print(f"\n=== SARIMAX Summary: {label} ===")
        print(results.summary())
    
        self.sarimax_models.append({
            'label': label,
            'results': results,
            'residuals': results.resid,
            'exog_train': exog_input,
            'endog_train': endog_series
        })
    
        self.sarimax_result = self.sarimax_models[-1]  
        
    def plot_sarimax_diagnostics(self, seasonal_period=52):
        if not hasattr(self, 'sarimax_models') or len(self.sarimax_models) == 0:
            print("No SARIMAX models have been fitted.")
            return
    
        if len(self.sarimax_models) < 2:
            print("There are less than two SARIMAX models to compare.")
            return
    
        num_models = len(self.sarimax_models)
        
        # === Residual Plot ===
        plt.figure(figsize=(12, 4))
        for model in self.sarimax_models:
            res = model['residuals'][seasonal_period + 6:]  # Get residuals (excluding initial data)
            label = model['label']
            plt.plot(res, label=f"{label} Residuals", alpha=0.7)
        
        plt.axhline(0, color='black', linestyle='--')
        plt.title("Residuals Comparison")
        plt.legend()
        plt.tight_layout()
        plt.show()
    
        
        # === ACF/PACF Plots (side-by-side per model) ===
        fig, axes = plt.subplots(num_models, 2, figsize=(14, 4 * num_models))
        for i, model in enumerate(self.sarimax_models):
            res = model['residuals'][seasonal_period + 6:]  # Get residuals (excluding initial data)
            tsa.plot_acf(res, lags=60, ax=axes[i, 0], alpha=0.05)
            axes[i, 0].set_title(f"ACF - {model['label']}")
            sm.graphics.tsa.plot_pacf(res, lags=60, ax=axes[i, 1], alpha=0.05)
            axes[i, 1].set_title(f"PACF - {model['label']}")
        plt.tight_layout()
        plt.show()
    
        # === Q-Q Plots ===
        fig, axes = plt.subplots(1, num_models, figsize=(5 * num_models, 5))
        if num_models == 1:
            axes = [axes]
        for i, model in enumerate(self.sarimax_models):
            res = model['residuals'][seasonal_period + 6:]  # Get residuals (excluding initial data)
            stats.probplot(res, dist="norm", plot=axes[i])
            axes[i].set_title(f"Q-Q Plot - {model['label']}")
        plt.tight_layout()
        plt.show()
    
        # === Statistical Tests ===
        print("=== Ljung-Box Test (lag=11) ===")
        for model in self.sarimax_models:
            res = model['residuals'][seasonal_period + 6:]  # Get residuals (excluding initial data)
            lb = acorr_ljungbox(res, lags=[11], return_df=True)
            print(f"{model['label']}:\n{lb}\n")
    
        print("=== Jarque-Bera Test ===")
        for model in self.sarimax_models:
            res = model['residuals'][seasonal_period + 8:]  # Get residuals (excluding initial data)
            jb_stat = jarque_bera(res)
            print(f"{model['label']} - JB Stat: {jb_stat[0]:.3f}, p-value: {jb_stat[1]:.4f}")

        
    def forecast_sarimax(self, label, exog_future):
        if not hasattr(self, 'sarimax_models'):
            raise ValueError("SARIMAX models not yet fitted.")
        
        # Normalize input to support single or multiple labels
        if isinstance(label, str):
            label = [label]
        if isinstance(exog_future, dict):
            exog_future_dict = exog_future
        else:
            raise ValueError("When using multiple models, exog_future must be a dict of {label: exog_df}")
    
        forecast_results = {}
    
        for lbl in label:
            model = next((m for m in self.sarimax_models if m['label'] == lbl), None)
            if model is None:
                raise ValueError(f"No SARIMAX model with label '{lbl}'.")
            
            results = model['results']
            exog = exog_future_dict.get(lbl)
            if exog is None:
                raise ValueError(f"No exogenous data provided for label '{lbl}'.")
    
            steps = len(exog)
            forecast = results.get_forecast(steps=steps, exog=exog, index=exog.index)
            forecast_mean = forecast.predicted_mean
            forecast_ci = forecast.conf_int()
            forecast_se = forecast.se_mean
    
            model['forecast_mean'] = forecast_mean
            model['forecast_ci'] = forecast_ci
            model['forecast_se'] = forecast_se
            forecast_results[lbl] = (forecast_mean, forecast_ci, forecast_se)
            
            # === Print Forecast ===
            print(f"\n=== Forecast for {lbl} ===")
            print(pd.concat([forecast_mean.rename('Forecast'), forecast_ci], axis=1))
    
        # === Plot All Forecasts Together ===
        plt.figure(figsize=(12, 6))
        plt.plot(self.train.index, self.train['count'], label='Training Data', color='blue', linewidth=0.5)
        plt.plot(self.test.index, self.test['count'], label='Test Data', color='green', linewidth=0.5)
    
        colors = ['red', 'purple', 'orange', 'brown']
        for i, (lbl, (mean, ci, se)) in enumerate(forecast_results.items()):
            color = colors[i % len(colors)]
            plt.plot(mean.index, mean, label=f'{lbl} Forecast', color=color, linewidth=1)
            plt.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], color=color, alpha=0.2)
    
        plt.title(f'Forecast Comparison: {", ".join(label)}')
        plt.xlabel('Time')
        plt.ylabel('Births')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
        # === Zoomed-In Forecast Only Plot ===
        plt.figure(figsize=(12, 6))
        plt.plot(self.test.index, self.test['count'], label='Test Data', color='green', linewidth=0.5)
        for i, (lbl, (mean, ci, se)) in enumerate(forecast_results.items()):
            color = colors[i % len(colors)]
            plt.plot(mean.index, mean, label=f'{lbl} Forecast', color=color, linewidth=1)
            plt.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], color=color, alpha=0.2)
    
        plt.title(f'Forecast (Zoomed-In): {", ".join(label)}')
        plt.xlabel('Date')
        plt.ylabel('Number of Births')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
        return forecast_results
    
    def compare_with_naive(self, y_test_col='count', seasonal_lag=52, exog_futures=None):
        if not hasattr(self, 'sarimax_models'):
            raise ValueError("SARIMAX models not fitted.")
    
        y_test = self.test[y_test_col]
        y_train = self.train[y_test_col]
    
        plt.figure(figsize=(12, 6))
        plt.plot(y_test.index, y_test, label='Test Data', color='black')
    
        for model in self.sarimax_models:
            label = model['label']
            if 'forecast_mean' not in model:
                exog_future = exog_futures[label] if exog_futures and label in exog_futures else None
                if exog_future is None:
                    raise ValueError(f"Missing exog_future for model {label}")
                self.forecast_sarimax(label, exog_future)
    
            forecast = model['forecast_mean']
            forecast = forecast.loc[y_test.index.intersection(forecast.index)]  # Align to y_test
            y_eval = y_test.loc[forecast.index]  # Align y_test to forecast index
    
            plt.plot(forecast.index, forecast, label=f'{label} Forecast')
    
            mse = mean_squared_error(y_eval, forecast)
            mae = mean_absolute_error(y_eval, forecast)
            print(f"{label} - MSE: {mse:.2f}, MAE: {mae:.2f}")
    
        naive = y_train.iloc[-seasonal_lag:].values[:len(y_test)]
        naive_forecast = pd.Series(naive, index=y_test.index[:len(naive)])
        plt.plot(naive_forecast.index, naive_forecast, label='Naive Forecast', linestyle='--', color='gray')
    
        mse_naive = mean_squared_error(y_test[:len(naive_forecast)], naive_forecast)
        mae_naive = mean_absolute_error(y_test[:len(naive_forecast)], naive_forecast)
        print(f"Naive - MSE: {mse_naive:.2f}, MAE: {mae_naive:.2f}")
    
        plt.title("SARIMAX Models vs Naive Forecast")
        plt.xlabel("Time")
        plt.ylabel("Births")
        plt.legend()
        plt.show()
    
        
