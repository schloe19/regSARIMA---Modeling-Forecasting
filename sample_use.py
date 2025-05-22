# -*- coding: utf-8 -*-
"""
Created on Thu May 22 14:13:33 2025

@author: Chloe
"""

from BirthsForecastPipeline import BirthsForecastPipeline

# Ensure the train and test DataFrames contain the following:
# - 'count' column (endogenous variable)
# - One or more columns used as exogenous regressors, e.g. 'age_25_29_lag40'
# endogenous and exogenous variables should be of the same size, with the same frequency, 
# preferably indexed by date (datetime)

# =======================================================================================
# Data Structure Requirements for BirthsForecastPipeline
# =======================================================================================
#
# The pipeline expects two pandas DataFrames: `train` and `test`.
#
# These DataFrames must meet the following requirements:
# INDEX: Must be a DateTimeIndex.
#
# CONSISTENCY:
#    - Both `train` and `test` must have the same structure and column names.
#    - No missing values in the modeling columns.
#
# Example Columns:
# ┌──────────────┬───────┬────────────────────┬────────────────────┐
# │   index      │ count │ age_25_29_lag40    │ age_30_34_lag40    │
# ├──────────────┼───────┼────────────────────┼────────────────────┤
# │ 2023-01-01   │ 250   │ 40000              │ 31000              │
# │ 2023-01-08   │ 260   │ 40500              │ 31200              │
# │     ...      │ ...   │ ...                │ ...                │
# └──────────────┴───────┴────────────────────┴────────────────────┘
#
# =======================================================================================


# =============================================================================
# # Instantiate the pipeline
# pipeline = BirthsForecastPipeline(train=train_df, test=test_df)
# 
# # Fit and visualize OLS models
# pipeline.fit_ols_model('count ~ age_25_29_lag40', label='Model 1: 25-29 Lag')
# pipeline.fit_ols_model('count ~ age_20_24_lag40 + age_30_34_lag40', label='Model 2: 20-24 & 30-34 Lag')
# 
# pipeline.plot_all_ols_predictions()
# pipeline.plot_residual_diagnostics()
# pipeline.plot_seasonal_residual_acf_pacf(seasonal_lag=52)
# pipeline.plot_seasonal_differenced_residuals_over_time(seasonal_lag=52)
#
# Define exogenous predictors for SARIMAX
# exog_train1 = train_df[['age_25_29_lag40']]
# exog_train2 = train_df[['age_20_24_lag40', 'age_30_34_lag40']]
#
# # Fit SARIMAX models
# pipeline.fit_custom_sarimax_model(
#     label='SARIMAX 25-29 Lag',
#     custom_endog=train_df['count'],
#     custom_exog=exog_train1,
#     order=(8, 0, 0),
#     seasonal_order=(0, 1, 1, 52),
#     estimate_params=['age_25_29_lag40', 'ar.L2', 'ar.L8', 'ma.S.L52', 'sigma2']
# )
#
# pipeline.fit_custom_sarimax_model(
#     label='SARIMAX 20-24 & 30-34 Lag',
#     custom_endog=train_df['count'],
#     custom_exog=exog_train2,
#     order=(8, 0, 0),
#     seasonal_order=(0, 1, 1, 52),
#     estimate_params=['age_20_24_lag40', 'age_30_34_lag40', 'ar.L2', 'ar.L8', 'ma.S.L52', 'sigma2']
# )
#
# # Exogenous predictors for forecasting
# exog_futures = {
#     'SARIMAX 25-29 Lag': test_df[['age_25_29_lag40']],
#     'SARIMAX 20-24 & 30-34 Lag': test_df[['age_20_24_lag40', 'age_30_34_lag40']]
# }
#
# # Forecast and plot
# pipeline.forecast_sarimax(label=['SARIMAX 25-29 Lag', 'SARIMAX 20-24 & 30-34 Lag'], exog_future=exog_futures)
# pipeline.plot_sarimax_diagnostics()
# pipeline.compare_with_naive(y_test_col='count', seasonal_lag=52, exog_futures=exog_futures)
#
# =============================================================================
