"""
Created on 6 Feb 2021
Explore the source data provided by WHOOP, and do
some exploratory data analysis before applying ML models
"""
import numpy as np
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    pd.set_option('display.max_columns', 30)
    pd.set_option('display.width', 600)

    whoop_data_dir = '/Users/philip_p/data/whoop_request_20210119'

    recovery_df = pd.read_csv(os.path.join(whoop_data_dir, 'recoveries.csv'))

    print(f"Overall number of columns is {len(recovery_df.columns)}")
    print(f'Shape of data is {recovery_df.shape}')

    # let's only focus on the sleep data
    sleep_columns = [x for x in recovery_df.columns if 'sleep' in x.lower()]
    sleep_data = recovery_df[['Date', 'Day of Week'] + sleep_columns].copy(True)

    print(sleep_data.dtypes)
    # only look a the numeric data
    numeric_sleep_data = sleep_data.select_dtypes(exclude=object)

    print(numeric_sleep_data[numeric_sleep_data.isnull().any(axis=1)])
    # looks like there is one day that has quite a few null values, let's investigate
    # further
    null_value_rows = numeric_sleep_data[numeric_sleep_data.isnull().any(axis=1)].index

    cleaned_sleep_df = numeric_sleep_data.loc[
        ~numeric_sleep_data.index.isin(null_value_rows)].copy(True)

    print(cleaned_sleep_df.describe().T)
    # the variables don't look of the same scale,
    # so we will use the standard scaler for the input variables

    # response variable is 'Sleep Score'
    # all other variables are input
    response_variable = 'Sleep Score'
    X = cleaned_sleep_df[[x for x in cleaned_sleep_df.columns if x != response_variable]]
    y = cleaned_sleep_df[response_variable]

    scaler = StandardScaler()
    scaler.fit(X)
    sleep_scaled = scaler.transform(X)

    # now the data is scaled we can define and fit the linear regression model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1)

    lin_reg_model = LinearRegression()
    lin_reg_model.fit(X_train, y_train)

    # we will also predict the results
    y_pred = lin_reg_model.predict(X_test)

    print(f'Mean squared error: {mean_squared_error(y_test, y_pred):.4f}')
    # The coefficient of determination: 1 is perfect prediction
    print(f'Coefficient of determination: {r2_score(y_test, y_pred):.4f}')

    # the mean squared error is low, and the R2 score is very high, so
    # the model fits the data well

    # Linear machine learning algorithms fit a model where the prediction
    # is a weighted sum of the input values. We can interpret the coefficients
    # of each feature crudely as a type of feature importance score

    # get the importance of the features
    importance = lin_reg_model.coef_

    for num, feature_imp in enumerate(importance):
        print(f"Feature: {X.columns[num]}, \t Importance {feature_imp:.2f}")

    fig, ax = plt.subplots()
    fig = plt.bar([x for x in range(len(importance))], importance)

    ax.set_xticks(range(len(list(X.columns))))
    ax.set_xticklabels(list(X.columns), rotation=45, fontsize=8)

    importance_df = pd.DataFrame(data={f'importance': importance},
                                 index=X.columns)
    importance_df.sort_values('importance', ascending=False, inplace=True)

    print(importance_df)

    # Therefore, we will drop the features which do not have much importance,
    # thus reducing the dimensionality of the problem we are solving, to investigate
    # how the sleep score is affected.
    # Remove anything with an absolute value of less than 10
    keep_features = importance_df.loc[np.abs(importance_df['importance']) > 10].index

    # reduced model
    X_new = X[keep_features]

    new_linear_reg_model = LinearRegression()
    new_linear_reg_model.fit(X_train, y_train)

    # we will also predict the results
    y_pred_new = new_linear_reg_model.predict(X_test)

    print(f'Mean squared error: {mean_squared_error(y_test, y_pred_new):.4f}')
    # The coefficient of determination: 1 is perfect prediction
    print(f'Coefficient of determination: {r2_score(y_test, y_pred_new):.4f}')

    # so, the coefficient of determination is still good, whilst using
    # many fewer dimensions
