import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.model_selection import RandomizedSearchCV, KFold, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import smogn
import pandas as pd
import xgboost as xgb
import scipy
from Python_Functions import test_train_split_func

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dropout, Flatten, Dense #These are the "layer" types that we will use.
import os
import sys
from tensorflow.keras import Sequential
from tensorflow.nn import relu
# To evaluate the performance of the Model
from sklearn.metrics import r2_score

# Instantiate StandardScaler
scaler = StandardScaler()

def save_outputs(model_stats, model_name, current_model, X_train, X_test, y_train,y_test, y_train_predicted, y_test_predicted, X_std, y, y_predicted) :
    # save the model statistics
    model_stats['Training_R2'][model_name] = round(current_model.score(X_train, y_train), 2)
    model_stats['Testing_R2'][model_name] = round(current_model.score(X_test, y_test), 2)
    model_stats['R2'][model_name] = round(current_model.score(X_std, y), 2)
    model_stats['Training_RMSE'][model_name] = (np.sqrt(mean_squared_error(y_train, y_train_predicted)))
    model_stats['Testing_RMSE'][model_name] = (np.sqrt(mean_squared_error(y_test, y_test_predicted)))
    model_stats['RMSE'][model_name] = (np.sqrt(mean_squared_error(y, y_predicted)))
    #CRMSE does not work with XGBoost Early Stopping
    #model_stats['Training_CRMSE'][model_name] = np.sqrt(np.mean(((y_train_predicted-np.mean(y_train_predicted))-(y_train-np.mean(y_train)))**2))
    #model_stats['Testing_CRMSE'][model_name] = np.sqrt(np.mean(((y_test_predicted-np.mean(y_test_predicted))-(y_test-np.mean(y_test)))**2))
    #model_stats['CRMSE'][model_name] = np.sqrt(np.mean(((y_predicted - np.mean(y_predicted)) - (y - np.mean(y))) ** 2))
    model_stats['Training_MBE'][model_name] = np.mean(y_train_predicted - y_train)
    model_stats['Testing_MBE'][model_name] = np.mean(y_test_predicted - y_test)
    model_stats['MBE'][model_name] = np.mean(y_predicted - y)

    return model_stats

def save_percentile_outputs(percentile_model_stats, model_name, current_model, X_train, X_test, y_train,y_test, y_train_predicted, y_test_predicted, X_std, y, y_predicted) :
    # save the model statistics
    for l_pct, u_pct in percentile_model_stats[model_name].index:
        lower_limit = np.percentile(y, l_pct)
        upper_limit = np.percentile(y, u_pct)
        # Filter training data to only include points where y_train is above the 75th percentile of the combined y
        X_train_filtered = X_train[(y_train > lower_limit) & (y_train < upper_limit)]
        y_train_filtered = y_train[(y_train > lower_limit) & (y_train < upper_limit)]
        y_train_predicted_filtered = y_train_predicted[(y_train > lower_limit) & (y_train < upper_limit)]

        # Filter test data to only include points where y_test is above the 75th percentile of the combined y
        X_test_filtered = X_test[(y_test > lower_limit)& (y_test < upper_limit)]
        y_test_filtered = y_test[(y_test > lower_limit) & (y_test < upper_limit)]
        y_test_predicted_filtered = y_test_predicted[(y_test > lower_limit) & (y_test < upper_limit)]

        # Filter test data to only include points where y_test is above the 75th percentile of the combined y
        X_filtered = X_std[(y > lower_limit)& (y < upper_limit)]
        y_filtered = y[(y > lower_limit) & (y < upper_limit)]
        y_predicted_filtered = y_predicted[(y > lower_limit) & (y < upper_limit)]

        if (len(X_train_filtered) == 0):
            percentile_model_stats[model_name]['Training_R2'][(l_pct, u_pct)] = np.nan
            percentile_model_stats[model_name]['Training_RMSE'][(l_pct, u_pct)] = np.nan
            percentile_model_stats[model_name]['Training_MBE'][(l_pct, u_pct)] = np.nan
            #percentile_model_stats[model_name]['Training_CRMSE'][(l_pct, u_pct)] = np.nan

        else:
            percentile_model_stats[model_name]['Training_R2'][(l_pct, u_pct)] = round(current_model.score(X_train_filtered, y_train_filtered), 2)
            percentile_model_stats[model_name]['Training_RMSE'][(l_pct, u_pct)] = (np.sqrt(mean_squared_error(y_train_filtered, y_train_predicted_filtered)))
            percentile_model_stats[model_name]['Training_MBE'][(l_pct, u_pct)] = np.mean(y_train_predicted_filtered - y_train_filtered)
            #percentile_model_stats[model_name]['Training_CRMSE'][(l_pct, u_pct)] = np.sqrt(np.mean(((y_train_predicted_filtered - np.mean(y_train_predicted_filtered)) - (y_train_filtered - np.mean(y_train_filtered))) ** 2))

        if  (len(X_test_filtered) == 0):
            percentile_model_stats[model_name]['Testing_R2'][(l_pct, u_pct)] = np.nan
            percentile_model_stats[model_name]['Testing_RMSE'][(l_pct, u_pct)] = np.nan
            percentile_model_stats[model_name]['Testing_MBE'][(l_pct, u_pct)] = np.nan
            #percentile_model_stats[model_name]['Testing_CRMSE'][(l_pct, u_pct)] = np.nan
        else:
            percentile_model_stats[model_name]['Testing_R2'][(l_pct, u_pct)] = round(current_model.score(X_test_filtered, y_test_filtered), 2)
            percentile_model_stats[model_name]['Testing_RMSE'][(l_pct, u_pct)] = (np.sqrt(mean_squared_error(y_test_filtered, y_test_predicted_filtered)))
            percentile_model_stats[model_name]['Testing_MBE'][(l_pct, u_pct)] = np.mean(y_test_predicted_filtered - y_test_filtered)
            #percentile_model_stats[model_name]['Testing_CRMSE'][(l_pct, u_pct)] = np.sqrt(np.mean(((y_test_predicted_filtered - np.mean( y_test_predicted_filtered)) - (y_test_filtered - np.mean(y_test_filtered))) ** 2))

        if (len(X_filtered) == 0):
            percentile_model_stats[model_name]['R2'][(l_pct, u_pct)] = np.nan
            percentile_model_stats[model_name]['RMSE'][(l_pct, u_pct)] = np.nan
            percentile_model_stats[model_name]['MBE'][(l_pct, u_pct)] = np.nan
            #percentile_model_stats[model_name]['CRMSE'][(l_pct, u_pct)] = np.nan

        else:
            percentile_model_stats[model_name]['R2'][(l_pct, u_pct)] = round(current_model.score(X_filtered, y_filtered), 2)
            percentile_model_stats[model_name]['RMSE'][(l_pct, u_pct)] = (np.sqrt(mean_squared_error(y_filtered, y_predicted_filtered)))
            percentile_model_stats[model_name]['MBE'][(l_pct, u_pct)] = np.mean(y_predicted_filtered - y_filtered)
            #percentile_model_stats[model_name]['CRMSE'][(l_pct, u_pct)] = np.sqrt(np.mean(((y_predicted_filtered - np.mean(y_predicted_filtered)) - (y_filtered - np.mean(y_filtered))) ** 2))


    return percentile_model_stats

def save_outputs_tensorflow(model_stats, model_name, current_model, X_train, X_test, y_train, y_test,
                               y_train_predicted, y_test_predicted, X_std, y, y_predicted):
    # save the model statistics
    model_stats['Training_R2'][model_name] = r2_score(y_train, y_train_predicted)
    model_stats['Testing_R2'][model_name] = r2_score(y_test, y_test_predicted)
    model_stats['R2'][model_name] = r2_score(y, y_predicted)
    model_stats['Training_RMSE'][model_name] = (np.sqrt(mean_squared_error(y_train, y_train_predicted)))
    model_stats['Testing_RMSE'][model_name] = (np.sqrt(mean_squared_error(y_test, y_test_predicted)))
    model_stats['RMSE'][model_name] = (np.sqrt(mean_squared_error(y, y_predicted)))
    model_stats['Training_MBE'][model_name] = np.mean(y_train_predicted - y_train)
    model_stats['Testing_MBE'][model_name] = np.mean(y_test_predicted - y_test)
    model_stats['MBE'][model_name] = np.mean(y_predicted - y)

    return model_stats

def save_percentile_outputs_tensorflow(percentile_model_stats, model_name, current_model, X_train, X_test,
                                                     y_train, y_test, y_train_predicted, y_test_predicted, X_std, y,
                                                     y_predicted):
    # save the model statistics
    for l_pct, u_pct in percentile_model_stats[model_name].index:
        lower_limit = np.percentile(y, l_pct)
        upper_limit = np.percentile(y, u_pct)
        # Filter training data to only include points where y_train is above the 75th percentile of the combined y
        X_train_filtered = X_train[(y_train > lower_limit) & (y_train < upper_limit)]
        y_train_filtered = y_train[(y_train > lower_limit) & (y_train < upper_limit)]
        y_train_predicted_filtered = y_train_predicted[(y_train > lower_limit) & (y_train < upper_limit)]

        # Filter test data to only include points where y_test is above the 75th percentile of the combined y
        X_test_filtered = X_test[(y_test > lower_limit) & (y_test < upper_limit)]
        y_test_filtered = y_test[(y_test > lower_limit) & (y_test < upper_limit)]
        y_test_predicted_filtered = y_test_predicted[(y_test > lower_limit) & (y_test < upper_limit)]

        # Filter test data to only include points where y_test is above the 75th percentile of the combined y
        X_filtered = X_std[(y > lower_limit) & (y < upper_limit)]
        y_filtered = y[(y > lower_limit) & (y < upper_limit)]
        y_predicted_filtered = y_predicted[(y > lower_limit) & (y < upper_limit)]

        if (len(X_train_filtered) == 0):
            percentile_model_stats[model_name]['Training_R2'][(l_pct, u_pct)] = np.nan
            percentile_model_stats[model_name]['Training_RMSE'][(l_pct, u_pct)] = np.nan
            percentile_model_stats[model_name]['Training_MBE'][(l_pct, u_pct)] = np.nan

        else:
            percentile_model_stats[model_name]['Training_R2'][(l_pct, u_pct)] = r2_score(y_train, y_train_predicted)
            percentile_model_stats[model_name]['Training_RMSE'][(l_pct, u_pct)] = (
                np.sqrt(mean_squared_error(y_train_filtered, y_train_predicted_filtered)))
            percentile_model_stats[model_name]['Training_MBE'][(l_pct, u_pct)] = np.mean(
                y_train_predicted_filtered - y_train_filtered)

        if (len(X_test_filtered) == 0):
            percentile_model_stats[model_name]['Testing_R2'][(l_pct, u_pct)] = np.nan
            percentile_model_stats[model_name]['Testing_RMSE'][(l_pct, u_pct)] = np.nan
            percentile_model_stats[model_name]['Testing_MBE'][(l_pct, u_pct)] = np.nan

        else:
            percentile_model_stats[model_name]['Testing_R2'][(l_pct, u_pct)] = r2_score(y_test, y_test_predicted)
            percentile_model_stats[model_name]['Testing_RMSE'][(l_pct, u_pct)] = (
                np.sqrt(mean_squared_error(y_test_filtered, y_test_predicted_filtered)))
            percentile_model_stats[model_name]['Testing_MBE'][(l_pct, u_pct)] = np.mean(
                y_test_predicted_filtered - y_test_filtered)

        if (len(X_filtered) == 0):
            percentile_model_stats[model_name]['R2'][(l_pct, u_pct)] = np.nan
            percentile_model_stats[model_name]['RMSE'][(l_pct, u_pct)] = np.nan
            percentile_model_stats[model_name]['MBE'][(l_pct, u_pct)] = np.nan


        else:
            percentile_model_stats[model_name]['R2'][(l_pct, u_pct)] = r2_score(y, y_predicted)
            percentile_model_stats[model_name]['RMSE'][(l_pct, u_pct)] = (
                np.sqrt(mean_squared_error(y_filtered, y_predicted_filtered)))
            percentile_model_stats[model_name]['MBE'][(l_pct, u_pct)] = np.mean(y_predicted_filtered - y_filtered)

    return percentile_model_stats

def ann(X_train, y_train, X_test, y_test, X_std, y, model_name, model_stats, percentile_model_stats, settings):
    current_model = Sequential([
        Dense(10, activation=relu),
        Dropout(.1),
        Dense(10, activation=relu),
        Dropout(.1),
        Dense(10, activation=relu),
        Dropout(.1),
        Dense(1)
    ])

    current_model.compile(optimizer=tf.optimizers.Adam(), loss='mse', metrics=['mae', 'mse'])

    # this helps makes our output less verbose but still shows progress
    class PrintDot(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs):
            if epoch % 100 == 0: print('')
            print('.', end='')


    early_stop = keras.callbacks.EarlyStopping(monitor='val_mse', patience=15)


    current_model.fit(X_train, y_train, epochs=1000, verbose=0, validation_split=0.1,
                         callbacks=[early_stop, PrintDot()])

    # get the predicted y values for the model
    y_train_predicted = current_model.predict(X_train).squeeze()
    y_test_predicted = current_model.predict(X_test).squeeze()
    y_predicted = current_model.predict(X_std).squeeze()

    model_stats = save_outputs_tensorflow(model_stats, model_name, current_model, X_train, X_test, y_train, y_test,
                                          y_train_predicted, y_test_predicted, X_std, y, y_predicted)
    percentile_model_stats = save_percentile_outputs_tensorflow(percentile_model_stats, model_name, current_model,
                                                                X_train, X_test,
                                                                y_train, y_test, y_train_predicted, y_test_predicted,
                                                                X_std, y,
                                                                y_predicted)

    return percentile_model_stats, model_stats, y_train_predicted, y_test_predicted, y_predicted, current_model

def lin_reg(X_train, y_train, X_test, y_test, X_std, y, model_name, model_stats, percentile_model_stats, settings):
    # Instantiate the LR model:
    current_model = LinearRegression()
    # train the model using training data
    current_model.fit(X_train, y_train)

    # get the predicted y values for the model
    y_train_predicted = current_model.predict(X_train)
    y_test_predicted = current_model.predict(X_test)
    y_predicted = current_model.predict(X_std)

    model_stats = save_outputs(model_stats, model_name, current_model, X_train, X_test, y_train,y_test, y_train_predicted, y_test_predicted, X_std, y, y_predicted)
    percentile_model_stats = save_percentile_outputs(percentile_model_stats, model_name, current_model, X_train, X_test, y_train,y_test, y_train_predicted, y_test_predicted, X_std, y, y_predicted)

    return percentile_model_stats, model_stats, y_train_predicted, y_test_predicted, y_predicted, current_model

def lasso(X_train, y_train, X_test, y_test, X_std, y, model_name, model_stats, percentile_model_stats, settings):
    lasso_params_grid = {'alpha': np.arange(0.001, 10, 0.05)}
    lasso_model = Lasso()

    # Create folds that are 5 chunks without shuffling the data
    kf = KFold(n_splits=5, shuffle=False)

    # cross validation to find the best parameter (alpha)
    lasso_cv = RandomizedSearchCV(lasso_model, lasso_params_grid, cv=kf, scoring='neg_mean_squared_error',
                                  random_state=42)

    # train the cross validation models
    lasso_cv.fit(X_train, y_train)
    # get the best parameters
    best_params = lasso_cv.best_params_

    # train a new model with the best parameters
    current_model = Lasso(**best_params)
    current_model.fit(X_train, y_train)

    # get the predicted y values for the model
    y_train_predicted = current_model.predict(X_train)
    y_test_predicted = current_model.predict(X_test)
    y_predicted = current_model.predict(X_std)

    # save the model statistics
    model_stats = save_outputs(model_stats, model_name, current_model, X_train, X_test, y_train,y_test, y_train_predicted, y_test_predicted, X_std, y, y_predicted)
    percentile_model_stats = save_percentile_outputs(percentile_model_stats, model_name, current_model, X_train, X_test,
                                                     y_train, y_test, y_train_predicted, y_test_predicted, X_std, y, y_predicted)

    return percentile_model_stats, model_stats, y_train_predicted, y_test_predicted, y_predicted, current_model

def ridge(X_train, y_train, X_test, y_test, X_std, y, model_name, model_stats, percentile_model_stats, settings):
    ridge_params_grid = {'alpha': np.arange(0.001, 10, 0.05)}
    ridge_model = Ridge()
    # Create folds that are 5 chunks without shuffling the data
    kf = KFold(n_splits=5, shuffle=False)

    # cross validation to find the best parameter (alpha)
    ridge_cv = RandomizedSearchCV(ridge_model, ridge_params_grid, cv=kf, scoring='neg_mean_squared_error',
                                  random_state=42)
    # train the cross validation models
    ridge_cv.fit(X_train, y_train)
    # get the best parameters
    best_params = ridge_cv.best_params_
    # train a new model with the best parameters
    current_model = Ridge(**best_params)
    current_model.fit(X_train, y_train)

    # get the predicted y values for the model
    y_train_predicted = current_model.predict(X_train)
    y_test_predicted = current_model.predict(X_test)
    y_predicted = current_model.predict(X_std)

    # save the model statistics
    model_stats = save_outputs(model_stats, model_name, current_model, X_train, X_test, y_train,y_test, y_train_predicted, y_test_predicted, X_std, y, y_predicted)
    percentile_model_stats = save_percentile_outputs(percentile_model_stats, model_name, current_model, X_train, X_test,
                                                     y_train, y_test, y_train_predicted, y_test_predicted, X_std, y, y_predicted)
    return percentile_model_stats, model_stats, y_train_predicted, y_test_predicted, y_predicted, current_model

def random_forest(X_train, y_train, X_test, y_test, X_std, y, model_name, model_stats, percentile_model_stats, settings):
    rf_regressor = RandomForestRegressor(random_state=42)
    rf_params = {'max_depth': np.arange(1, 10, 1),
                 'min_samples_split': np.arange(2, 50, 1),
                 'min_samples_leaf': np.arange(2, 50, 1),
                 'max_features': ['sqrt', 'log2']}
    # Create folds that are 5 chunks without shuffling the data
    kf = KFold(n_splits=5, shuffle=False)

    # cross validation to find the best parameter (see rf_params)
    rf_regressor_cv = RandomizedSearchCV(rf_regressor, rf_params, cv=kf, scoring='neg_mean_squared_error',
                                         random_state=42)
    # train the cross validation models
    rf_regressor_cv.fit(X_train, y_train)
    # get the best parameters
    best_params = rf_regressor_cv.best_params_
    np.random.seed(42)

    # train a new model with the best parameters
    current_model = RandomForestRegressor(**best_params)
    current_model.fit(X_train, y_train)

    # get the predicted y values for the model
    y_train_predicted = current_model.predict(X_train)
    y_test_predicted = current_model.predict(X_test)
    y_predicted = current_model.predict(X_std)

    # save the model statistics
    model_stats = save_outputs(model_stats, model_name, current_model, X_train, X_test, y_train,y_test, y_train_predicted, y_test_predicted, X_std, y, y_predicted)
    percentile_model_stats = save_percentile_outputs(percentile_model_stats, model_name, current_model, X_train, X_test,
                                                     y_train, y_test, y_train_predicted, y_test_predicted, X_std, y, y_predicted)

    return percentile_model_stats, model_stats, y_train_predicted, y_test_predicted, y_predicted, current_model

def adaboost(X_train, y_train, X_test, y_test, X_std, y, model_name, model_stats, percentile_model_stats, settings):
    #testing Adaboost model on data with reduced features
    adaboost = AdaBoostRegressor(random_state=42)
    adaboost_params = {'learning_rate' : np.arange(0.01,1.5,0.05),
                            'n_estimators': np.arange(10,200,10),
                            'loss' : ['linear', 'square', 'exponential']}

    # Create folds that are 5 chunks without shuffling the data
    kf = KFold(n_splits=5, shuffle=False)

    adaboost_cv = RandomizedSearchCV(adaboost, adaboost_params, random_state=42, cv=kf, scoring='neg_mean_squared_error')
    # train the cross validation models
    adaboost_cv.fit(X_train, y_train)

    # get the best parameters
    best_params = adaboost_cv.best_params_
    # train a new model with the best parameters
    current_model = AdaBoostRegressor(**best_params)
    current_model.fit(X_train, y_train)

    # get the predicted y values for the model
    y_train_predicted = current_model.predict(X_train)
    y_test_predicted = current_model.predict(X_test)
    y_predicted = current_model.predict(X_std)

    # save the model statistics
    model_stats = save_outputs(model_stats, model_name, current_model, X_train, X_test, y_train, y_test, y_train_predicted,
                               y_test_predicted, X_std, y, y_predicted)
    percentile_model_stats = save_percentile_outputs(percentile_model_stats, model_name, current_model, X_train, X_test,
                                                     y_train, y_test, y_train_predicted, y_test_predicted, X_std, y, y_predicted)

    return percentile_model_stats, model_stats, y_train_predicted, y_test_predicted, y_predicted, current_model

def gradboost(X_train, y_train, X_test, y_test, X_std, y, model_name, model_stats, percentile_model_stats, settings):
    gb_regressor = GradientBoostingRegressor(random_state=42)
    gb_params = dict(learning_rate= [0.01, 0.05, 0.1, 0.2, 0.3],
                     n_estimators=[50, 100, 300, 500, 750, 1000],
                     subsample= [0.3, 0.5, 0.7, 0.9, 1.0],
                     max_depth= [1, 3, 5, 7, 10],
                     max_features=['sqrt', 'log2', None],
                     min_samples_split= [2, 5, 10],
                     min_samples_leaf=[1, 3, 5, 10])

    ''' #old params
    dict(learning_rate=np.arange(0.05, 0.3, 0.05),
                     n_estimators=[50, 100, 300, 500, 800, 1000],
                     subsample=np.arange(0.1, 0.9, 0.05),
                     max_depth=[int(i) for i in np.arange(1, 10, 1)],
                     max_features=['sqrt', 'log2'])'''

    # Create folds that are 5 chunks without shuffling the data
    kf = KFold(n_splits=5, shuffle=False)

    # cross validation to find the best parameter (see rf_params)
    gb_cv = RandomizedSearchCV(gb_regressor, gb_params, random_state=42, cv=kf, scoring='neg_mean_squared_error')

    # train the cross validation models
    gb_cv.fit(X_train, y_train)
    # get the best parameters
    best_params = gb_cv.best_params_
    np.random.seed(42)

    # train a new model with the best parameters
    current_model = GradientBoostingRegressor(**best_params)
    current_model.fit(X_train, y_train)

    # get the predicted y values for the model
    y_train_predicted = current_model.predict(X_train)
    y_test_predicted = current_model.predict(X_test)
    y_predicted = current_model.predict(X_std)

    # save the model statistics
    model_stats = save_outputs(model_stats, model_name, current_model, X_train, X_test, y_train,y_test, y_train_predicted, y_test_predicted, X_std, y, y_predicted)
    percentile_model_stats = save_percentile_outputs(percentile_model_stats, model_name, current_model, X_train, X_test,
                                                     y_train, y_test, y_train_predicted, y_test_predicted, X_std, y, y_predicted)

    return percentile_model_stats, model_stats, y_train_predicted, y_test_predicted, y_predicted, current_model

def xg_boost(X_train, y_train, X_test, y_test, X_std, y, model_name, model_stats, percentile_model_stats, settings):
    # Create an XGBoost regressor object
    xgb_regressor = xgb.XGBRegressor()

    # Define the parameter grid for RandomizedSearchCV
    xgb_params = {
    'n_estimators': [50, 100, 300, 500, 800, 1000],
    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],  # Default is 0.3. Ranges from loc to loc+scale.
    'subsample': [0.3, 0.5, 0.7, 0.9, 1.0],
    'colsample_bytree': [0.5, 0.75, 1.0],
    "min_child_weight": [1, 3, 5, 7, 10],  # Default is 1
    "max_depth": [1, 3, 5, 7, 10],  # Default is 6
    "alpha": [0, 0.01, 1, 5, 10, 50, 100],  # Default is 0. AKA reg_alpha.
    "lambda": [0, 0.01, 1, 5, 10, 50, 100]  # Default is 0. AKA reg_lambda.
    }

    ''' #old params
    xgb_params = {
    'n_estimators': [50, 100, 300, 500, 800, 1000],
    "learning_rate": scipy.stats.uniform(loc=0.003, scale=0.19),  # Default is 0.3. Ranges from loc to loc+scale.
    #"subsample": scipy.stats.uniform(loc=0.5, scale=0.5),  # Default is 1
    #"colsample_bytree": scipy.stats.uniform(loc=0.5, scale=0.5),  # Default is 1
    "min_child_weight": [1, 3, 5, 7,10],  # Default is 1
    "max_depth": np.append(0, np.arange(1, 10)),  # Default is 6
    "alpha": [0, 0.01, 1, 5, 10, 50, 100],  # Default is 0. AKA reg_alpha.
    "lambda": [0, 0.01, 1, 5, 10, 50, 100]  # Default is 0. AKA reg_lambda.
    }'''


    # Create folds that are 5 chunks without shuffling the data
    kf = KFold(n_splits=5, shuffle=False)

    # Create a RandomizedSearchCV object
    xgb_cv = RandomizedSearchCV(
        estimator=xgb_regressor,
        param_distributions=xgb_params,
        n_iter=200,  # Number of parameter settings that are sampled
        scoring='neg_mean_squared_error',  # Use negative MSE as the scoring metric
        cv=kf,  # Number of cross-validation folds
        random_state=42,  # Set random state for reproducibility
        n_jobs=-1,  # Use all available cores for parallel processing
    )

    # train the cross validation models
    xgb_cv.fit(X_train, y_train)
    # get the best parameters
    best_params = xgb_cv.best_params_
    np.random.seed(42)

    # train a new model with the best parameters
    current_model = xgb.XGBRegressor(**best_params)
    current_model.fit(X_train, y_train)
    # get the predicted y values for the model
    y_train_predicted = current_model.predict(X_train)
    y_test_predicted = current_model.predict(X_test)
    y_predicted = current_model.predict(X_std)

    # save the model statistics
    model_stats = save_outputs(model_stats, model_name, current_model, X_train, X_test, y_train, y_test,
                               y_train_predicted, y_test_predicted, X_std, y, y_predicted)
    percentile_model_stats = save_percentile_outputs(percentile_model_stats, model_name, current_model, X_train, X_test,
                                                     y_train, y_test, y_train_predicted, y_test_predicted, X_std, y,
                                                     y_predicted)

    return percentile_model_stats, model_stats, y_train_predicted, y_test_predicted, y_predicted, current_model

def svr_(X_train, y_train, X_test, y_test, X_std, y, model_name, model_stats, percentile_model_stats, settings):
    svr_regressor = SVR()
    svr_params = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                  'gamma': ['scale', 'auto'],
                  'C': [0.1, 1, 10, 100]}

    # Create folds that are 5 chunks without shuffling the data
    kf = KFold(n_splits=5, shuffle=False)

    # cross validation to find the best parameter (see rf_params)
    svr_cv = RandomizedSearchCV(svr_regressor, svr_params, random_state=42, cv=kf, scoring='neg_mean_squared_error')

    # train the cross validation models
    svr_cv.fit(X_train, y_train)

    # get the best parameters
    best_params = svr_cv.best_params_
    np.random.seed(42)

    # train a new model with the best parameters
    current_model = SVR(**best_params)
    current_model.fit(X_train, y_train)

    # get the predicted y values for the model
    y_train_predicted = current_model.predict(X_train)
    y_test_predicted = current_model.predict(X_test)
    y_predicted = current_model.predict(X_std)

    # save the model statistics
    model_stats = save_outputs(model_stats, model_name, current_model, X_train, X_test, y_train,y_test, y_train_predicted, y_test_predicted, X_std, y, y_predicted)
    percentile_model_stats = save_percentile_outputs(percentile_model_stats, model_name, current_model, X_train, X_test,
                                                     y_train, y_test, y_train_predicted, y_test_predicted, X_std, y, y_predicted)

    return percentile_model_stats, model_stats, y_train_predicted, y_test_predicted, y_predicted, current_model

def random_forest_SMOTER(X_train, y_train, X_test, y_test, X_std, y, model_name, model_stats, percentile_model_stats, settings):
    X_train = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
    y_train = pd.Series(y_train)
    y_train = np.array(y_train).ravel()

    # Add the target column to X_train
    X_train['target'] = y_train

    # Apply SMOGN to the combined data
    X_train_smogn = smogn.smoter(data=X_train, y='target', k=5,  rel_thres=0.2, rel_xtrm_type = 'high', samp_method = 'balance',rel_method= 'auto')

    # Separate the resampled data
    y_train = X_train_smogn['target']
    X_train = X_train_smogn.drop(columns=['target'])

    rf_regressor = RandomForestRegressor(random_state=42)
    rf_params = {'max_depth': np.arange(1, 10, 1),
                 'min_samples_split': np.arange(2, 50, 1),
                 'min_samples_leaf': np.arange(2, 50, 1),
                 'max_features': ['sqrt', 'log2']}
    # Create folds that are 5 chunks without shuffling the data
    kf = KFold(n_splits=5, shuffle=False)

    # cross validation to find the best parameter (see rf_params)
    rf_regressor_cv = RandomizedSearchCV(rf_regressor, rf_params, cv=kf, scoring='neg_mean_squared_error',
                                         random_state=42)
    # train the cross validation models
    rf_regressor_cv.fit(X_train, y_train)
    # get the best parameters
    best_params = rf_regressor_cv.best_params_
    np.random.seed(42)

    # train a new model with the best parameters
    current_model = RandomForestRegressor(**best_params)
    current_model.fit(X_train, y_train)

    # get the predicted y values for the model
    y_train_predicted = current_model.predict(X_train)
    y_test_predicted = current_model.predict(X_test)
    y_predicted = current_model.predict(X_std)

    # save the model statistics
    model_stats = save_outputs(model_stats, model_name, current_model, X_train, X_test, y_train,y_test, y_train_predicted, y_test_predicted, X_std, y, y_predicted)
    percentile_model_stats = save_percentile_outputs(percentile_model_stats, model_name, current_model, X_train, X_test,
                                                     y_train, y_test, y_train_predicted, y_test_predicted, X_std, y, y_predicted)

    return percentile_model_stats, model_stats, y_train_predicted, y_test_predicted, y_predicted, current_model
def random_forest_sigweight(X_train, y_train, X_test, y_test, X_std, y, model_name, model_stats, percentile_model_stats, settings):
    y_train_zscore = scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    weights = 1/(1+np.exp(-y_train_zscore))**2
    rf_regressor = RandomForestRegressor(random_state=42)
    rf_params = {'max_depth': np.arange(1, 10, 1),
                 'min_samples_split': np.arange(2, 50, 1),
                 'min_samples_leaf': np.arange(2, 50, 1),
                 'max_features': ['sqrt', 'log2']}
    # Create folds that are 5 chunks without shuffling the data
    kf = KFold(n_splits=5, shuffle=False)

    # cross validation to find the best parameter (see rf_params)
    rf_regressor_cv = RandomizedSearchCV(rf_regressor, rf_params, cv=kf, scoring='neg_mean_squared_error',
                                         random_state=42)
    # train the cross validation models
    rf_regressor_cv.fit(X_train, y_train, sample_weight=weights)
    # get the best parameters
    best_params = rf_regressor_cv.best_params_
    np.random.seed(42)

    # train a new model with the best parameters
    current_model = RandomForestRegressor(**best_params)
    current_model.fit(X_train, y_train, sample_weight=weights)

    # get the predicted y values for the model
    y_train_predicted = current_model.predict(X_train)
    y_test_predicted = current_model.predict(X_test)
    y_predicted = current_model.predict(X_std)

    # save the model statistics
    model_stats = save_outputs(model_stats, model_name, current_model, X_train, X_test, y_train,y_test, y_train_predicted, y_test_predicted, X_std, y, y_predicted)
    percentile_model_stats = save_percentile_outputs(percentile_model_stats, model_name, current_model, X_train, X_test,
                                                     y_train, y_test, y_train_predicted, y_test_predicted, X_std, y, y_predicted)

    return percentile_model_stats, model_stats, y_train_predicted, y_test_predicted, y_predicted, current_model

def random_forest_sigweight0_5(X_train, y_train, X_test, y_test, X_std, y, model_name, model_stats, percentile_model_stats, settings):
    y_train_zscore = scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    weights = 1/(1+np.exp(-y_train_zscore+0.5))**2
    rf_regressor = RandomForestRegressor(random_state=42)
    rf_params = {'max_depth': np.arange(1, 10, 1),
                 'min_samples_split': np.arange(2, 50, 1),
                 'min_samples_leaf': np.arange(2, 50, 1),
                 'max_features': ['sqrt', 'log2']}
    # Create folds that are 5 chunks without shuffling the data
    kf = KFold(n_splits=5, shuffle=False)

    # cross validation to find the best parameter (see rf_params)
    rf_regressor_cv = RandomizedSearchCV(rf_regressor, rf_params, cv=kf, scoring='neg_mean_squared_error',
                                         random_state=42)
    # train the cross validation models
    rf_regressor_cv.fit(X_train, y_train, sample_weight=weights)
    # get the best parameters
    best_params = rf_regressor_cv.best_params_
    np.random.seed(42)

    # train a new model with the best parameters
    current_model = RandomForestRegressor(**best_params)
    current_model.fit(X_train, y_train, sample_weight=weights)

    # get the predicted y values for the model
    y_train_predicted = current_model.predict(X_train)
    y_test_predicted = current_model.predict(X_test)
    y_predicted = current_model.predict(X_std)

    # save the model statistics
    model_stats = save_outputs(model_stats, model_name, current_model, X_train, X_test, y_train,y_test, y_train_predicted, y_test_predicted, X_std, y, y_predicted)
    percentile_model_stats = save_percentile_outputs(percentile_model_stats, model_name, current_model, X_train, X_test,
                                                     y_train, y_test, y_train_predicted, y_test_predicted, X_std, y, y_predicted)

    return percentile_model_stats, model_stats, y_train_predicted, y_test_predicted, y_predicted, current_model

def random_forest_sigweight1(X_train, y_train, X_test, y_test, X_std, y, model_name, model_stats, percentile_model_stats, settings):
    y_train_zscore = scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    weights = 1/(1+np.exp(-y_train_zscore+1))**2
    rf_regressor = RandomForestRegressor(random_state=42)
    rf_params = {'max_depth': np.arange(1, 10, 1),
                 'min_samples_split': np.arange(2, 50, 1),
                 'min_samples_leaf': np.arange(2, 50, 1),
                 'max_features': ['sqrt', 'log2']}

    # Create folds that are 5 chunks without shuffling the data
    kf = KFold(n_splits=5, shuffle=False)

    # cross validation to find the best parameter (see rf_params)
    rf_regressor_cv = RandomizedSearchCV(rf_regressor, rf_params, cv=kf, scoring='neg_mean_squared_error',
                                         random_state=42)
    # train the cross validation models
    rf_regressor_cv.fit(X_train, y_train, sample_weight=weights)
    # get the best parameters
    best_params = rf_regressor_cv.best_params_
    np.random.seed(42)

    # train a new model with the best parameters
    current_model = RandomForestRegressor(**best_params)
    current_model.fit(X_train, y_train, sample_weight=weights)

    # get the predicted y values for the model
    y_train_predicted = current_model.predict(X_train)
    y_test_predicted = current_model.predict(X_test)
    y_predicted = current_model.predict(X_std)

    # save the model statistics
    model_stats = save_outputs(model_stats, model_name, current_model, X_train, X_test, y_train,y_test, y_train_predicted, y_test_predicted, X_std, y, y_predicted)
    percentile_model_stats = save_percentile_outputs(percentile_model_stats, model_name, current_model, X_train, X_test,
                                                     y_train, y_test, y_train_predicted, y_test_predicted, X_std, y, y_predicted)

    return percentile_model_stats, model_stats, y_train_predicted, y_test_predicted, y_predicted, current_model

def random_forest_sigweight1_5(X_train, y_train, X_test, y_test, X_std, y, model_name, model_stats, percentile_model_stats, settings):
    y_train_zscore = scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    weights = 1/(1+np.exp(-y_train_zscore+1.5))**2
    rf_regressor = RandomForestRegressor(random_state=42)
    rf_params = {'max_depth': np.arange(1, 10, 1),
                 'min_samples_split': np.arange(2, 50, 1),
                 'min_samples_leaf': np.arange(2, 50, 1),
                 'max_features': ['sqrt', 'log2']}
    # Create folds that are 5 chunks without shuffling the data
    kf = KFold(n_splits=5, shuffle=False)

    # cross validation to find the best parameter (see rf_params)
    rf_regressor_cv = RandomizedSearchCV(rf_regressor, rf_params, cv=kf, scoring='neg_mean_squared_error',
                                         random_state=42)
    # train the cross validation models
    rf_regressor_cv.fit(X_train, y_train, sample_weight=weights)
    # get the best parameters
    best_params = rf_regressor_cv.best_params_
    np.random.seed(42)

    # train a new model with the best parameters
    current_model = RandomForestRegressor(**best_params)
    current_model.fit(X_train, y_train, sample_weight=weights)

    # get the predicted y values for the model
    y_train_predicted = current_model.predict(X_train)
    y_test_predicted = current_model.predict(X_test)
    y_predicted = current_model.predict(X_std)

    # save the model statistics
    model_stats = save_outputs(model_stats, model_name, current_model, X_train, X_test, y_train,y_test, y_train_predicted, y_test_predicted, X_std, y, y_predicted)
    percentile_model_stats = save_percentile_outputs(percentile_model_stats, model_name, current_model, X_train, X_test,
                                                     y_train, y_test, y_train_predicted, y_test_predicted, X_std, y, y_predicted)

    return percentile_model_stats, model_stats, y_train_predicted, y_test_predicted, y_predicted, current_model

def random_forest_sigweight2(X_train, y_train, X_test, y_test, X_std, y, model_name, model_stats, percentile_model_stats, settings):
    y_train_zscore = scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    weights = 1/(1+np.exp(-y_train_zscore+2))**2
    rf_regressor = RandomForestRegressor(random_state=42)
    rf_params = {'max_depth': np.arange(1, 10, 1),
                 'min_samples_split': np.arange(2, 50, 1),
                 'min_samples_leaf': np.arange(2, 50, 1),
                 'max_features': ['sqrt', 'log2']}
    # Create folds that are 5 chunks without shuffling the data
    kf = KFold(n_splits=5, shuffle=False)

    # cross validation to find the best parameter (see rf_params)
    rf_regressor_cv = RandomizedSearchCV(rf_regressor, rf_params, cv=kf, scoring='neg_mean_squared_error',
                                         random_state=42)
    # train the cross validation models
    rf_regressor_cv.fit(X_train, y_train, sample_weight=weights)
    # get the best parameters
    best_params = rf_regressor_cv.best_params_
    np.random.seed(42)

    # train a new model with the best parameters
    current_model = RandomForestRegressor(**best_params)
    current_model.fit(X_train, y_train, sample_weight=weights)

    # get the predicted y values for the model
    y_train_predicted = current_model.predict(X_train)
    y_test_predicted = current_model.predict(X_test)
    y_predicted = current_model.predict(X_std)

    # save the model statistics
    model_stats = save_outputs(model_stats, model_name, current_model, X_train, X_test, y_train,y_test, y_train_predicted, y_test_predicted, X_std, y, y_predicted)
    percentile_model_stats = save_percentile_outputs(percentile_model_stats, model_name, current_model, X_train, X_test,
                                                     y_train, y_test, y_train_predicted, y_test_predicted, X_std, y, y_predicted)

    return percentile_model_stats, model_stats, y_train_predicted, y_test_predicted, y_predicted, current_model

def random_forest_sigweight2_5(X_train, y_train, X_test, y_test, X_std, y, model_name, model_stats, percentile_model_stats, settings):
    y_train_zscore = scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    weights = 1/(1+np.exp(-y_train_zscore+2.5))**2
    rf_regressor = RandomForestRegressor(random_state=42)
    rf_params = {'max_depth': np.arange(1, 10, 1),
                 'min_samples_split': np.arange(2, 50, 1),
                 'min_samples_leaf': np.arange(2, 50, 1),
                 'max_features': ['sqrt', 'log2']}
    # Create folds that are 5 chunks without shuffling the data
    kf = KFold(n_splits=5, shuffle=False)

    # cross validation to find the best parameter (see rf_params)
    rf_regressor_cv = RandomizedSearchCV(rf_regressor, rf_params, cv=kf, scoring='neg_mean_squared_error',
                                         random_state=42)
    # train the cross validation models
    rf_regressor_cv.fit(X_train, y_train, sample_weight=weights)
    # get the best parameters
    best_params = rf_regressor_cv.best_params_
    np.random.seed(42)

    # train a new model with the best parameters
    current_model = RandomForestRegressor(**best_params)
    current_model.fit(X_train, y_train, sample_weight=weights)

    # get the predicted y values for the model
    y_train_predicted = current_model.predict(X_train)
    y_test_predicted = current_model.predict(X_test)
    y_predicted = current_model.predict(X_std)

    # save the model statistics
    model_stats = save_outputs(model_stats, model_name, current_model, X_train, X_test, y_train,y_test, y_train_predicted, y_test_predicted, X_std, y, y_predicted)
    percentile_model_stats = save_percentile_outputs(percentile_model_stats, model_name, current_model, X_train, X_test,
                                                     y_train, y_test, y_train_predicted, y_test_predicted, X_std, y, y_predicted)

    return percentile_model_stats, model_stats, y_train_predicted, y_test_predicted, y_predicted, current_model

def random_forest_sigweight3(X_train, y_train, X_test, y_test, X_std, y, model_name, model_stats, percentile_model_stats, settings):
    y_train_zscore = scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    weights = 1/(1+np.exp(-y_train_zscore+3))**2
    rf_regressor = RandomForestRegressor(random_state=42)
    rf_params = {'max_depth': np.arange(1, 10, 1),
                 'min_samples_split': np.arange(2, 50, 1),
                 'min_samples_leaf': np.arange(2, 50, 1),
                 'n_estimators': np.arange(20, 301, 20),
                 'max_features': ['sqrt', 'log2']}
    # Create folds that are 5 chunks without shuffling the data
    kf = KFold(n_splits=5, shuffle=False)

    # cross validation to find the best parameter (see rf_params)
    rf_regressor_cv = RandomizedSearchCV(rf_regressor, rf_params, cv=kf, scoring='neg_mean_squared_error',
                                         random_state=42)
    # train the cross validation models
    rf_regressor_cv.fit(X_train, y_train, sample_weight=weights)
    # get the best parameters
    best_params = rf_regressor_cv.best_params_
    np.random.seed(42)

    # train a new model with the best parameters
    current_model = RandomForestRegressor(**best_params)
    current_model.fit(X_train, y_train, sample_weight=weights)

    # get the predicted y values for the model
    y_train_predicted = current_model.predict(X_train)
    y_test_predicted = current_model.predict(X_test)
    y_predicted = current_model.predict(X_std)

    # save the model statistics
    model_stats = save_outputs(model_stats, model_name, current_model, X_train, X_test, y_train,y_test, y_train_predicted, y_test_predicted, X_std, y, y_predicted)
    percentile_model_stats = save_percentile_outputs(percentile_model_stats, model_name, current_model, X_train, X_test,
                                                     y_train, y_test, y_train_predicted, y_test_predicted, X_std, y, y_predicted)

    return percentile_model_stats, model_stats, y_train_predicted, y_test_predicted, y_predicted, current_model

def lin_reg_sigweight2(X_train, y_train, X_test, y_test, X_std, y, model_name, model_stats, percentile_model_stats, settings):
    from sklearn.linear_model import LinearRegression
    y_train_zscore = scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    weights = 1/(1+np.exp(-y_train_zscore+2))**2
    # Instantiate the LR model:
    current_model = LinearRegression()
    # train the model using training data
    current_model.fit(X_train, y_train, sample_weight=weights)

    # get the predicted y values for the model
    y_train_predicted = current_model.predict(X_train)
    y_test_predicted = current_model.predict(X_test)
    y_predicted = current_model.predict(X_std)

    model_stats = save_outputs(model_stats, model_name, current_model, X_train, X_test, y_train,y_test, y_train_predicted, y_test_predicted, X_std, y, y_predicted)
    percentile_model_stats = save_percentile_outputs(percentile_model_stats, model_name, current_model, X_train, X_test,
                                                     y_train, y_test, y_train_predicted, y_test_predicted, X_std, y, y_predicted)

    return percentile_model_stats, model_stats, y_train_predicted, y_test_predicted, y_predicted, current_model

def lin_reg_sigweight(X_train, y_train, X_test, y_test, X_std, y, model_name, model_stats, percentile_model_stats, settings):
    from sklearn.linear_model import LinearRegression
    y_train_zscore = scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    weights = 1/(1+np.exp(-y_train_zscore))**2
    # Instantiate the LR model:
    current_model = LinearRegression()
    # train the model using training data
    current_model.fit(X_train, y_train, sample_weight=weights)

    # get the predicted y values for the model
    y_train_predicted = current_model.predict(X_train)
    y_test_predicted = current_model.predict(X_test)
    y_predicted = current_model.predict(X_std)

    model_stats = save_outputs(model_stats, model_name, current_model, X_train, X_test, y_train,y_test, y_train_predicted, y_test_predicted, X_std, y, y_predicted)
    percentile_model_stats = save_percentile_outputs(percentile_model_stats, model_name, current_model, X_train, X_test,
                                                     y_train, y_test, y_train_predicted, y_test_predicted, X_std, y, y_predicted)

    return percentile_model_stats, model_stats, y_train_predicted, y_test_predicted, y_predicted, current_model

def lin_reg_sigweight0_5(X_train, y_train, X_test, y_test, X_std, y, model_name, model_stats, percentile_model_stats, settings):
    from sklearn.linear_model import LinearRegression
    y_train_zscore = scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    weights = 1/(1+np.exp(-y_train_zscore+0.5))**2
    # Instantiate the LR model:
    current_model = LinearRegression()
    # train the model using training data
    current_model.fit(X_train, y_train, sample_weight=weights)

    # get the predicted y values for the model
    y_train_predicted = current_model.predict(X_train)
    y_test_predicted = current_model.predict(X_test)
    y_predicted = current_model.predict(X_std)

    model_stats = save_outputs(model_stats, model_name, current_model, X_train, X_test, y_train,y_test, y_train_predicted, y_test_predicted, X_std, y, y_predicted)
    percentile_model_stats = save_percentile_outputs(percentile_model_stats, model_name, current_model, X_train, X_test,
                                                     y_train, y_test, y_train_predicted, y_test_predicted, X_std, y, y_predicted)

    return percentile_model_stats, model_stats, y_train_predicted, y_test_predicted, y_predicted, current_model

def lin_reg_sigweight1(X_train, y_train, X_test, y_test, X_std, y, model_name, model_stats, percentile_model_stats, settings):
    from sklearn.linear_model import LinearRegression
    y_train_zscore = scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    weights = 1/(1+np.exp(-y_train_zscore+1))**2
    # Instantiate the LR model:
    current_model = LinearRegression()
    # train the model using training data
    current_model.fit(X_train, y_train, sample_weight=weights)

    # get the predicted y values for the model
    y_train_predicted = current_model.predict(X_train)
    y_test_predicted = current_model.predict(X_test)
    y_predicted = current_model.predict(X_std)

    model_stats = save_outputs(model_stats, model_name, current_model, X_train, X_test, y_train,y_test, y_train_predicted, y_test_predicted, X_std, y, y_predicted)
    percentile_model_stats = save_percentile_outputs(percentile_model_stats, model_name, current_model, X_train, X_test,
                                                     y_train, y_test, y_train_predicted, y_test_predicted, X_std, y, y_predicted)

    return percentile_model_stats, model_stats, y_train_predicted, y_test_predicted, y_predicted, current_model

def lin_reg_sigweight1_5(X_train, y_train, X_test, y_test, X_std, y, model_name, model_stats, percentile_model_stats, settings):
    from sklearn.linear_model import LinearRegression
    y_train_zscore = scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    weights = 1/(1+np.exp(-y_train_zscore+1.5))**2
    # Instantiate the LR model:
    current_model = LinearRegression()
    # train the model using training data
    current_model.fit(X_train, y_train, sample_weight=weights)

    # get the predicted y values for the model
    y_train_predicted = current_model.predict(X_train)
    y_test_predicted = current_model.predict(X_test)
    y_predicted = current_model.predict(X_std)

    model_stats = save_outputs(model_stats, model_name, current_model, X_train, X_test, y_train,y_test, y_train_predicted, y_test_predicted, X_std, y, y_predicted)
    percentile_model_stats = save_percentile_outputs(percentile_model_stats, model_name, current_model, X_train, X_test,
                                                     y_train, y_test, y_train_predicted, y_test_predicted, X_std, y, y_predicted)

    return percentile_model_stats, model_stats, y_train_predicted, y_test_predicted, y_predicted, current_model

def lin_reg_sigweight2_5(X_train, y_train, X_test, y_test, X_std, y, model_name, model_stats, percentile_model_stats, settings):
    from sklearn.linear_model import LinearRegression
    y_train_zscore = scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    weights = 1/(1+np.exp(-y_train_zscore+2.5))**2
    # Instantiate the LR model:
    current_model = LinearRegression()
    # train the model using training data
    current_model.fit(X_train, y_train, sample_weight=weights)

    # get the predicted y values for the model
    y_train_predicted = current_model.predict(X_train)
    y_test_predicted = current_model.predict(X_test)
    y_predicted = current_model.predict(X_std)

    model_stats = save_outputs(model_stats, model_name, current_model, X_train, X_test, y_train,y_test, y_train_predicted, y_test_predicted, X_std, y, y_predicted)
    percentile_model_stats = save_percentile_outputs(percentile_model_stats, model_name, current_model, X_train, X_test,
                                                     y_train, y_test, y_train_predicted, y_test_predicted, X_std, y, y_predicted)

    return percentile_model_stats, model_stats, y_train_predicted, y_test_predicted, y_predicted, current_model

def lin_reg_sigweight3(X_train, y_train, X_test, y_test, X_std, y, model_name, model_stats, percentile_model_stats, settings):
    from sklearn.linear_model import LinearRegression
    y_train_zscore = scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    weights = 1/(1+np.exp(-y_train_zscore+3))**2
    # Instantiate the LR model:
    current_model = LinearRegression()
    # train the model using training data
    current_model.fit(X_train, y_train, sample_weight=weights)

    # get the predicted y values for the model
    y_train_predicted = current_model.predict(X_train)
    y_test_predicted = current_model.predict(X_test)
    y_predicted = current_model.predict(X_std)

    model_stats = save_outputs(model_stats, model_name, current_model, X_train, X_test, y_train,y_test, y_train_predicted, y_test_predicted, X_std, y, y_predicted)
    percentile_model_stats = save_percentile_outputs(percentile_model_stats, model_name, current_model, X_train, X_test,
                                                     y_train, y_test, y_train_predicted, y_test_predicted, X_std, y, y_predicted)

    return percentile_model_stats, model_stats, y_train_predicted, y_test_predicted, y_predicted, current_model

def xg_boost_sigweight(X_train, y_train, X_test, y_test, X_std, y, model_name, model_stats, percentile_model_stats, settings):
    y_train_zscore = scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    weights = 1/(1+np.exp(-y_train_zscore))**2
    # Create an XGBoost regressor object
    xgb_regressor = xgb.XGBRegressor()

    # Define the parameter grid for RandomizedSearchCV
    xgb_params = {
    'n_estimators': [50, 100, 300, 500, 800, 1000],
    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],  # Default is 0.3. Ranges from loc to loc+scale.
    'subsample': [0.3, 0.5, 0.7, 0.9, 1.0],
    'colsample_bytree': [0.5, 0.75, 1.0],
    "min_child_weight": [1, 3, 5, 7, 10],  # Default is 1
    "max_depth": [1, 3, 5, 7, 10],  # Default is 6
    "alpha": [0, 0.01, 1, 5, 10, 50, 100],  # Default is 0. AKA reg_alpha.
    "lambda": [0, 0.01, 1, 5, 10, 50, 100]  # Default is 0. AKA reg_lambda.
    }

    ''' #old params
    xgb_params = {
    'n_estimators': [50, 100, 300, 500, 800, 1000],
    "learning_rate": scipy.stats.uniform(loc=0.003, scale=0.19),  # Default is 0.3. Ranges from loc to loc+scale.
    #"subsample": scipy.stats.uniform(loc=0.5, scale=0.5),  # Default is 1
    #"colsample_bytree": scipy.stats.uniform(loc=0.5, scale=0.5),  # Default is 1
    "min_child_weight": [1, 3, 5, 7,10],  # Default is 1
    "max_depth": np.append(0, np.arange(1, 10)),  # Default is 6
    "alpha": [0, 0.01, 1, 5, 10, 50, 100],  # Default is 0. AKA reg_alpha.
    "lambda": [0, 0.01, 1, 5, 10, 50, 100]  # Default is 0. AKA reg_lambda.
    }'''


    # Create folds that are 5 chunks without shuffling the data
    kf = KFold(n_splits=5, shuffle=False)

    # Create a RandomizedSearchCV object
    xgb_cv = RandomizedSearchCV(
        estimator=xgb_regressor,
        param_distributions=xgb_params,
        n_iter=200,  # Number of parameter settings that are sampled
        scoring='neg_mean_squared_error',  # Use negative MSE as the scoring metric
        cv=kf,  # Number of cross-validation folds
        random_state=42,  # Set random state for reproducibility
        n_jobs=-1,  # Use all available cores for parallel processing
    )

    # train the cross validation models
    xgb_cv.fit(X_train, y_train, sample_weight=weights)
    # get the best parameters
    best_params = xgb_cv.best_params_
    np.random.seed(42)

    # train a new model with the best parameters
    current_model = xgb.XGBRegressor(**best_params)
    current_model.fit(X_train, y_train, sample_weight=weights)
    # get the predicted y values for the model
    y_train_predicted = current_model.predict(X_train)
    y_test_predicted = current_model.predict(X_test)
    y_predicted = current_model.predict(X_std)

    # save the model statistics
    model_stats = save_outputs(model_stats, model_name, current_model, X_train, X_test, y_train, y_test,
                               y_train_predicted, y_test_predicted, X_std, y, y_predicted)
    percentile_model_stats = save_percentile_outputs(percentile_model_stats, model_name, current_model, X_train, X_test,
                                                     y_train, y_test, y_train_predicted, y_test_predicted, X_std, y,
                                                     y_predicted)

    return percentile_model_stats, model_stats, y_train_predicted, y_test_predicted, y_predicted, current_model

def xg_boost_sigweight0_5(X_train, y_train, X_test, y_test, X_std, y, model_name, model_stats, percentile_model_stats, settings):
    y_train_zscore = scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    weights = 1/(1+np.exp(-y_train_zscore+0.5))**2

    # Create an XGBoost regressor object
    xgb_regressor = xgb.XGBRegressor()

    # Define the parameter grid for RandomizedSearchCV
    xgb_params = {
    'n_estimators': [50, 100, 300, 500, 800, 1000],
    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],  # Default is 0.3. Ranges from loc to loc+scale.
    'subsample': [0.3, 0.5, 0.7, 0.9, 1.0],
    'colsample_bytree': [0.5, 0.75, 1.0],
    "min_child_weight": [1, 3, 5, 7, 10],  # Default is 1
    "max_depth": [1, 3, 5, 7, 10],  # Default is 6
    "alpha": [0, 0.01, 1, 5, 10, 50, 100],  # Default is 0. AKA reg_alpha.
    "lambda": [0, 0.01, 1, 5, 10, 50, 100]  # Default is 0. AKA reg_lambda.
    }

    ''' #old params
    xgb_params = {
    'n_estimators': [50, 100, 300, 500, 800, 1000],
    "learning_rate": scipy.stats.uniform(loc=0.003, scale=0.19),  # Default is 0.3. Ranges from loc to loc+scale.
    #"subsample": scipy.stats.uniform(loc=0.5, scale=0.5),  # Default is 1
    #"colsample_bytree": scipy.stats.uniform(loc=0.5, scale=0.5),  # Default is 1
    "min_child_weight": [1, 3, 5, 7,10],  # Default is 1
    "max_depth": np.append(0, np.arange(1, 10)),  # Default is 6
    "alpha": [0, 0.01, 1, 5, 10, 50, 100],  # Default is 0. AKA reg_alpha.
    "lambda": [0, 0.01, 1, 5, 10, 50, 100]  # Default is 0. AKA reg_lambda.
    }'''


    # Create folds that are 5 chunks without shuffling the data
    kf = KFold(n_splits=5, shuffle=False)

    # Create a RandomizedSearchCV object
    xgb_cv = RandomizedSearchCV(
        estimator=xgb_regressor,
        param_distributions=xgb_params,
        n_iter=200,  # Number of parameter settings that are sampled
        scoring='neg_mean_squared_error',  # Use negative MSE as the scoring metric
        cv=kf,  # Number of cross-validation folds
        random_state=42,  # Set random state for reproducibility
        n_jobs=-1,  # Use all available cores for parallel processing
    )

    # train the cross validation models
    xgb_cv.fit(X_train, y_train, sample_weight=weights)
    # get the best parameters
    best_params = xgb_cv.best_params_
    np.random.seed(42)

    # train a new model with the best parameters
    current_model = xgb.XGBRegressor(**best_params)
    current_model.fit(X_train, y_train, sample_weight=weights)
    # get the predicted y values for the model
    y_train_predicted = current_model.predict(X_train)
    y_test_predicted = current_model.predict(X_test)
    y_predicted = current_model.predict(X_std)

    # save the model statistics
    model_stats = save_outputs(model_stats, model_name, current_model, X_train, X_test, y_train, y_test,
                               y_train_predicted, y_test_predicted, X_std, y, y_predicted)
    percentile_model_stats = save_percentile_outputs(percentile_model_stats, model_name, current_model, X_train, X_test,
                                                     y_train, y_test, y_train_predicted, y_test_predicted, X_std, y,
                                                     y_predicted)

    return percentile_model_stats, model_stats, y_train_predicted, y_test_predicted, y_predicted, current_model


def xg_boost_sigweight1(X_train, y_train, X_test, y_test, X_std, y, model_name, model_stats, percentile_model_stats,
                          settings):
    y_train_zscore = scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    weights = 1 / (1 + np.exp(-y_train_zscore + 1)) ** 2

    # Create an XGBoost regressor object
    xgb_regressor = xgb.XGBRegressor()

    # Define the parameter grid for RandomizedSearchCV
    xgb_params = {
        'n_estimators': [50, 100, 300, 500, 800, 1000],
        "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],  # Default is 0.3. Ranges from loc to loc+scale.
        'subsample': [0.3, 0.5, 0.7, 0.9, 1.0],
        'colsample_bytree': [0.5, 0.75, 1.0],
        "min_child_weight": [1, 3, 5, 7, 10],  # Default is 1
        "max_depth": [1, 3, 5, 7, 10],  # Default is 6
        "alpha": [0, 0.01, 1, 5, 10, 50, 100],  # Default is 0. AKA reg_alpha.
        "lambda": [0, 0.01, 1, 5, 10, 50, 100]  # Default is 0. AKA reg_lambda.
    }

    ''' #old params
    xgb_params = {
    'n_estimators': [50, 100, 300, 500, 800, 1000],
    "learning_rate": scipy.stats.uniform(loc=0.003, scale=0.19),  # Default is 0.3. Ranges from loc to loc+scale.
    #"subsample": scipy.stats.uniform(loc=0.5, scale=0.5),  # Default is 1
    #"colsample_bytree": scipy.stats.uniform(loc=0.5, scale=0.5),  # Default is 1
    "min_child_weight": [1, 3, 5, 7,10],  # Default is 1
    "max_depth": np.append(0, np.arange(1, 10)),  # Default is 6
    "alpha": [0, 0.01, 1, 5, 10, 50, 100],  # Default is 0. AKA reg_alpha.
    "lambda": [0, 0.01, 1, 5, 10, 50, 100]  # Default is 0. AKA reg_lambda.
    }'''

    # Create folds that are 5 chunks without shuffling the data
    kf = KFold(n_splits=5, shuffle=False)

    # Create a RandomizedSearchCV object
    xgb_cv = RandomizedSearchCV(
        estimator=xgb_regressor,
        param_distributions=xgb_params,
        n_iter=200,  # Number of parameter settings that are sampled
        scoring='neg_mean_squared_error',  # Use negative MSE as the scoring metric
        cv=kf,  # Number of cross-validation folds
        random_state=42,  # Set random state for reproducibility
        n_jobs=-1,  # Use all available cores for parallel processing
    )

    # train the cross validation models
    xgb_cv.fit(X_train, y_train, sample_weight=weights)
    # get the best parameters
    best_params = xgb_cv.best_params_
    np.random.seed(42)

    # train a new model with the best parameters
    current_model = xgb.XGBRegressor(**best_params)
    current_model.fit(X_train, y_train, sample_weight=weights)
    # get the predicted y values for the model
    y_train_predicted = current_model.predict(X_train)
    y_test_predicted = current_model.predict(X_test)
    y_predicted = current_model.predict(X_std)

    # save the model statistics
    model_stats = save_outputs(model_stats, model_name, current_model, X_train, X_test, y_train, y_test,
                               y_train_predicted, y_test_predicted, X_std, y, y_predicted)
    percentile_model_stats = save_percentile_outputs(percentile_model_stats, model_name, current_model, X_train, X_test,
                                                     y_train, y_test, y_train_predicted, y_test_predicted, X_std, y,
                                                     y_predicted)

    return percentile_model_stats, model_stats, y_train_predicted, y_test_predicted, y_predicted, current_model

def xg_boost_sigweight1_5(X_train, y_train, X_test, y_test, X_std, y, model_name, model_stats, percentile_model_stats,
                          settings):
    y_train_zscore = scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    weights = 1 / (1 + np.exp(-y_train_zscore + 1.5)) ** 2

    # Create an XGBoost regressor object
    xgb_regressor = xgb.XGBRegressor()

    # Define the parameter grid for RandomizedSearchCV
    xgb_params = {
        'n_estimators': [50, 100, 300, 500, 800, 1000],
        "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],  # Default is 0.3. Ranges from loc to loc+scale.
        'subsample': [0.3, 0.5, 0.7, 0.9, 1.0],
        'colsample_bytree': [0.5, 0.75, 1.0],
        "min_child_weight": [1, 3, 5, 7, 10],  # Default is 1
        "max_depth": [1, 3, 5, 7, 10],  # Default is 6
        "alpha": [0, 0.01, 1, 5, 10, 50, 100],  # Default is 0. AKA reg_alpha.
        "lambda": [0, 0.01, 1, 5, 10, 50, 100]  # Default is 0. AKA reg_lambda.
    }

    ''' #old params
    xgb_params = {
    'n_estimators': [50, 100, 300, 500, 800, 1000],
    "learning_rate": scipy.stats.uniform(loc=0.003, scale=0.19),  # Default is 0.3. Ranges from loc to loc+scale.
    #"subsample": scipy.stats.uniform(loc=0.5, scale=0.5),  # Default is 1
    #"colsample_bytree": scipy.stats.uniform(loc=0.5, scale=0.5),  # Default is 1
    "min_child_weight": [1, 3, 5, 7,10],  # Default is 1
    "max_depth": np.append(0, np.arange(1, 10)),  # Default is 6
    "alpha": [0, 0.01, 1, 5, 10, 50, 100],  # Default is 0. AKA reg_alpha.
    "lambda": [0, 0.01, 1, 5, 10, 50, 100]  # Default is 0. AKA reg_lambda.
    }'''

    # Create folds that are 5 chunks without shuffling the data
    kf = KFold(n_splits=5, shuffle=False)

    # Create a RandomizedSearchCV object
    xgb_cv = RandomizedSearchCV(
        estimator=xgb_regressor,
        param_distributions=xgb_params,
        n_iter=200,  # Number of parameter settings that are sampled
        scoring='neg_mean_squared_error',  # Use negative MSE as the scoring metric
        cv=kf,  # Number of cross-validation folds
        random_state=42,  # Set random state for reproducibility
        n_jobs=-1,  # Use all available cores for parallel processing
    )

    # train the cross validation models
    xgb_cv.fit(X_train, y_train, sample_weight=weights)
    # get the best parameters
    best_params = xgb_cv.best_params_
    np.random.seed(42)

    # train a new model with the best parameters
    current_model = xgb.XGBRegressor(**best_params)
    current_model.fit(X_train, y_train, sample_weight=weights)
    # get the predicted y values for the model
    y_train_predicted = current_model.predict(X_train)
    y_test_predicted = current_model.predict(X_test)
    y_predicted = current_model.predict(X_std)

    # save the model statistics
    model_stats = save_outputs(model_stats, model_name, current_model, X_train, X_test, y_train, y_test,
                               y_train_predicted, y_test_predicted, X_std, y, y_predicted)
    percentile_model_stats = save_percentile_outputs(percentile_model_stats, model_name, current_model, X_train, X_test,
                                                     y_train, y_test, y_train_predicted, y_test_predicted, X_std, y,
                                                     y_predicted)

    return percentile_model_stats, model_stats, y_train_predicted, y_test_predicted, y_predicted, current_model

def xg_boost_sigweight2(X_train, y_train, X_test, y_test, X_std, y, model_name, model_stats, percentile_model_stats,
                          settings):
    y_train_zscore = scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    weights = 1 / (1 + np.exp(-y_train_zscore + 2)) ** 2

    # Create an XGBoost regressor object
    xgb_regressor = xgb.XGBRegressor()

    # Define the parameter grid for RandomizedSearchCV
    xgb_params = {
        'n_estimators': [50, 100, 300, 500, 800, 1000],
        "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],  # Default is 0.3. Ranges from loc to loc+scale.
        'subsample': [0.3, 0.5, 0.7, 0.9, 1.0],
        'colsample_bytree': [0.5, 0.75, 1.0],
        "min_child_weight": [1, 3, 5, 7, 10],  # Default is 1
        "max_depth": [1, 3, 5, 7, 10],  # Default is 6
        "alpha": [0, 0.01, 1, 5, 10, 50, 100],  # Default is 0. AKA reg_alpha.
        "lambda": [0, 0.01, 1, 5, 10, 50, 100]  # Default is 0. AKA reg_lambda.
    }

    ''' #old params
    xgb_params = {
    'n_estimators': [50, 100, 300, 500, 800, 1000],
    "learning_rate": scipy.stats.uniform(loc=0.003, scale=0.19),  # Default is 0.3. Ranges from loc to loc+scale.
    #"subsample": scipy.stats.uniform(loc=0.5, scale=0.5),  # Default is 1
    #"colsample_bytree": scipy.stats.uniform(loc=0.5, scale=0.5),  # Default is 1
    "min_child_weight": [1, 3, 5, 7,10],  # Default is 1
    "max_depth": np.append(0, np.arange(1, 10)),  # Default is 6
    "alpha": [0, 0.01, 1, 5, 10, 50, 100],  # Default is 0. AKA reg_alpha.
    "lambda": [0, 0.01, 1, 5, 10, 50, 100]  # Default is 0. AKA reg_lambda.
    }'''

    # Create folds that are 5 chunks without shuffling the data
    kf = KFold(n_splits=5, shuffle=False)

    # Create a RandomizedSearchCV object
    xgb_cv = RandomizedSearchCV(
        estimator=xgb_regressor,
        param_distributions=xgb_params,
        n_iter=200,  # Number of parameter settings that are sampled
        scoring='neg_mean_squared_error',  # Use negative MSE as the scoring metric
        cv=kf,  # Number of cross-validation folds
        random_state=42,  # Set random state for reproducibility
        n_jobs=-1,  # Use all available cores for parallel processing
    )

    # train the cross validation models
    xgb_cv.fit(X_train, y_train, sample_weight=weights)
    # get the best parameters
    best_params = xgb_cv.best_params_
    np.random.seed(42)

    # train a new model with the best parameters
    current_model = xgb.XGBRegressor(**best_params)
    current_model.fit(X_train, y_train, sample_weight=weights)
    # get the predicted y values for the model
    y_train_predicted = current_model.predict(X_train)
    y_test_predicted = current_model.predict(X_test)
    y_predicted = current_model.predict(X_std)

    # save the model statistics
    model_stats = save_outputs(model_stats, model_name, current_model, X_train, X_test, y_train, y_test,
                               y_train_predicted, y_test_predicted, X_std, y, y_predicted)
    percentile_model_stats = save_percentile_outputs(percentile_model_stats, model_name, current_model, X_train, X_test,
                                                     y_train, y_test, y_train_predicted, y_test_predicted, X_std, y,
                                                     y_predicted)

    return percentile_model_stats, model_stats, y_train_predicted, y_test_predicted, y_predicted, current_model

def xg_boost_sigweight2_5(X_train, y_train, X_test, y_test, X_std, y, model_name, model_stats, percentile_model_stats,
                          settings):
    y_train_zscore = scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    weights = 1 / (1 + np.exp(-y_train_zscore + 2.5)) ** 2

    # Create an XGBoost regressor object
    xgb_regressor = xgb.XGBRegressor()

    # Define the parameter grid for RandomizedSearchCV
    xgb_params = {
        'n_estimators': [50, 100, 300, 500, 800, 1000],
        "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],  # Default is 0.3. Ranges from loc to loc+scale.
        'subsample': [0.3, 0.5, 0.7, 0.9, 1.0],
        'colsample_bytree': [0.5, 0.75, 1.0],
        "min_child_weight": [1, 3, 5, 7, 10],  # Default is 1
        "max_depth": [1, 3, 5, 7, 10],  # Default is 6
        "alpha": [0, 0.01, 1, 5, 10, 50, 100],  # Default is 0. AKA reg_alpha.
        "lambda": [0, 0.01, 1, 5, 10, 50, 100]  # Default is 0. AKA reg_lambda.
    }

    ''' #old params
    xgb_params = {
    'n_estimators': [50, 100, 300, 500, 800, 1000],
    "learning_rate": scipy.stats.uniform(loc=0.003, scale=0.19),  # Default is 0.3. Ranges from loc to loc+scale.
    #"subsample": scipy.stats.uniform(loc=0.5, scale=0.5),  # Default is 1
    #"colsample_bytree": scipy.stats.uniform(loc=0.5, scale=0.5),  # Default is 1
    "min_child_weight": [1, 3, 5, 7,10],  # Default is 1
    "max_depth": np.append(0, np.arange(1, 10)),  # Default is 6
    "alpha": [0, 0.01, 1, 5, 10, 50, 100],  # Default is 0. AKA reg_alpha.
    "lambda": [0, 0.01, 1, 5, 10, 50, 100]  # Default is 0. AKA reg_lambda.
    }'''

    # Create folds that are 5 chunks without shuffling the data
    kf = KFold(n_splits=5, shuffle=False)

    # Create a RandomizedSearchCV object
    xgb_cv = RandomizedSearchCV(
        estimator=xgb_regressor,
        param_distributions=xgb_params,
        n_iter=200,  # Number of parameter settings that are sampled
        scoring='neg_mean_squared_error',  # Use negative MSE as the scoring metric
        cv=kf,  # Number of cross-validation folds
        random_state=42,  # Set random state for reproducibility
        n_jobs=-1,  # Use all available cores for parallel processing
    )

    # train the cross validation models
    xgb_cv.fit(X_train, y_train, sample_weight=weights)
    # get the best parameters
    best_params = xgb_cv.best_params_
    np.random.seed(42)

    # train a new model with the best parameters
    current_model = xgb.XGBRegressor(**best_params)
    current_model.fit(X_train, y_train, sample_weight=weights)
    # get the predicted y values for the model
    y_train_predicted = current_model.predict(X_train)
    y_test_predicted = current_model.predict(X_test)
    y_predicted = current_model.predict(X_std)

    # save the model statistics
    model_stats = save_outputs(model_stats, model_name, current_model, X_train, X_test, y_train, y_test,
                               y_train_predicted, y_test_predicted, X_std, y, y_predicted)
    percentile_model_stats = save_percentile_outputs(percentile_model_stats, model_name, current_model, X_train, X_test,
                                                     y_train, y_test, y_train_predicted, y_test_predicted, X_std, y,
                                                     y_predicted)

    return percentile_model_stats, model_stats, y_train_predicted, y_test_predicted, y_predicted, current_model

def xg_boost_sigweight3(X_train, y_train, X_test, y_test, X_std, y, model_name, model_stats, percentile_model_stats,
                          settings):
    y_train_zscore = scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    weights = 1 / (1 + np.exp(-y_train_zscore + 3)) ** 2

    # Create an XGBoost regressor object
    xgb_regressor = xgb.XGBRegressor()

    # Define the parameter grid for RandomizedSearchCV
    xgb_params = {
        'n_estimators': [50, 100, 300, 500, 800, 1000],
        "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],  # Default is 0.3. Ranges from loc to loc+scale.
        'subsample': [0.3, 0.5, 0.7, 0.9, 1.0],
        'colsample_bytree': [0.5, 0.75, 1.0],
        "min_child_weight": [1, 3, 5, 7, 10],  # Default is 1
        "max_depth": [1, 3, 5, 7, 10],  # Default is 6
        "alpha": [0, 0.01, 1, 5, 10, 50, 100],  # Default is 0. AKA reg_alpha.
        "lambda": [0, 0.01, 1, 5, 10, 50, 100]  # Default is 0. AKA reg_lambda.
    }

    ''' #old params
    xgb_params = {
    'n_estimators': [50, 100, 300, 500, 800, 1000],
    "learning_rate": scipy.stats.uniform(loc=0.003, scale=0.19),  # Default is 0.3. Ranges from loc to loc+scale.
    #"subsample": scipy.stats.uniform(loc=0.5, scale=0.5),  # Default is 1
    #"colsample_bytree": scipy.stats.uniform(loc=0.5, scale=0.5),  # Default is 1
    "min_child_weight": [1, 3, 5, 7,10],  # Default is 1
    "max_depth": np.append(0, np.arange(1, 10)),  # Default is 6
    "alpha": [0, 0.01, 1, 5, 10, 50, 100],  # Default is 0. AKA reg_alpha.
    "lambda": [0, 0.01, 1, 5, 10, 50, 100]  # Default is 0. AKA reg_lambda.
    }'''

    # Create folds that are 5 chunks without shuffling the data
    kf = KFold(n_splits=5, shuffle=False)

    # Create a RandomizedSearchCV object
    xgb_cv = RandomizedSearchCV(
        estimator=xgb_regressor,
        param_distributions=xgb_params,
        n_iter=200,  # Number of parameter settings that are sampled
        scoring='neg_mean_squared_error',  # Use negative MSE as the scoring metric
        cv=kf,  # Number of cross-validation folds
        random_state=42,  # Set random state for reproducibility
        n_jobs=-1,  # Use all available cores for parallel processing
    )

    # train the cross validation models
    xgb_cv.fit(X_train, y_train, sample_weight=weights)
    # get the best parameters
    best_params = xgb_cv.best_params_
    np.random.seed(42)

    # train a new model with the best parameters
    current_model = xgb.XGBRegressor(**best_params)
    current_model.fit(X_train, y_train, sample_weight=weights)
    # get the predicted y values for the model
    y_train_predicted = current_model.predict(X_train)
    y_test_predicted = current_model.predict(X_test)
    y_predicted = current_model.predict(X_std)

    # save the model statistics
    model_stats = save_outputs(model_stats, model_name, current_model, X_train, X_test, y_train, y_test,
                               y_train_predicted, y_test_predicted, X_std, y, y_predicted)
    percentile_model_stats = save_percentile_outputs(percentile_model_stats, model_name, current_model, X_train, X_test,
                                                     y_train, y_test, y_train_predicted, y_test_predicted, X_std, y,
                                                     y_predicted)

    return percentile_model_stats, model_stats, y_train_predicted, y_test_predicted, y_predicted, current_model

