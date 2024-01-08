# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 12:29:59 2023

@author: cfris
"""

import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold

def random_forest(X_train, y_train, X_test, y_test, X_std, model_name,pollutant,output_folder_name,model_stats):
    rf_regressor = RandomForestRegressor(random_state=42)
    rf_params = {'max_depth':np.arange(1,10,1),
             'min_samples_split':np.arange(2,50,1),
             'min_samples_leaf':np.arange(2,50,1),
             'max_features':['sqrt', 'log2']}
    # Create folds that are 5 chunks without shuffling the data
    kf = KFold(n_splits=5, shuffle=False)
    
    #cross validation to find the best parameter (see rf_params)
    rf_regressor_cv = RandomizedSearchCV(rf_regressor, rf_params, cv=kf, scoring='neg_root_mean_squared_error', random_state=42)
    #train the cross validation models
    rf_regressor_cv.fit(X_train, y_train)
    #get the best parameters
    best_params = rf_regressor_cv.best_params_
    np.random.seed(42)
    
    #train a new model with the best parameters
    current_model = RandomForestRegressor(**best_params)
    current_model.fit(X_train, y_train)
    
    #get the predicted y values for the model
    y_train_predicted = current_model.predict(X_train)
    y_test_predicted = current_model.predict(X_test)
    y_predicted = current_model.predict(X_std)
     
    #save the model statistics
    model_stats['Training_R2'][model_name] = round(current_model.score(X_train, y_train),2)
    model_stats['Training_RMSE'][model_name] = (np.sqrt(mean_squared_error(y_train, y_train_predicted)))
    model_stats['Testing_RMSE'][model_name] = (np.sqrt(mean_squared_error(y_test, y_test_predicted)))
    model_stats['Training_MBE'][model_name] = np.mean(y_train_predicted - y_train)
    model_stats['Testing_MBE'][model_name] = np.mean(y_test_predicted - y_test)
   
    
    return model_stats, y_train_predicted, y_test_predicted, y_predicted, current_model