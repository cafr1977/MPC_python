# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 13:24:51 2023

@author: cfris
"""
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.linear_model import Lasso
from sklearn.model_selection import RandomizedSearchCV, KFold


def lasso(X_train, y_train, X_test, y_test, X_std, model_name,pollutant,output_folder_name,model_stats):
    lasso_params_grid = {'alpha':np.arange(0.001, 10, 0.05)}
    lasso_model = Lasso()
    
    # Create folds that are 5 chunks without shuffling the data
    kf = KFold(n_splits=5, shuffle=False)
    
    #cross validation to find the best parameter (alpha)
    lasso_cv =  RandomizedSearchCV(lasso_model, lasso_params_grid, cv=kf, scoring ='neg_root_mean_squared_error', random_state=42)
    
    #train the cross validation models
    lasso_cv.fit(X_train,y_train)
    #get the best parameters
    best_params=lasso_cv.best_params_
   
    #train a new model with the best parameters
    current_model = Lasso(**best_params)
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