import numpy as np
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from Python_Functions import colo_model_func
from Python_Functions import test_train_split_func
from sklearn.linear_model import LinearRegression
import xgboost as xgb

def weights_set(y_train, percentile, weight):
    # Calculate the target variable values for training set
    y_train_values = y_train.values

    # Calculate the quantiles
    peaks = np.percentile(y_train_values, percentile)

    # Create weights based on quartiles
    #anything in the 95th percentile or below is weighted with weight
    #everything above that is weighted 1
    weights = np.where(y_train_values >= peaks, 1.0, weight)
    return weights

def rf_pieceweight(X_train, y_train, X_test, y_test, X_std, y, percentile, weight, model_name, model_stats, percentile_model_stats):
    weights = weights_set(y_train, percentile, weight)

    rf_regressor = RandomForestRegressor(random_state=42)
    rf_params = {'max_depth': np.arange(1, 10, 1),
                 'min_samples_split': np.arange(2, 50, 1),
                 'min_samples_leaf': np.arange(2, 50, 1),
                 'max_features': ['sqrt', 'log2']}
    # Create folds that are 5 chunks without shuffling the data
    kf = KFold(n_splits=5, shuffle=False)

    # cross validation to find the best parameter (see rf_params)
    rf_regressor_cv = RandomizedSearchCV(rf_regressor, rf_params, cv=kf, scoring='neg_root_mean_squared_error',
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
    X_std_values = X_std.values
    model_stats = colo_model_func.save_outputs(model_stats, model_name, current_model, X_train, X_test, y_train, y_test,
                                               y_train_predicted, y_test_predicted, X_std_values, y, y_predicted)
    percentile_model_stats = colo_model_func.save_percentile_outputs(percentile_model_stats, model_name, current_model,
                                                                     X_train, X_test, y_train, y_test,
                                                                     y_train_predicted, y_test_predicted, X_std_values,
                                                                     y, y_predicted)

    return percentile_model_stats, model_stats, y_train_predicted, y_test_predicted, y_predicted, current_model

def lin_reg_pieceweight(X_train, y_train, X_test, y_test, X_std, y, percentile, weight, model_name, model_stats, percentile_model_stats):
    weights = weights_set(y_train, percentile, weight)

    # Instantiate the LR model:
    current_model = LinearRegression()
    # train the model using training data
    current_model.fit(X_train, y_train, sample_weight=weights)

    # get the predicted y values for the model
    y_train_predicted = current_model.predict(X_train)
    y_test_predicted = current_model.predict(X_test)
    y_predicted = current_model.predict(X_std)

    X_std_values = X_std.values
    model_stats = colo_model_func.save_outputs(model_stats, model_name, current_model, X_train, X_test, y_train,y_test, y_train_predicted, y_test_predicted, X_std_values, y, y_predicted)
    percentile_model_stats = colo_model_func.save_percentile_outputs(percentile_model_stats, model_name, current_model, X_train, X_test, y_train,y_test, y_train_predicted, y_test_predicted, X_std_values, y, y_predicted)

    return percentile_model_stats, model_stats, y_train_predicted, y_test_predicted, y_predicted, current_model

def xgb_pieceweight(X_train, y_train, X_test, y_test, X_std, y, percentile, weight, model_name, model_stats, percentile_model_stats):
    weights = weights_set(y_train, percentile, weight)

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
    model_stats = colo_model_func.save_outputs(model_stats, model_name, current_model, X_train, X_test, y_train, y_test,
                               y_train_predicted, y_test_predicted, X_std, y, y_predicted)
    percentile_model_stats = colo_model_func.save_percentile_outputs(percentile_model_stats, model_name, current_model, X_train, X_test,
                                                     y_train, y_test, y_train_predicted, y_test_predicted, X_std, y,
                                                     y_predicted)

    return percentile_model_stats, model_stats, y_train_predicted, y_test_predicted, y_predicted, current_model
