import numpy as np
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from Python_Functions import colo_model_func
from sklearn.svm import SVR

def weights_set(y_train, percentile, weight):
    # Calculate the target variable values for training set
    y_train_values = y_train.values

    # Calculate the quantiles
    peaks = np.percentile(y_train_values, percentile)

    # Create weights based on quartiles
    #anything in the 95th percentile or above is weighted 8
    #everything below that is weighted 1
    weights = np.where(y_train_values >= peaks, weight, 1.0)
    return weights

def rf_qw_tuned(X_train, y_train, X_test, y_test, X_std, percentile, weight, model_name, model_stats):
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
    model_stats = colo_model_func.save_outputs(model_stats, model_name, current_model, X_train, X_test, y_train, y_test, y_train_predicted,
                               y_test_predicted)

    return model_stats, y_train_predicted, y_test_predicted, y_predicted, current_model


def svr_qw_tuned(X_train, y_train, X_test, y_test, X_std, percentile, weight, model_name, model_stats):
    weights = weights_set(y_train, percentile, weight)

    svr_regressor = SVR()
    svr_params = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                  'gamma': ['scale', 'auto'],
                  'C': np.arange(0.1, 5, 0.4)}

    # Create folds that are 5 chunks without shuffling the data
    kf = KFold(n_splits=5, shuffle=False)

    # cross validation to find the best parameter (see rf_params)
    svr_cv = RandomizedSearchCV(svr_regressor, svr_params, random_state=42, cv=kf,
                                scoring='neg_root_mean_squared_error')

    # train the cross validation models
    svr_cv.fit(X_train, y_train, sample_weight=weights)

    # get the best parameters
    best_params = svr_cv.best_params_
    np.random.seed(42)

    # train a new model with the best parameters
    current_model = SVR(**best_params)
    current_model.fit(X_train, y_train, sample_weight=weights)

    # get the predicted y values for the model
    y_train_predicted = current_model.predict(X_train)
    y_test_predicted = current_model.predict(X_test)
    y_predicted = current_model.predict(X_std)

    # save the model statistics
    model_stats = colo_model_func.save_outputs(model_stats, model_name, current_model, X_train, X_test, y_train, y_test, y_train_predicted,
                               y_test_predicted)

    return model_stats, y_train_predicted, y_test_predicted, y_predicted, current_model
