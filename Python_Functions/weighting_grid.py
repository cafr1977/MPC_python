

 def quantile_weights(y_train, percentile, weight):
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
    weights = quantile_weights(y_train, percentile, weight)

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
    current_model.fit(X_train, y_train)

    # get the predicted y values for the model
    y_train_predicted = current_model.predict(X_train)
    y_test_predicted = current_model.predict(X_test)
    y_predicted = current_model.predict(X_std)

    # save the model statistics
    model_stats = save_outputs(model_stats, model_name, current_model, X_train, y_train, y_test, y_train_predicted,
                               y_test_predicted)

    return model_stats, y_train_predicted, y_test_predicted, y_predicted, current_model


# first, establish a dataframe to save model statistics in
 model_stats = pd.DataFrame(columns=['Training_R2', 'Training_RMSE', 'Testing_RMSE', 'Training_MBE', 'Testing_MBE'])

for p in percentile:
    for w in weight:
        p_str = str(p)
        w_str = str(w)
        model_name = f'rf_p_{p_str}_w_{w_str}'
        # Appending an empty row to model stats with the model name index
        model_stats.loc[model_name] = [None] * len(df.columns)
        print(f'Fitting colocation pod data to reference data using random forest with p= {p_str} and w= {w_str}...')
        model_stats, y_train_predicted, y_test_predicted, y_predicted, current_model= rf_qw_tuned(X_train, y_train, X_test, y_test, X_std, percentile, weight, model_name, model_stats)

         # save out the model and the y predicted
         y_predicted = pd.DataFrame(data=y_predicted, columns=[settings['pollutant']], index=X_std.index)
         y_predicted.to_csv(os.path.join('Outputs', output_folder_name, f'{model_name}_colo_y_predicted.csv'))
         joblib.dump(current_model, os.path.join('Outputs', output_folder_name, f'{model_name}_model.joblib'))

         # plotting of modelled data
         if 'colo_timeseries' in settings['colo_plot_list']:
             plotting_func.colo_timeseries(y_train, y_train_predicted, y_test, y_test_predicted, settings['pollutant'],
                                           model_name, output_folder_name, settings['colo_run_name'])

         if 'colo_scatter' in settings['colo_plot_list']:
             plotting_func.colo_scatter(y_train, y_train_predicted, y_test, y_test_predicted, settings['pollutant'],
                                        model_name, output_folder_name, settings['colo_run_name'])

         if 'colo_residual' in settings['colo_plot_list']:
             plotting_func.colo_residual(y_train, y_train_predicted, y_test, y_test_predicted, X_train, X_test,
                                         settings['pollutant'], model_name, output_folder_name, X_std.columns,
                                         settings['colo_run_name'])

 # save out the model for later analysis and use in field data
 model_stats.to_csv(os.path.join('Outputs', output_folder_name, 'colo_model_stats.csv'), index=True)

settings['models']=list(model_stats.index)

 # stats_plot plots the R2, RMSE, and MBE of train and test data as a bar graph
 if "colo_stats_plot" in settings['colo_plot_list']:
     plotting_func.colo_stats_plot(settings['models'], model_stats, settings['pollutant'], output_folder_name,
                                   settings['colo_run_name'])

 # save out settings for future reference
 joblib.dump(settings, os.path.join('Outputs', output_folder_name, 'run_settings.joblib'))