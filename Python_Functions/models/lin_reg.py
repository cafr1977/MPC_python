import numpy as np
from sklearn.metrics import mean_squared_error

def lin_reg(X_train, y_train, X_test, y_test, X_std, model_name,pollutant,output_folder_name,model_stats):
    from sklearn.linear_model import LinearRegression
    #Instantiate the LR model:
    current_model= LinearRegression()
    #train the model using training data
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