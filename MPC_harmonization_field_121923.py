# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 16:51:02 2023

@author: cfris
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#packages from python_functions folder
from Python_Functions import preprocessing_func
from Python_Functions import data_loading_func
from Python_Functions import plotting_func

#initialize a dictionary (do not edit this line)
hf_set = {}

####
#edit the variables in this section for the harmonization - field run
colo_output_folder = 'Output_Oct2Dec_Reference_CH4_RFQWTuned' #the code will pull the best_model from this, and also save new stuff into it
hf_set['run_field'] = True    #True if you want to apply calibration to field data,
                                    #False if you only want to look at harmonization data
hf_set['best_model'] = 'rf_p_95_w_10'    #model that you would like to apply to field data from the output_folder
hf_set['k_folds'] = 5   #number of folds to split the data into for the k-fold cross validation, usually 5 or 10
hf_set['field_plot_list'] = ['field_boxplot','field_timeseries'] #field plots
hf_set['harmon_plot_list'] = [] #harmonization plots
####

###############
# Check if the output folder exists
if not os.path.exists(os.path.join('Outputs', colo_output_folder)):
    raise FileNotFoundError("The colocation output folder does not exist. Please double-check 'colo_output_folder'.")
#load run settings
settings = joblib.load(os.path.join('Outputs', colo_output_folder, 'run_settings.joblib')) #do not change this line!

settings = {**settings, **hf_set}

#close previous figures
plt.close('all')


#Create harmonization_field output folder
# Get the current time as YYMMDDHHss
current_time = datetime.now().strftime('%y%m%d%H%M%S')
# Create the output folder name
output_folder_name = f'Output_{settings["best_model"]}_{current_time}'

# Create the output folder
os.makedirs(os.path.join('Outputs', colo_output_folder, output_folder_name))

# Load deployment log
print('Loading deployment log...')
deployment_log = data_loading_func.load_deployment_log()


#Load harmonization data
#get list of all harmonization files to combine
harmon_file_list = deployment_log[(deployment_log['deployment']=='H')]['file_name']
#shorten the list to just the pod names (not the whole file name)
harmon_pod_list = [string.split('_')[0] for string in harmon_file_list]

#make a dictionary with a dataframe for each unique harmonization pod
pod_harmonization_data = dict.fromkeys(harmon_pod_list)

#Identify the colocation pod in the harmonization data
colo_pod_name= settings['colo_pod_name']
if not isinstance(colo_pod_name, str): #first check that colo_pod_name is a string)
    colo_pod_name = colo_pod_name[0]

#load pod data
print('Loading harmonization pod data...')
pod_harmonization_data = data_loading_func.load_data(harmon_file_list, deployment_log, settings['column_names'], 'H', settings['pollutant'])

# Check if there is any harmonization data
assert bool(pod_harmonization_data), "No harmonization data was found in the Harmonization folder that matched the deployment log. Stopping execution."

print('Preprocessing harmonization pod and reference data...')
for podname in pod_harmonization_data:
    #harmonization data preprocessing
   pod_harmonization_data[podname] = preprocessing_func.preprocessing_func(pod_harmonization_data[podname], settings['sensors_included'], settings['t_warmup'], settings['preprocess'])

if colo_pod_name not in pod_harmonization_data:
    raise KeyError(f'Harmonization data for the colocation pod {colo_pod_name} was not found in Harmonization folder. Run cannot continue.')
    
#create a dictionary to add harmonized pod timeseries into
pod_fitted = {key: None for key in pod_harmonization_data}
del pod_fitted[colo_pod_name]

#create a dictionary to add preprocessed, unharmonized pod timeseries into
preprocessed_harmon_data = {key: None for key in pod_harmonization_data}
del preprocessed_harmon_data [colo_pod_name]

#create a dataframe to add colocation pod harmonization data into
colo_pod_harmon_data =pd.DataFrame(columns=settings['sensors_included'])


#create a dictionary of model stats (R2, RMSE, MBE)
model_stats = dict.fromkeys(['Training_R2','Training_RMSE','Testing_RMSE','Training_MBE','Testing_MBE'])
for stat in model_stats:
    model_stats[stat]=dict.fromkeys(settings['sensors_included'])
    for sensor in settings['sensors_included']:
        model_stats[stat][sensor]=pd.DataFrame(columns=harmon_pod_list,index=range(settings['k_folds']))
        model_stats[stat][sensor]= model_stats[stat][sensor].drop(columns=colo_pod_name)

#create a dictionary to add lin reg models into for each sensor for each pod
harmonization_mdls = {key: None for key in pod_harmonization_data}
del harmonization_mdls[colo_pod_name]

#rename the colocation pod columns so we can differentiate in later code between regular pod and colo pod
column_coloPod =  {col: col + '_colo' for col in pod_harmonization_data[colo_pod_name].columns}
pod_harmonization_data[colo_pod_name]=pod_harmonization_data[colo_pod_name].rename(columns=column_coloPod)

for pod_num, podname in enumerate(pod_fitted):
    #make dataframes to put the preprocessed and fitted data into.
    pod_fitted[podname]=pd.DataFrame(columns=settings['sensors_included'])
    preprocessed_harmon_data[podname]=pd.DataFrame(columns=settings['sensors_included'])
    #make a dictionary to save the harmonization models into
    harmonization_mdls[podname] = dict.fromkeys(pod_fitted[podname].columns)
    
    #time average and align the data between the colocation pod (secondary standard) and other pods
    for sensor in settings['sensors_included']:
        if settings['retime_calc']=='median':
            temp= pd.concat([pod_harmonization_data[colo_pod_name][sensor + '_colo'], pod_harmonization_data[podname][sensor]], axis=1).resample(settings['time_interval']+'T').median()
        if settings['retime_calc']=='mean':
            temp= pd.concat([pod_harmonization_data[colo_pod_name][sensor + '_colo'], pod_harmonization_data[podname][sensor]], axis=1).resample(settings['time_interval']+'T').mean()
        temp.dropna(inplace=True)
        
        
        #create X and y dataframes for harmonization step
        X=temp.drop([sensor + '_colo'],axis=1)
        y=temp[sensor + '_colo']

        #k-fold cross validation of sensor linear regression
        #doing k fold here because less data and no hyperparameter tuning
        kf = KFold(n_splits=settings['k_folds'], shuffle=False)
        
        #initiate lin reg model for colo pod sensor to field pod sensor
        CV_model= LinearRegression()
        current_model= LinearRegression()

        print(f'Harmonizing pod {podname} {sensor} sensor to the colocation pod {sensor}...')
        # Perform k-fold cross-validation
        for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
            # Fit the model on the training data
            CV_model.fit(X_train, y_train)
        
            # Make predictions on the data
            y_train_predicted = CV_model.predict(X_train)
            y_test_predicted = CV_model.predict(X_test)
            
            # Evaluate the model's performance across each fold
            model_stats['Training_RMSE'][sensor].iloc[fold,pod_num] = (np.sqrt(mean_squared_error(y_train, y_train_predicted)))
            model_stats['Testing_RMSE'][sensor].iloc[fold,pod_num] = (np.sqrt(mean_squared_error(y_test, y_test_predicted)))
            model_stats['Training_R2'][sensor].iloc[fold,pod_num] = round(CV_model.score(X_train, y_train),2)
            model_stats['Training_MBE'][sensor].iloc[fold,pod_num] = np.mean(y_train_predicted - y_train)
            model_stats['Testing_MBE'][sensor].iloc[fold,pod_num] = np.mean(y_test_predicted - y_test)

        # build a model using all data (no CV splitting)
        current_model.fit(X, y)
        pod_fitted[podname][sensor]=current_model.predict(X)
        #save the model
        harmonization_mdls[podname][sensor]=current_model
        
        #save the raw, unfitted data 
        preprocessed_harmon_data[podname][sensor] = X
        
        #save the colocation pod harmonization data
        colo_pod_harmon_data[sensor] = y
    
    pod_fitted[podname] = pod_fitted[podname].set_index(X.index)

# Harmonization plotting
if 'harmon_stats_plot' in settings['harmon_plot_list']:
    plotting_func.harmon_stats_plot(model_stats, output_folder_name, colo_output_folder, settings['sensors_included'])
    
if 'harmon_scatter' in settings['harmon_plot_list']:
    plotting_func.harmon_scatter(colo_pod_harmon_data, pod_fitted, colo_output_folder, output_folder_name)
    
if 'harmon_timeseries' in settings['harmon_plot_list']:
    plotting_func.harmon_timeseries(colo_pod_harmon_data, pod_fitted, colo_output_folder, output_folder_name)


####### start field data
if settings['run_field'] == False:
    print("settings['run_field'] set to False, so no field analysis will be conducted.")
elif settings['run_field'] == True:
    print('Beginning field data analysis...')
    #after running MPC_python_120523, upload the chosen model using joblib
    fit_model= joblib.load(os.path.join('Outputs', colo_output_folder, f'{settings["best_model"]}_model.joblib'))
    
    #Load field data
    print('Loading field data...')
    #get list of all field files to combine
    field_file_list = deployment_log[(deployment_log['deployment']=='F')]['file_name']
    #shorten the list to just the pod names (not the whole file name)
    field_pod_list = [string.split('_')[0] for string in field_file_list]     
    
    #make a dictionary with a dataframe for each unique pod
    pod_field_data = dict.fromkeys(field_pod_list)
    
    #load pod data
    pod_field_data = data_loading_func.load_data(field_file_list, deployment_log, settings['column_names'], 'F',settings['pollutant'])
    
    # Check for pods in field_list not present in harmonization_list
    not_in_harmonization = [item for item in field_pod_list if item not in list(pod_harmonization_data)]
    
    # If there are items not in harmonization_list, raise a KeyError
    if not_in_harmonization:
        print()
        print(f"Pods in field_pod_list do not have harmonization model: {not_in_harmonization}. This pods will be skipped!")
        print()
        for i in not_in_harmonization:
            del pod_field_data[i]
    
    # Check if there is any field data
    assert bool(pod_field_data), "No field data was found in the Field folder that matched the deployment log. Stopping execution."
    
    #create a dictionary to add fitted field pod timeseries into
    X_fitted_field = {key: None for key in pod_field_data}
    X_fitted_field_std = {key: None for key in pod_field_data}
    
    Y_field_dict={key: None for key in pod_field_data} #they will have diff timeseries, so start by making dictionary in each then combine into a dataframe at the end
    Y_field_noindex={key: None for key in pod_field_data}
    
    
    for podname in pod_field_data:
        print(f'Preprocessing, harmonizing, and calibrating pod {podname}...')
        #field data preprocessing
        pod_field_data[podname] = preprocessing_func.preprocessing_func(pod_field_data[podname], settings['sensors_included'], settings['t_warmup'], settings['preprocess'])
    
        #time average the pod field data
        if settings['retime_calc']=='median':
            temp= pod_field_data[podname].resample(settings['time_interval']+'T').median()
        if settings['retime_calc']=='mean':
            temp= pod_field_data[podname].resample(settings['time_interval']+'T').mean()
        temp.dropna(inplace=True)
        
        # Fit and transform the data, and convert it back to a DataFrame
        X_fitted_field[podname]=pd.DataFrame(columns=settings['sensors_included'], index=temp.index)
        
        #apply harmonization models to the field sensors
        for i, sensor in enumerate(settings['sensors_included']):
            X=pd.DataFrame(temp[sensor], index=temp.index)
            X_fitted_field[podname][sensor]=harmonization_mdls[podname][sensor].predict(X)
        
        #create interaction terms if using in the colocation model
        if "interaction_terms" in settings['preprocess']:
            X_fitted_field[podname] = preprocessing_func.interaction_terms(X_fitted_field[podname])
    
        #time elapsed needs to come after time averaging to be accurate (at least for median)
        if "add_time_elapsed" in settings['preprocess']:
            X_fitted_field[podname] = preprocessing_func.add_time_elapsed(X_fitted_field[podname], settings['earliest_time'])
        
            #if you are using a fig2600/fig2602 ratio, make that column here
        if "fig_ratio" in settings['preprocess']:
            X_fitted_field[podname] = preprocessing_func.fig_ratio(X_fitted_field[podname])
    
        #Scaling of fitted X field data
        X_fitted_field_std[podname] = settings['scaler'].fit_transform(X_fitted_field[podname])
    
        #apply colocation model to the harmonized field data
        Y_field_dict[podname]=pd.DataFrame(fit_model.predict(X_fitted_field_std[podname]),index=X_fitted_field[podname].index,columns=[settings['pollutant']])
        
        Y_field_noindex[podname]=Y_field_dict[podname].reset_index()
    
    #convert the Y_field data into a single data frame instead of separated into a dictionary by pod names. Add a column that lists the pod name for the data sample    
    Y_field_df=pd.concat([df.assign(pod=name) for name, df in Y_field_noindex.items()])
    
    # Add location column to plot by this instead of pod
    Y_field_df = data_loading_func.field_location(Y_field_df, deployment_log)

    #field plotting
    if 'field_timeseries' in settings['field_plot_list']:
        plotting_func.field_timeseries(Y_field_df, settings['best_model'], output_folder_name, colo_output_folder, settings['pollutant'],settings['unit'])
    
    
    if 'field_boxplot' in settings['field_plot_list']:
        plotting_func.field_boxplot(Y_field_df, settings['best_model'], output_folder_name, colo_output_folder, settings['pollutant'],settings['unit'])
    
    #save out important info
    print('Saving important run data...')
    #save y_field data by pod
    excel_name = os.path.join('Outputs', colo_output_folder, output_folder_name, 'y_field_estimates_by_pod.xlsx')
    with pd.ExcelWriter(excel_name, engine='xlsxwriter') as writer:
        # Iterate through the dictionary and write each DataFrame to a sheet
        for sheet_name, df in Y_field_dict.items():
            df.to_excel(writer, sheet_name=sheet_name)
            
    #save y_field data by location
    # Split y_field_df into a dictionary based on the 'location' column
    Y_field_loc_dict = {location: group.drop(columns='location') for location, group in Y_field_df.groupby('location')}
    excel_name = os.path.join('Outputs', colo_output_folder, output_folder_name, 'y_field_estimates_by_loc.xlsx')
    with pd.ExcelWriter(excel_name, engine='xlsxwriter') as writer:
        # Iterate through the dictionary and write each DataFrame to a sheet
        for sheet_name, df in Y_field_loc_dict.items():
            df.to_excel(writer, sheet_name=sheet_name)
    
    #save X_field data
    excel_name = os.path.join('Outputs', colo_output_folder, output_folder_name, 'X_field.xlsx')
    with pd.ExcelWriter(excel_name, engine='xlsxwriter') as writer:
        # Iterate through the dictionary and write each DataFrame to a sheet
        for sheet_name, df in X_fitted_field.items(): 
            df.to_excel(writer, sheet_name=sheet_name)
    
    #save X_field_standardized data
    excel_name = os.path.join('Outputs', colo_output_folder, output_folder_name, 'X_field_standardized.xlsx')
    for pod_name in X_fitted_field_std:
        X_fitted_field_std[pod_name] = pd.DataFrame(data=X_fitted_field_std[pod_name], columns = X_fitted_field[pod_name].columns, index = X_fitted_field[pod_name].index)
    with pd.ExcelWriter(excel_name, engine='xlsxwriter') as writer:
        # Iterate through the dictionary and write each DataFrame to a sheet
        for sheet_name, df in X_fitted_field_std.items():
            df.to_excel(writer, sheet_name=sheet_name)
        
#save preprocessed harmonization pod data
excel_name = os.path.join('Outputs', colo_output_folder, output_folder_name, 'X_preprocessed_unfitted.xlsx')
with pd.ExcelWriter(excel_name, engine='xlsxwriter') as writer:
    # Iterate through the dictionary and write each DataFrame to a sheet
    for sheet_name, df in preprocessed_harmon_data.items():
        df.to_excel(writer, sheet_name=sheet_name)


#save fitted harmonization pod data
excel_name = os.path.join('Outputs', colo_output_folder, output_folder_name, 'X_harmonized.xlsx')
with pd.ExcelWriter(excel_name, engine='xlsxwriter') as writer:
    # Iterate through the dictionary and write each DataFrame to a sheet
    for sheet_name, df in pod_fitted.items():
        df.to_excel(writer, sheet_name=sheet_name)

#save colocation pod harmonization data
colo_pod_harmon_data.to_excel(os.path.join('Outputs', colo_output_folder, output_folder_name, 'colo_pod_harmon_data.xlsx'))

#save settings
joblib.dump(settings, os.path.join('Outputs', colo_output_folder, output_folder_name, 'run_settings.joblib'))

#save harmonization lin reg models
joblib.dump(harmonization_mdls, os.path.join('Outputs', colo_output_folder, output_folder_name, 'harmonization_models.joblib'))
   
# save model stats
for stat in model_stats:
    excel_name = os.path.join('Outputs', colo_output_folder, output_folder_name, f'harmonization_{stat}_.xlsx')
    with pd.ExcelWriter(excel_name, engine='xlsxwriter') as writer:
        # Iterate through the dictionary and write each DataFrame to a sheet
        for sheet_name, df in model_stats[stat].items():
            df.to_excel(writer, sheet_name=sheet_name,index=False)
