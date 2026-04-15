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

# packages from python_functions folder
from Python_Functions import preprocessing_func
from Python_Functions import data_loading_func
from Python_Functions import plotting_func

# initialize a dictionary (do not edit this line)
hf_set = {}

####
# edit the variables in this section for the harmonization - c  field run
hf_set['hf_run_name'] = 'final_field_gradboost_all'  # #Name of the harmonization/field outputs file. If you leave it blank inside the quotes, the output folder will be named with current datetime
# ^^ If you want the harmonization/field outputs folder to just be named with current datetime, set settings['run_name'] = '' (YOU NEED THE APOSTROPHES/QUOTES)
colo_output_folder = 'Output_O3_corrected_scaler_UTC'  # the code will pull the best_model from this, and also save new stuff into it

hf_set['run_field'] = True  # This should always be set to true for one cal (there is no harmonization)
hf_set['best_model'] = {'YPODA2':'gradboost','YPODG5':'gradboost', 'YPODL1':'gradboost','YPODL2':'gradboost','YPODL6':'gradboost','YPODL9':'gradboost','YPODR9':'gradboost'}  # model that you would like to apply to field data from the output_folder. this is individual to each pod


hf_set['field_plot_list'] = ['field_boxplot', 'field_timeseries', 'field_histogram']  # field plots: 'field_boxplot', 'field_timeseries', 'field_histogram', 'harmonized_field_hist'

hf_set['crop_field_time'] = False  # set to true if you want to crop the field times that are fit/plotted

hf_set['field_start'] = '2023-10-11 01:00:00'  # if field_crop_time is True, set the start time. everything before will be cropped.
hf_set['field_end'] = '2023-12-21 00:00:00'  # if field_crop_time is True, set the end time. everything after will be cropped.

####


###############
# Check if the output folder exists
if not os.path.exists(os.path.join('Outputs', colo_output_folder)):
    raise FileNotFoundError("The colocation output folder does not exist. Please double-check 'colo_output_folder'.")
# load run settings
settings = joblib.load(os.path.join('Outputs', colo_output_folder, 'run_settings.joblib'))  # do not change this line!

settings = {**settings, **hf_set}

# close previous figures
plt.close('all')

# Create harmonization_field output folder
if hf_set['hf_run_name'] == '':
    # Get the current time as YYMMDDHHss
    current_time = datetime.now().strftime('%y%m%d%H%M%S')
    # Create the output folder name
    output_folder_name = f'Output_{current_time}'
    del current_time
else:
    output_folder_name = f'Output_{settings["hf_run_name"]}'

# Check if the directory already exists
if os.path.exists(os.path.join('Outputs', colo_output_folder, output_folder_name)):
    raise FileExistsError(
        f"The Output folder '{output_folder_name}' already exists. Please choose a different output folder name or use the current time option (settings['colo_run_name'] == '').")
else:
    # Create the output folder
    os.makedirs(os.path.join('Outputs', colo_output_folder, output_folder_name))

# Load deployment log
print('Loading deployment log...')
deployment_log = data_loading_func.load_deployment_log('onecal')

# Identify the colocation pod in the harmonization data
colo_pod_names = settings['colo_pod_names']

####### start field data
if settings['run_field'] == False:
    print("settings['run_field'] set to False, so no field analysis will be conducted.")
elif settings['run_field'] == True:
    print('Beginning field data analysis...')
    # after running MPC_python_120523, upload the chosen model using joblib

    # Load field data
    # get list of all field files to combine
    field_file_list = deployment_log[(deployment_log['deployment'] == 'F')]['file_name']
    # shorten the list to just the pod names (not the whole file name)
    field_pod_list = [string.split('_')[0] for string in field_file_list]

    if not os.path.exists(os.path.join('Outputs', colo_output_folder, 'pod_field_data.joblib')):

        # make a dictionary with a dataframe for each unique pod
        pod_field_data = dict.fromkeys(field_pod_list)

        # load pod data
        pod_field_data, deployment_log = data_loading_func.onecal_load_data(field_file_list, deployment_log,
                                                                     settings['column_names'], 'F',
                                                                     settings['pollutant'], settings['ref_timezone'])

        for podname in pod_field_data:
            # field data preprocessing
            print(f'Preprocessing pod {podname}...')
            pod_field_data[podname] = preprocessing_func.preprocessing_func(pod_field_data[podname],
                                                                            settings['sensors_included'],
                                                                            settings['t_warmup'],
                                                                            settings['preprocess'])

        joblib.dump(pod_field_data, os.path.join('Outputs', colo_output_folder, 'pod_field_data.joblib'))

    else:
        print('Loading preprocessed field pod data from joblib...')

    pod_field_data = joblib.load(os.path.join('Outputs', colo_output_folder, 'pod_field_data.joblib'))

    # Check for pods in field_list not present in harmonization_list
    not_in_colocation = set([item for item in field_pod_list if item not in colo_pod_names])

    # If there are items not in harmonization_list, raise a KeyError
    if not_in_colocation:
        print()
        print(
            f"Pods in field_pod_list do not have a colocation model: {not_in_colocation}. This pods will be skipped!")
        print()
        for i in not_in_colocation:
            if i in pod_field_data:
                del pod_field_data[i]

    # Check if there is any field data
    assert bool(
        pod_field_data), "No field data was found in the Field folder that matched the deployment log. Stopping execution."

    # create a dictionary to add fitted field pod timeseries into
    X_field = {key: None for key in pod_field_data}
    X_field_std = {key: None for key in pod_field_data}
    X_field_noindex = {key: None for key in pod_field_data}

    Y_field_dict = {key: None for key in pod_field_data}  # they will have diff timeseries, so start by making dictionary in each then combine into a dataframe at the end
    Y_field_noindex = {key: None for key in pod_field_data}
    melted_X = {key: None for key in pod_field_data}

    podnames_copy = list(pod_field_data.keys())
    for podname in podnames_copy:
        print(f'Calibrating pod {podname}...')

        if hf_set['crop_field_time']:
            time_removed = (pod_field_data[podname].index < hf_set['field_start']) | (
                        pod_field_data[podname].index > hf_set['field_end'])
            pod_field_data[podname] = pod_field_data[podname][~time_removed]

        if pod_field_data[podname].empty:
            del pod_field_data[podname]
            del Y_field_dict[podname]
            del Y_field_noindex[podname]
            del melted_X[podname]
        else:
            # time average the pod field data
            if settings['retime_calc'] == 'median':
                temp = pod_field_data[podname].resample(settings['time_interval'] + 'T').median()
            if settings['retime_calc'] == 'mean':
                temp = pod_field_data[podname].resample(settings['time_interval'] + 'T').mean()
            temp.dropna(inplace=True)

            # Fit and transform the data, and convert it back to a DataFrame
            X_field[podname] = temp

            # create interaction terms if using in the colocation model
            if "interaction_terms" in settings['preprocess']:
                X_field[podname] = preprocessing_func.interaction_terms(X_field[podname])

                # if you are using a fig2600/fig2602 ratio, make that column here
            if "fig2600_2602_ratio" in settings['preprocess']:
                X_field[podname] = preprocessing_func.fig2600_2602_ratio(X_field[podname])

            # time elapsed needs to come after time averaging to be accurate (at least for median)
            if "add_time_elapsed" in settings['preprocess']:
                X_field[podname] = preprocessing_func.add_time_elapsed(X_field[podname],
                                                                              settings['earliest_time'])

            temp2 = X_field[podname].copy()

            # zscore the data based on the scaler set in the colocation step
            # each scaler is pod specific, but we use the colocation data to calculate the scaler values instead of field so that we don't cancel out real differences between field sites!
            temp2[settings['sensors_included']] = settings['scaler'][podname].transform(X_field[podname][settings['sensors_included']])

            X_field_std[podname] = temp2.to_numpy()

            # X_fitted_field_std[podname] = X_fitted_field[podname]
            # ^^^^ need to undo this z scoring at the end!!


            fit_model = joblib.load(os.path.join('Outputs', colo_output_folder,
                                                     f'{podname}_{settings["best_model"][podname]}_model.joblib'))

            # apply colocation model to the harmonized field data
            Y_field_dict[podname] = pd.DataFrame(fit_model.predict(X_field_std[podname]),
                                                 index=X_field[podname].index, columns=[settings['pollutant']])

            Y_field_noindex[podname] = Y_field_dict[podname].reset_index()

            X_field_noindex[podname] = X_field[podname].reset_index()
            melted_X[podname] = pd.melt(X_field_noindex[podname], id_vars='datetime', var_name='Sensor',
                                        value_name='Reading')

    del podnames_copy
    del X_field_noindex
    del temp2

    # convert the Y_field data into a single data frame instead of separated into a dictionary by pod names. Add a column that lists the pod name for the data sample
    Y_field_df = pd.concat([df.assign(pod=name) for name, df in Y_field_noindex.items()])
    X_field_df = pd.concat([df.assign(pod=name) for name, df in melted_X.items()])
    del melted_X

    # Add location column to plot by this instead of pod
    Y_field_df = data_loading_func.field_location(Y_field_df, deployment_log, settings['ref_timezone'])
    X_field_df = data_loading_func.field_location(X_field_df, deployment_log, settings['ref_timezone'])

    # field plotting
    if 'field_timeseries' in settings['field_plot_list']:
        plotting_func.field_timeseries(Y_field_df, settings['best_model'], output_folder_name, colo_output_folder,
                                       settings['pollutant'], settings['unit'])

    if 'field_boxplot' in settings['field_plot_list']:
        plotting_func.field_boxplot(Y_field_df, settings['best_model'], output_folder_name, colo_output_folder,
                                    settings['pollutant'], settings['unit'])

    if 'field_histogram' in settings['field_plot_list']:
        plotting_func.field_histogram(Y_field_df, settings['best_model'], output_folder_name, colo_output_folder,
                                      settings['pollutant'], settings['unit'])

    if 'harmonized_field_hist' in settings['field_plot_list']:
        plotting_func.harmonized_field_hist(X_field_df, output_folder_name, colo_output_folder,
                                            settings['sensors_included'])

    # save out important info
    print('Saving important field data...')
    # save y_field data by pod
    excel_name = os.path.join('Outputs', colo_output_folder, output_folder_name, 'y_field_estimates_by_pod.xlsx')
    with pd.ExcelWriter(excel_name, engine='xlsxwriter') as writer:
        # Iterate through the dictionary and write each DataFrame to a sheet
        for sheet_name, df in Y_field_dict.items():
            df.to_excel(writer, sheet_name=sheet_name)

    # save y_field data by location
    # Split y_field_df into a dictionary based on the 'location' column
    Y_field_loc_dict = {location: group.drop(columns='location') for location, group in Y_field_df.groupby('location')}
    excel_name = os.path.join('Outputs', colo_output_folder, output_folder_name, 'y_field_estimates_by_loc.xlsx')
    with pd.ExcelWriter(excel_name, engine='xlsxwriter') as writer:
        # Iterate through the dictionary and write each DataFrame to a sheet
        for sheet_name, df in Y_field_loc_dict.items():
            df.to_excel(writer, sheet_name=sheet_name)

    # save X_field data
    excel_name = os.path.join('Outputs', colo_output_folder, output_folder_name, 'X_field.xlsx')
    with pd.ExcelWriter(excel_name, engine='xlsxwriter') as writer:
        # Iterate through the dictionary and write each DataFrame to a sheet
        for sheet_name, df in X_field.items():
            df.to_excel(writer, sheet_name=sheet_name)

    # save X_field_standardized data
    excel_name = os.path.join('Outputs', colo_output_folder, output_folder_name, 'X_field_standardized.xlsx')
    for pod_name in X_field_std:
        X_field_std[pod_name] = pd.DataFrame(data=X_field_std[pod_name],
                                                    columns=X_field[pod_name].columns,
                                                    index=X_field[pod_name].index)

    with pd.ExcelWriter(excel_name, engine='xlsxwriter') as writer:
        # Iterate through the dictionary and write each DataFrame to a sheet
        for sheet_name, df in X_field_std.items():
            df.to_excel(writer, sheet_name=sheet_name)

# save settings
joblib.dump(settings, os.path.join('Outputs', colo_output_folder, output_folder_name, 'run_settings.joblib'))



