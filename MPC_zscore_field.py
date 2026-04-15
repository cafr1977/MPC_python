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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# packages from python_functions folder
from Python_Functions import preprocessing_func
from Python_Functions import data_loading_func
from Python_Functions import plotting_func

# initialize a dictionary (do not edit this line)
hf_set = {}

####
# edit the variables in this section for the harmonization - field run
hf_set['hf_run_name'] = ''  # #Name of the harmonization/field outputs file. If you leave it blank inside the quotes, the output folder will be named with current datetime
# ^^ If you want the harmonization/field outputs folder to just be named with current datetime, set settings['run_name'] = '' (YOU NEED THE APOSTROPHES/QUOTES)
colo_output_folder = 'Output_testingTimeElapsed'  # the code will pull the best_model from this, and also save new stuff into it

hf_set['run_field'] = True  # True if you want to apply calibration to field data,
# False if you only want to look at harmonization data
hf_set['best_model'] = 'lin_reg'  # model that you would like to apply to field data from the output_folder

hf_set['field_plot_list'] = ['field_boxplot', 'field_timeseries']  # field plots: 'field_boxplot', 'field_timeseries', 'field_histogram', 'harmonized_field_hist'
# ^^ harmonized_field_hist plots the field data after the harmonization correction is applied but before the field data is calibrated to the colocaiton model
hf_set['harmon_plot_list'] = ['harmon_timeseries', 'harmon_scatter']  # harmonization plots: 'harmon_timeseries','harmon_stats_plot', 'harmon_scatter',

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
    output_folder_name = f'Output_{settings["best_model"]}_{current_time}'
    del current_time
else:
    output_folder_name = f'Output_{settings["best_model"]}_{settings["hf_run_name"]}'

# Check if the directory already exists
if os.path.exists(os.path.join('Outputs', colo_output_folder, output_folder_name)):
    raise FileExistsError(
        f"The Output folder '{output_folder_name}' already exists. Please choose a different output folder name or use the current time option (settings['colo_run_name'] == '').")
else:
    # Create the output folder
    os.makedirs(os.path.join('Outputs', colo_output_folder, output_folder_name))

# Load deployment log
print('Loading deployment log...')
deployment_log = data_loading_func.load_deployment_log('onehop')

# get list of all harmonization files to combine
harmon_file_list = deployment_log[(deployment_log['deployment'] == 'H')]['file_name']
# shorten the list to just the pod names (not the whole file name)
harmon_pod_list = [string.split('_')[0] for string in harmon_file_list]
harmon_pod_list = list(set(harmon_pod_list))

# Identify the colocation pod in the harmonization data
colo_pod_name = settings['colo_pod_name']
if not isinstance(colo_pod_name, str):  # first check that colo_pod_name is a string)
    colo_pod_name = colo_pod_name[0]

# load pod data
if not os.path.exists(os.path.join('Outputs', colo_output_folder, 'pod_harmonization_data.joblib')):
    # Load harmonization data
    # make a dictionary with a dataframe for each unique harmonization pod
    pod_harmonization_data = dict.fromkeys(harmon_pod_list)

    print('Loading harmonization pod data from txt files...')
    pod_harmonization_data, deployment_log = data_loading_func.onehop_load_data(harmon_file_list, deployment_log,
                                                                         settings['column_names'], 'H',
                                                                         settings['pollutant'],
                                                                         settings['ref_timezone'])

    # Check if there is any harmonization data
    assert bool(
        pod_harmonization_data), "No harmonization data was found in the Harmonization folder that matched the deployment log. Stopping execution."

    print('Preprocessing harmonization pod and reference data...')
    for podname in pod_harmonization_data:
        # harmonization data preprocessing
        pod_harmonization_data[podname] = preprocessing_func.preprocessing_func(pod_harmonization_data[podname],
                                                                                settings['sensors_included'],
                                                                                settings['t_warmup'],
                                                                                settings['preprocess'])

    if colo_pod_name not in pod_harmonization_data:
        raise KeyError(
            f'Harmonization data for the colocation pod {colo_pod_name} was not found in Harmonization folder. Run cannot continue.')

    joblib.dump(pod_harmonization_data, os.path.join('Outputs', colo_output_folder, 'pod_harmonization_data.joblib'))

else:
    print('Loading preprocessed harmonization pod data from joblib...')

pod_harmonization_data = joblib.load(os.path.join('Outputs', colo_output_folder, 'pod_harmonization_data.joblib'))

# create a dictionary to add harmonized pod timeseries into
pod_fitted = {key: None for key in pod_harmonization_data}
del pod_fitted[colo_pod_name]

# create a dictionary to add preprocessed, unharmonized pod timeseries into
preprocessed_harmon_data = {key: None for key in pod_harmonization_data}
del preprocessed_harmon_data[colo_pod_name]

# create a dataframe to add colocation pod harmonization data into
colo_pod_harmon_data = pd.DataFrame(columns=settings['sensors_included'])

# add the time averaged colocation pod data to a dataframe for plotting later
for sensor in settings['sensors_included']:
    if settings['retime_calc'] == 'median':
        colo_pod_harmon_data[sensor] = pod_harmonization_data[colo_pod_name][sensor].resample(
            settings['time_interval'] + 'T').median()
    if settings['retime_calc'] == 'mean':
        colo_pod_harmon_data[sensor] = pod_harmonization_data[colo_pod_name][sensor].resample(
            settings['time_interval'] + 'T').mean()

#added this so we can compare side by side scatter plot of colo pod and others after z scoring
colo_scaler = StandardScaler()
colo_scaler.fit(colo_pod_harmon_data)
colo_pod_harmon_data = pd.DataFrame(colo_scaler.transform(colo_pod_harmon_data), index = colo_pod_harmon_data.index, columns = colo_pod_harmon_data.columns)

# rename the colocation pod columns so we can differentiate in later code between regular pod and colo pod
column_coloPod = {col: col + '_colo' for col in pod_harmonization_data[colo_pod_name].columns}
pod_harmonization_data[colo_pod_name] = pod_harmonization_data[colo_pod_name].rename(columns=column_coloPod)

# create a dictionary to save each pod's scaler values into.
scaler = {}

for pod_num, podname in enumerate(pod_fitted):
    # make dataframes to put the preprocessed and fitted data into.
    #pod_fitted[podname] = pd.DataFrame(columns=settings['sensors_included'])
    preprocessed_harmon_data[podname] = pd.DataFrame(columns=settings['sensors_included'])

    # time average and align the data between the colocation pod (secondary standard) and other pods
    for sensor in settings['sensors_included']:
        if settings['retime_calc'] == 'median':
            temp = pd.concat(
                [pod_harmonization_data[colo_pod_name][sensor + '_colo'], pod_harmonization_data[podname][sensor]],
                axis=1).resample(settings['time_interval'] + 'T').median()
        if settings['retime_calc'] == 'mean':
            temp = pd.concat(
                [pod_harmonization_data[colo_pod_name][sensor + '_colo'], pod_harmonization_data[podname][sensor]],
                axis=1).resample(settings['time_interval'] + 'T').mean()
        temp.dropna(inplace=True)

        # create X and y dataframes for harmonization step
        X = temp.drop([sensor + '_colo'], axis=1)

        # save the raw, unfitted data
        preprocessed_harmon_data[podname][sensor] = X[sensor]

    #calculate a pod-specific scaler using the harmonization data (Not the field data!!)
    #we use the harmonization data to make sure we are only correcting for sensor bias, NOT actual differences in the field sites
    scaler[podname] = StandardScaler()
    scaler[podname].fit(preprocessed_harmon_data[podname])

    #apply the pod-specific scaler to the harmonization data so we can visualize how much the zscored data correlates across pods in the plots below
    pod_fitted[podname] = pd.DataFrame(scaler[podname].transform(preprocessed_harmon_data[podname]),
                                       index = preprocessed_harmon_data[podname].index, columns = preprocessed_harmon_data[podname].columns)

# Harmonization plotting
if 'harmon_scatter' in settings['harmon_plot_list']:
    plotting_func.harmon_scatter(colo_pod_harmon_data, pod_fitted, colo_output_folder, output_folder_name)

if 'harmon_timeseries' in settings['harmon_plot_list']:
    plotting_func.harmon_timeseries(colo_pod_harmon_data, pod_fitted, colo_output_folder, output_folder_name)

print('Saving important harmonization data...')

# save preprocessed harmonization pod data
excel_name = os.path.join('Outputs', colo_output_folder, output_folder_name, 'X_preprocessed_unfitted.xlsx')
with pd.ExcelWriter(excel_name, engine='xlsxwriter') as writer:
    # Iterate through the dictionary and write each DataFrame to a sheet
    for sheet_name, df in preprocessed_harmon_data.items():
        df.to_excel(writer, sheet_name=sheet_name)

# save fitted harmonization pod data
excel_name = os.path.join('Outputs', colo_output_folder, output_folder_name, 'X_harmonized.xlsx')
with pd.ExcelWriter(excel_name, engine='xlsxwriter') as writer:
    # Iterate through the dictionary and write each DataFrame to a sheet
    for sheet_name, df in pod_fitted.items():
        df.to_excel(writer, sheet_name=sheet_name)

# save colocation pod harmonization data
colo_pod_harmon_data.to_excel(
    os.path.join('Outputs', colo_output_folder, output_folder_name, 'colo_pod_harmon_data.xlsx'))


harmonized_pods = list(pod_harmonization_data)
del pod_harmonization_data, preprocessed_harmon_data, X, temp, colo_pod_harmon_data

####### start field data
if settings['run_field'] == False:
    print("settings['run_field'] set to False, so no field analysis will be conducted.")
elif settings['run_field'] == True:
    print('Beginning field data analysis...')
    # after running MPC_python_120523, upload the chosen model using joblib
    fit_model = joblib.load(os.path.join('Outputs', colo_output_folder, f'{settings["best_model"]}_model.joblib'))

    # Load field data
    # get list of all field files to combine
    field_file_list = deployment_log[(deployment_log['deployment'] == 'F')]['file_name']
    # shorten the list to just the pod names (not the whole file name)
    field_pod_list = [string.split('_')[0] for string in field_file_list]

    if not os.path.exists(os.path.join('Outputs', colo_output_folder, 'pod_field_data.joblib')):

        # make a dictionary with a dataframe for each unique pod
        pod_field_data = dict.fromkeys(field_pod_list)

        # load pod data
        pod_field_data, deployment_log = data_loading_func.onehop_load_data(field_file_list, deployment_log,
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
    not_in_harmonization = set([item for item in field_pod_list if item not in harmonized_pods])

    # If there are items not in harmonization_list, raise a KeyError
    if not_in_harmonization:
        print()
        print(
            f"Pods in field_pod_list do not have harmonization model: {not_in_harmonization}. This pods will be skipped!")
        print()
        for i in not_in_harmonization:
            if i in pod_field_data:
                del pod_field_data[i]

    # Check if there is any field data
    assert bool(
        pod_field_data), "No field data was found in the Field folder that matched the deployment log. Stopping execution."

    # create a dictionary to add fitted field pod timeseries into
    X_fitted_field = {key: None for key in pod_field_data}
    X_fitted_field_std = {key: None for key in pod_field_data}
    X_fitted_field_noindex = {key: None for key in pod_field_data}

    Y_field_dict = {key: None for key in pod_field_data}  # they will have diff timeseries, so start by making dictionary in each then combine into a dataframe at the end
    Y_field_noindex = {key: None for key in pod_field_data}
    melted_X = {key: None for key in pod_field_data}

    podnames_copy = list(pod_field_data.keys())
    for podname in podnames_copy:
        print(f'Harmonizing, and calibrating pod {podname}...')

        #crop field time if needed
        if hf_set['crop_field_time']:
            time_removed = (pod_field_data[podname].index < hf_set['field_start']) | (
                        pod_field_data[podname].index > hf_set['field_end'])
            pod_field_data[podname] = pod_field_data[podname][~time_removed]

        if pod_field_data[podname].empty:
            del pod_field_data[podname]
            del Y_field_dict[podname]
            del Y_field_noindex[podname]
            del X_fitted_field[podname]
            del X_fitted_field_std[podname]
            del X_fitted_field_noindex[podname]
            del melted_X[podname]
        else:
            # time average the pod field data
            if settings['retime_calc'] == 'median':
                temp = pod_field_data[podname].resample(settings['time_interval'] + 'T').median()
            if settings['retime_calc'] == 'mean':
                temp = pod_field_data[podname].resample(settings['time_interval'] + 'T').mean()
            temp.dropna(inplace=True)

            #zscore the field data using each pod's scaler calculated from the harmonization data
            X_fitted_field[podname] = pd.DataFrame(scaler[podname].transform(temp), index = temp.index, columns = temp.columns)

            # create interaction terms if using in the colocation model
            if "interaction_terms" in settings['preprocess']:
                X_fitted_field[podname] = preprocessing_func.interaction_terms(X_fitted_field[podname])

                # if you are using a fig2600/fig2602 ratio, make that column here
            if "fig2600_2602_ratio" in settings['preprocess']:
                X_fitted_field[podname] = preprocessing_func.fig2600_2602_ratio(X_fitted_field[podname])

            # time elapsed needs to come after time averaging to be accurate (at least for median)
            if "add_time_elapsed" in settings['preprocess']:
                X_fitted_field[podname] = preprocessing_func.add_time_elapsed(X_fitted_field[podname], settings['earliest_time'])

            # apply colocation model to the harmonized field data
            Y_field_dict[podname] = pd.DataFrame(fit_model.predict(X_fitted_field[podname]),
                                                 index=temp.index, columns=[settings['pollutant']])

            Y_field_noindex[podname] = Y_field_dict[podname].reset_index()

            X_fitted_field_noindex[podname] = X_fitted_field[podname].reset_index()
            melted_X[podname] = pd.melt(X_fitted_field_noindex[podname], id_vars='datetime', var_name='Sensor',
                                        value_name='Reading')

    del podnames_copy

    # convert the Y_field data into a single data frame instead of separated into a dictionary by pod names. Add a column that lists the pod name for the data sample
    Y_field_df = pd.concat([df.assign(pod=name) for name, df in Y_field_noindex.items()])
    X_fitted_field_df = pd.concat([df.assign(pod=name) for name, df in melted_X.items()])
    del melted_X

    # Add location column to plot by this instead of pod
    Y_field_df = data_loading_func.field_location(Y_field_df, deployment_log, settings['ref_timezone'])
    X_fitted_field_df = data_loading_func.field_location(X_fitted_field_df, deployment_log, settings['ref_timezone'])

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
        plotting_func.harmonized_field_hist(X_fitted_field_df, output_folder_name, colo_output_folder,
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
        for sheet_name, df in X_fitted_field.items():
            df.to_excel(writer, sheet_name=sheet_name)


# save settings
joblib.dump(settings, os.path.join('Outputs', colo_output_folder, output_folder_name, 'run_settings.joblib'))



