# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 14:33:19 2023

@author: cfris
"""
import pandas as pd
from atmos import calculate
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

def temp_C_2_K(data):
    #convert temp from celsius to kelvin
    # Check if 'temperature' is not in columns
    if 'Temperature' not in data.columns:
        raise KeyError("'Temperature' column not found in pod data, so temp_C_2_K cannot run. Check column_names variable.")

    data["Temperature"]=data["Temperature"]+273.15
    return data

def rmv_warmup(data, t_warmup):
    # remove the first warm up period
    initial_warmup_rows = data.index < (data.index[0] + pd.Timedelta(minutes=t_warmup))
    data = data[~initial_warmup_rows]
    # remove an additional warm up period after anytime pod turns off
    time_diff = data.index.to_series().diff()
    median_timestep = time_diff.median()  # finding median time step
    rows_to_remove = time_diff > 10 * median_timestep  # identifying rows where the time gap is 10 times longer than the median time gap between samples (pod was likely turned off)

    # the following lines identify rows where the pod likely turned off, and then remove a warm up period length of time after it turned off
    indices_to_remove = set()
    for i, remove in enumerate(rows_to_remove):
        if remove:
            indices_to_remove.add(data.index[
                                      i])  # if the row is one that has a greater time diff, it is added to the list of what should be removed
            for j in range(i + 1, len(rows_to_remove)):
                if data.index[j] - data.index[i] > pd.Timedelta(
                        minutes=t_warmup):  # if the next row is less than the total warm up time, it should also be removed. here, this inner loop breaks if the time is past the warm up period
                    break
                indices_to_remove.add(data.index[j])

    data = data.drop(index=list(indices_to_remove))
    return data

def interaction_terms(data):
    #create columns of interaction terms that are a each 2-sensor combination multiplied together
    #for CO, temp, humidity sensor, it would be CO*temp, CO*humidity, temp*humidity
    poly = PolynomialFeatures(degree=2,interaction_only=True, include_bias=False)
    data = poly.fit_transform(data)
    return data

def hum_rel_2_abs(data):
    if 'Humidity' not in data.columns:
        raise KeyError(
            "'Humidity' column not found in pod data, so hum_rel_2_abs cannot run. Check column_names variable.")

    if 'Pressure' not in data.columns:
        raise KeyError(
            "'Pressure' column not found in pod data, so hum_rel_2_abs cannot run. Check column_names variable.")

    # convert humidity from relative to absolute using the atmos package
    AH = calculate('AH', RH=data["Humidity"], p=data["Pressure"] * 100, T=data["Temperature"], debug=True)
    data["Humidity"] = AH[0]
    return data

def add_time_elapsed(data, earliest_time):
    # Create a new column for time elapsed since the first time index
    data['time_elapsed'] = data.index - earliest_time
    # Convert the time elapsed column to seconds
    data['time_elapsed_seconds'] = data['time_elapsed'].dt.total_seconds()
    data = data.drop('time_elapsed',axis=1)
    return data

def fig_ratio(data):
    if 'Fig2600' not in data.columns or 'Fig2602' not in data.columns:
        raise KeyError(
            "'Fig2600' AND/OR 'Fig2602' column(s) not found in pod data, so fig_ratio cannot run. Check column_names variable.")

    data['fig ratio'] = data['Fig2600'] / data['Fig2602']
    return data

def binned_resample(X_train, y_train, n_bins):
    downsampled_series_list = []
    # Run on reference data
    target_bin_multiplier = 2  #Target samples per bin is set as 1/n_bins. However, this can be adjusted if you don't want to remove so much data
    # When target_bin_multiplier is 2, the target samples per bin is twice as much as the mean number of samples per bin. Less data will be removed this way.
    bins, edges = np.histogram(y_train, bins=n_bins, density=False)

    # Determine the desired number of samples per bin
    target_samples_per_bin = round(target_bin_multiplier * len(y_train) // n_bins)

    # Downsampling to ensure the same number of samples in each bin
    rows_to_remove = []
    for i in range(n_bins):
        bin_indices = np.where((y_train >= edges[i]) & (y_train <= edges[i + 1]))[0]
        if len(bin_indices) > target_samples_per_bin:
            # If more samples in the bin than the target, randomly select samples
            selected_indices = np.random.choice(bin_indices, size=target_samples_per_bin, replace=False)
            selected_values = y_train.iloc[selected_indices]
            downsampled_series_list.append(selected_values)
            # Save indices to remove from the DataFrame
            rows_to_remove.extend(bin_indices[~np.isin(bin_indices, selected_indices)])
        else:
            # If fewer samples, include all samples in the bin
            downsampled_series_list.append(y_train.iloc[bin_indices])

    # Concatenate the downsampled Series into a single Series
    y_selected = pd.concat(downsampled_series_list) #y selected is shuffled!
    y_selected = y_selected.sort_index()
    # Remove rows from df_to_remove based on the indices removed during downsampling
    #X_selected = X_train.drop(index=X_train.index[rows_to_remove])
    X_selected = np.delete(X_train, rows_to_remove, axis=0) #X selected is not!!! this is causing model fit problems.

    return X_selected, y_selected

def resample_quartile(X_train, y_train, quartile, downsampling_rate):

    quartiles = pd.DataFrame({'Lower':[0, 0.25, 0.5, 0.75],
                              'Upper': [0.25, 0.5, 0.75, 1]},
                             index = ['first','second','third','fourth'])

    upper_edge = y_train.quantile(quartiles['Upper'].loc[quartile])
    lower_edge = y_train.quantile(quartiles['Lower'].loc[quartile])
    quartile_filter = (y_train < upper_edge) & (y_train >= lower_edge)

    quartile_indices = y_train[quartile_filter].index

    # Calculate the number of instances to keep
    num_instances_to_remove = int(len(quartile_indices) * (1-downsampling_rate))

    # Randomly sample instances to keep
    removed_quartile_indices = np.random.choice(quartile_indices, size=num_instances_to_remove,
                                                          replace=False)

    # Create the downsampled dataset
    y_selected = y_train.drop(index=removed_quartile_indices)
    X_selected = X_train.drop(index=removed_quartile_indices)

    return X_selected, y_selected
    # Now, 'downsampled_df' represents the dataset with downsampled lower quartile values

def preprocessing_func(data, sensors_included, t_warmup, preprocess):
     #change 999 to NA
     data.replace(999, pd.NA, inplace=True)

     #make sure all columns are numeric to avoid errors later:
     for col in data:
         data[col] = pd.to_numeric(data[col], errors='coerce').astype(float)
    
     #apply preprocess (rmv 999 and NaN, rmv warm up, humid and temp conversion)
     #scaling happens in the ML section instead of the preprocess section here BECAUSE WE WANT TO SCALE ONLY THE TRAINING DATA
     
     #convert temperature from C to K (this is needed for humidity conversion)
     if "temp_C_2_K" in preprocess:
         data = temp_C_2_K(data)
     
     #convert relative humidity to absolute
     if "hum_rel_2_abs" in preprocess:
         if "temp_C_2_K" in preprocess:
             data= hum_rel_2_abs(data)
         else: 
             raise KeyError('Humidity could not be converted because temperature was not converted to K. Add TempC2K to preprocessing')
    
     
     #remove unused columns in colo pod and ref
     data = data[sensors_included]
     
     #put data in time order (otherwise a lot of data might be deleted when removing warm up)
     data = data.sort_index()
     
     # Remove warm-up
     if "rmv_warmup" in preprocess:
         data = rmv_warmup(data, t_warmup)
     
     # Drop rows with NaN values
     data.dropna(inplace=True)
     return data