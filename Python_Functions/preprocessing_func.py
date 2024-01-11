# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 14:33:19 2023

@author: cfris
"""
import pandas as pd
from atmos import calculate
from sklearn.preprocessing import PolynomialFeatures

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