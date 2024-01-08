# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 14:33:19 2023

@author: cfris
"""
import pandas as pd

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
         from Python_Functions.preprocess import temp_C_2_K
         data = temp_C_2_K.temp_C_2_K(data)
     
     #convert relative humidity to absolute
     if "hum_rel_2_abs" in preprocess:
         if "temp_C_2_K" in preprocess:
             from Python_Functions.preprocess import hum_rel_2_abs
             data= hum_rel_2_abs.hum_rel_2_abs(data) 
         else: 
             raise KeyError('Humidity could not be converted because temperature was not converted to K. Add TempC2K to preprocessing')
    
     
     #remove unused columns in colo pod and ref
     data = data[sensors_included]
     
     #put data in time order (otherwise a lot of data might be deleted when removing warm up)
     data = data.sort_index()
     
     # Remove warm-up
     if "rmv_warmup" in preprocess:
         from Python_Functions.preprocess import rmv_warmup
         data = rmv_warmup.rmv_warmup(data, t_warmup)
     
     # Drop rows with NaN values
     data.dropna(inplace=True)
     return data