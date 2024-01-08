# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 10:21:08 2023

@author: cfris
"""

def temp_C_2_K(data):
    #convert temp from celsius to kelvin
    # Check if 'temperature' is not in columns
    if 'Temperature' not in data.columns:
        raise KeyError("'Temperature' column not found in pod data, so temp_C_2_K cannot run. Check column_names variable.")

    data["Temperature"]=data["Temperature"]+273.15
    return data