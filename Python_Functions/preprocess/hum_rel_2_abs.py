# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 10:38:58 2023

@author: cfris
"""
from atmos import calculate
#pressure in Pa, written as hPa in data
#AH in kg/m3
#RH in %
#T in K

def hum_rel_2_abs(data):
        
    if 'Humidity' not in data.columns:
        raise KeyError("'Humidity' column not found in pod data, so hum_rel_2_abs cannot run. Check column_names variable.")
        
    if 'Pressure' not in data.columns:
        raise KeyError("'Pressure' column not found in pod data, so hum_rel_2_abs cannot run. Check column_names variable.")

    #convert humidity from relative to absolute using the atmos package
    AH=calculate('AH',RH=data["Humidity"], p=data["Pressure"]*100, T=data["Temperature"], debug=True)
    data["Humidity"]=AH[0]
    return data