# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 11:44:55 2023

@author: cfris
"""

def fig_ratio(data):
    if 'Fig2600' not in data.columns or 'Fig2602' not in data.columns:
        raise KeyError("'Fig2600' AND/OR 'Fig2602' column(s) not found in pod data, so fig_ratio cannot run. Check column_names variable.")
        
    data['fig ratio'] = data['Fig2600']/data['Fig2602']
    return data