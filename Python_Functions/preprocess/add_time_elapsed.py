# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 10:43:08 2023

@author: cfris
"""
def add_time_elapsed(data, earliest_time):
    # Create a new column for time elapsed since the first time index
    data['time_elapsed'] = data.index - earliest_time
    # Convert the time elapsed column to seconds
    data['time_elapsed_seconds'] = data['time_elapsed'].dt.total_seconds()
    data = data.drop('time_elapsed',axis=1)
    return data