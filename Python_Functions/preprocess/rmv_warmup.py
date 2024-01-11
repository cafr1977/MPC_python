# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 10:46:32 2023

@author: cfris
"""

import pandas as pd

def rmv_warmup(data, t_warmup):
    #remove the first warm up period
    initial_warmup_rows=data.index < (data.index[0] + pd.Timedelta(minutes=t_warmup))
    data=data[~initial_warmup_rows]
    #remove an additional warm up period after anytime pod turns off 
    time_diff=data.index.to_series().diff()
    median_timestep=time_diff.median() #finding median time step
    rows_to_remove = time_diff > 10 * median_timestep #identifying rows where the time gap is 10 times longer than the median time gap between samples (pod was likely turned off)
    
   #the following lines identify rows where the pod likely turned off, and then remove a warm up period length of time after it turned off 
    indices_to_remove = set()
    for i,remove in enumerate(rows_to_remove):
        if remove:
            indices_to_remove.add(data.index[i])  #if the row is one that has a greater time diff, it is added to the list of what should be removed
            for j in range(i+1, len(rows_to_remove)):
                if data.index[j]-data.index[i] > pd.Timedelta(minutes=t_warmup): #if the next row is less than the total warm up time, it should also be removed. here, this inner loop breaks if the time is past the warm up period
                    break
                indices_to_remove.add(data.index[j])
                
    data = data.drop(index=list(indices_to_remove))
    return data
