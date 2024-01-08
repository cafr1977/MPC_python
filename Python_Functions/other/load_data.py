# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 14:59:08 2023

@author: cfris
"""
import os
import pandas as pd


def load_data(data_file_list, deployment_log, column_names, deployment_type, pollutant):
    if deployment_type == 'H':
        #get a list of all the pods (not full file names)
        pod_list = [string.split('_')[0] for string in data_file_list]
        #make a dictionary to put loaded data into, separated by pod names
        pod_data = dict.fromkeys(pod_list)
        data_path = 'Harmonization'
    
    if deployment_type == 'F':
        #get a list of all the pods (not full file names)
        pod_list = [string.split('_')[0] for string in data_file_list]
        #make a dictionary to put loaded data into, separated by pod names
        pod_data = dict.fromkeys(pod_list)
        data_path = 'Field'
        
    if deployment_type == 'C':
        data_path = os.path.join('Colocation', 'Pods')
        pod_data = pd.DataFrame()
        
    
    for i, file in enumerate(data_file_list):
        #get the correct column names list based on the "header_type" in deployment log
        if deployment_type == 'C':
            header_type = deployment_log[(deployment_log['deployment']=='C') & (deployment_log['pollutant']==pollutant)]['header_type'].to_string(index=False)
        else:
            header_type = deployment_log[deployment_log['file_name'] == file]['header_type'].to_string(index=False)
        if header_type not in column_names:
            raise KeyError("Header type in deployment log does not match any column names options")     
            
        #read the individual data file (to be combined after correcting the datetime)
        if os.path.exists(os.path.join(data_path, f'{file}.txt')):
            temp=pd.read_csv(os.path.join(data_path, f'{file}.txt'))
            if len(column_names[header_type]) != len(temp.columns):
                raise KeyError("Number of column names does not match the number of columns in the colocation data.")

            #add column names to the temporary colocation data
            temp.columns=column_names[header_type]
            
            
            if 'datetime' not in temp:
                if 'date' in temp and 'time' in temp:
                    temp['datetime'] = temp['date'] + 'T' + temp['time']
                    temp = temp.drop(['date', 'time'], axis=1)
                else: raise KeyError(f"File {file}  does not include datetime column OR date column and time column. Fix 'column_names' variable")
            
            temp['datetime']=pd.to_datetime(temp['datetime'])
            temp.set_index('datetime',inplace=True)
            
            #crop data based on deployment log
            start = deployment_log[(deployment_log['file_name']==file)]['start'].iat[0]
            end = deployment_log[(deployment_log['file_name']==file)]['end'].iat[0]
            time_removed = (temp.index < start) | (temp.index > end)
            temp=temp[~time_removed]
            
            #correct datetime to the reference data timezone
            timezone_change_from_ref = deployment_log[(deployment_log['file_name']==file)]['timezone_change_from_ref'].iloc[0]
            temp.index=temp.index - pd.to_timedelta(timezone_change_from_ref, unit='h')
            
            if deployment_type == 'C':
                #merge multiple colo files (if applicable)
                if i == 0:
                   pod_data = temp
                else: 
                    pod_data = pd.merge([pod_data, temp], how='outer', on='A')
           
            elif deployment_type == 'H' or deployment_type == 'F':
                pod_name = pod_list[i]
                #either save the pod data in a new dataframe in the field dictionary, or add the data to the preexisting dataframe for that pod 
                #(if there is multiple field files for a pod)
                if isinstance(pod_data[pod_name], pd.DataFrame):
                    pod_data[pod_name] = pd.merge(pod_data[pod_name], temp, how = 'outer', on= 'A')
                else: pod_data[pod_name]=temp 
                
        
        else: 
            print()
            print(f"File {file} listed in deployment log does not exist in folder. This data will be skipped!")
            print()
    
    # Remove None values from the dictionary in place
    if isinstance(pod_data, dict):
        pod_data = {key: value for key, value in pod_data.items() if value is not None}

    return pod_data
    
