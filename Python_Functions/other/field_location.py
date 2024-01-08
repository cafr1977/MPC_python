# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 11:33:20 2024

@author: cfris
"""

import pandas as pd 

def field_location(Y_field_df, deployment_log):
    Y_field_df['location']=str()
    field_deployment_log = pd.DataFrame(deployment_log[deployment_log['deployment']=='F']).set_index('file_name')
    field_deployment_log['pod'] = [filename.split('_')[0] for filename in field_deployment_log.index]
    
    for file in field_deployment_log.index:
        
        #unconvert the pod datetime so that we can compare it to start/stop before the reference timezone change
        timezone_change_from_ref = field_deployment_log.loc[file,'timezone_change_from_ref']
        
        mask = (Y_field_df['pod'] == field_deployment_log.loc[file, 'pod']) & \
        (Y_field_df['datetime']  + pd.to_timedelta(timezone_change_from_ref, unit='h') > field_deployment_log.loc[file, 'start']) & \
        (Y_field_df['datetime']  + pd.to_timedelta(timezone_change_from_ref, unit='h') < field_deployment_log.loc[file, 'end'])
        Y_field_df.loc[mask, 'location'] = field_deployment_log.loc[file, 'location']
    
    return Y_field_df