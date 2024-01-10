# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 11:21:47 2023

@author: cfris
"""

# Caroline's attempt at a python MPC that fixes current problems in Matlab MPC
# and improves machine learning capabilities
# for one-hop only

########
#Packages to import
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import importlib
import joblib

#Other packages that must be installed prior to running:
    #atmos (for humidity conversion)
    #numpy
    #xlsxwriter
    

#########
#initiate settings
settings={}

#close previous figures
plt.close('all')

#Variable to change for your analysis
settings['ref_file_name'] = "InnerPort_101023_122023_voc" #Name of the reference CSV or XLSX file you are using (do not type .csv for the name)
settings['pollutant']='TVOC' #make sure this matches the column name in the reference data file (CSV or XLSX)
settings['unit'] = 'ppb' #concentration units of the target pollutant (for plot labels)
settings['time_interval'] = 5 #time averaging in minutes. needs to be at least as high as the time resolution of the data
settings['retime_calc'] = "median" #How the time averaging is calculated. Options are median and mean right now and are the same for pod and ref
settings['sensors_included'] = ['Temperature','Humidity','Fig2600','Fig2602','Fig3','Fig4']
settings['scaler'] = StandardScaler() #How the data is scaled. StandardScaler is mean zero and st dev 1
settings['t_warmup'] = 45 #warm up period in minutes 
settings['test_percentage'] = 0.2 #what percentage of data goes into the test set, usually 0.2 or 0.3
settings['traintest_split_type'] = 'mid_end_split' #how the data is split into train and test
# 'mid_end_split' takes % of middle data and % of data at end to form test set
# 'end_test' takes % of end data to form test set

settings['models']=['lin_reg','random_forest','lasso','ridge'] #which models are run on the data
#'lin_reg','random_forest','lasso','ridge'

settings['preprocess'] = ["temp_C_2_K","hum_rel_2_abs","rmv_warmup"]
# temp_C_2_K": converts temperature from C to K, required for HumRel2Abs to run
#hum_rel_2_abs: converts humidity from relative to absolute
#rmv_warmup: Removes the first 45 minutes of data, as well as 45 minutes of data after the pod cuts out for more than 10 min
#add_time_elapsed: Adds a column to the X dataframe of how much time has elapsed since the first data point. Helpful for drift
#fig_ratio: adds a column to the X dataframe that is fig 2600 divided by fig 2602

#Preprocessing time sort, remove NaN, remove 999 are done automatically to avoid errors.

settings['colo_plot_list'] = ['colo_residual','colo_timeseries','colo_scatter','colo_stats_plot'] # plots to plot and save

#Column_names is a dictionary of column name lists. The name of the list corresponds to the deployment log "header_type" column
settings['column_names'] = {'3.1.0':["datetime","Volts", "Fig2600", "Fig2602","Fig3","Fig3heater", "Fig4","Fig4heater", "Misc2611",
                 "PID", "CO_aux","CO_main", "CO2", "Temperature", "Pressure", "Humidity",
                 "Quad1C1", "Quad1C2","Quad1C3","Quad1C4", "Quad2C1", "Quad2C2","Quad2C3","Quad2C4",
                 "MQ131","PM 10 ENV", "PM 25 ENV", "PM 100 ENV", "PM 03 um", "PM 05 um", "PM 10 um",
                 "PM 25 um", "PM 50 um", "PM 100 um",'OPC1','OPC2','OPC3','OPC4','OPC5','WS_mph','WD','unk'],
               
               '3.2.0':["datetime","Volts", "Figaro2600", "Figaro2602","Figaro3","Fig3heater", "Figaro4","Fig4heater", "Misc2611",
                                "PID", "CO_aux","CO_main", "CO2", "Temperature", "Pressure", "Humidity", "Quad1C1", "Quad1C2",
                                "Quad1C3","Quad1C4","Quad2C1", "Quad2C2","Quad2C3","Quad2C4",
                                "MQ131","PM 10 ENV", "PM 25 ENV", "PM 100 ENV", "PM 03 um", "PM 05 um", "PM 10 um",
                                "PM 25 um", "PM 50 um", "PM 100 um",'OPC1','OPC2','OPC3','OPC4','OPC5','WS_mph','WD','unk']}

###############

#Begin code

#save the settings for harmonization_field run


#Create output folder
# Get the current time as YYMMDDHHss
current_time = datetime.now().strftime('%y%m%d%H%M%S')
# Create the output folder name
output_folder_name = f'Output_{current_time}'
# Create the output folder
os.makedirs(os.path.join('Outputs', output_folder_name))

del current_time

# Load deployment log
from Python_Functions.other.load_deployment_log import load_deployment_log
deployment_log = load_deployment_log()

#get the earliest start time (for time elapsed)
if "add_time_elapsed" in settings['preprocess']:
    #get the earliest time in the deployment log
    settings['earliest_time'] = deployment_log['start'].min()

#Load colocation data
colo_file_list = deployment_log[(deployment_log['deployment']=='C') & (deployment_log['pollutant']==settings['pollutant'])]['file_name'] #list of all colo files to combine

#make sure there just one colocation pod
settings['colo_pod_name'] = [string.split('_')[0] for string in colo_file_list]
if len(colo_file_list) == 0:
    raise KeyError('No colocation files listed in deployment log for given pollutant.')   
if len(settings['colo_pod_name']) != 1:
    raise KeyError('Run cannot continue because there is more than one unique colocation pod listed in the deployment log for the pollutant of interest.')

#load pod data
from Python_Functions.other import load_data
colo_pod_data = load_data.load_data(colo_file_list,deployment_log,settings['column_names'], 'C',settings['pollutant'])

if colo_pod_data.empty:
    raise AssertionError("No colocation pod data was found in the Colocation Pod folder that matched the deployment log. Stopping execution.")

# Load reference data from either CSV or Excel file
# Check if the CSV file exists, and if not, try loading the Excel file
try:
    ref_data = pd.read_csv(os.path.join("Colocation", "Reference", f"{settings['ref_file_name']}.csv"))
except FileNotFoundError:
    try:
        ref_data = pd.read_excel(os.path.join("Colocation", "Reference", f"{settings['ref_file_name']}.xlsx"))
    except FileNotFoundError:
        # Handle the case when neither CSV nor Excel file is found
        raise FileNotFoundError("Reference data file not found")


#set index of ref data as the datetime column
ref_data = ref_data.rename(columns={ref_data.columns[0]: 'datetime'})
ref_data['datetime']=pd.to_datetime(ref_data['datetime'])
ref_data.set_index('datetime',inplace=True)

#only include the ref species of interest 
if isinstance(ref_data, pd.DataFrame):
    ref_data = ref_data[settings['pollutant']]

# Rename the pollutant column to differentiate from pod data
ref_data = ref_data.rename(settings['pollutant'] + '_ref')

#apply preprocess (rmv 999 and NaN, rmv warm up, humid and temp conversion)
#scaling happens in the ML section instead of the preprocess section here BECAUSE WE WANT TO SCALE ONLY THE TRAINING DATA

#colo pod preprocessing
from Python_Functions.preprocess import preprocessing_func
colo_pod_data = preprocessing_func.preprocessing_func(colo_pod_data, settings['sensors_included'], settings['t_warmup'], settings['preprocess'])

#ref data preprocessing
#change 999 to NA
ref_data.replace(999, pd.NA, inplace=True)

#make sure all columns are numeric to avoid errors later:
ref_data = pd.to_numeric(ref_data, errors='coerce').astype(float)

# Drop rows with NaN values
ref_data.dropna(inplace=True)

# Combine ref and colo pod data into one data frame using time_interval and retime_calc
if not isinstance(settings['time_interval'], str): #first check that time_interval is a string
    settings['time_interval'] = str(settings['time_interval'])

#time average and align the colocation and reference data    
if settings['retime_calc']=='median':
    data_combined= pd.concat([colo_pod_data, ref_data], axis=1).resample(settings['time_interval']+'T').median()
if settings['retime_calc']=='mean':
    data_combined= pd.concat([colo_pod_data, ref_data], axis=1).resample(settings['time_interval']+'T').mean()
  
#rename the reference column to the pollutant name    
data_combined.rename(columns={data_combined.columns[-1]:settings['pollutant']+'_ref'},inplace=True)
data_combined.dropna(inplace=True)

#add some preprocessing that has to happen after the data is aligned, and therefore can't happen in the "preprocessing" function
if "add_time_elapsed" in settings['preprocess']:
    from Python_Functions.preprocess import add_time_elapsed
    data_combined = add_time_elapsed.add_time_elapsed(data_combined, settings['earliest_time'])

if "fig_ratio" in settings['preprocess']:
    from Python_Functions.preprocess import fig_ratio
    data_combined = fig_ratio.fig_ratio(data_combined)         
     

#begin ML 
#create X and y dataframes
X=data_combined.drop([settings['pollutant'] + '_ref'],axis=1)
y=data_combined[settings['pollutant'] + '_ref']

#if using interaction terms in the model, this is where you add it
if "interaction_terms" in settings['preprocess']:
    from Python_Functions.preprocess import interaction_terms
    X = interaction_terms.interaction_terms(X)       

#delete data_combined
del data_combined

#Train and Test split
if settings['traintest_split_type'] == 'end_test':
    from Python_Functions.test_train_split import end_test
    X_train, y_train, X_test, y_test = end_test.end_test(settings['test_percentage'], X, y)

elif settings['traintest_split_type'] == 'mid_end_split':
    from Python_Functions.test_train_split import mid_end_split
    X_train, y_train, X_test, y_test = mid_end_split.mid_end_split(settings['test_percentage'], X, y)
    
else: 
    raise KeyError('Invalid traintest_split_type, run is ended')
    
#Scale the data using the technique specified in "scaler"
X_train_std = settings['scaler'].fit_transform(X_train)
X_test_std = settings['scaler'].transform(X_test)
X_std = pd.DataFrame(data=settings['scaler'].transform(X),columns=X.columns,index=X.index)
X_std_values = X_std.values
X_train = X_train_std
X_test = X_test_std

#save out variables for later analysis
X_std.to_csv(os.path.join('Outputs', output_folder_name, 'colo_X_std.csv'))
y.to_csv(os.path.join('Outputs', output_folder_name, 'colo_y_reference.csv'))
X.to_csv(os.path.join('Outputs', output_folder_name, 'colo_X.csv'))

#delete unneeded variables
del y, X, X_train_std, X_test_std

####Begin running models

#first, establish a dataframe to save model statistics in
model_stats=pd.DataFrame(index=settings['models'], columns = ['Training_R2','Training_RMSE','Testing_RMSE','Training_MBE','Testing_MBE'])
models_folder = "Python_Functions." + "models"
for i, model_name in enumerate(settings['models']):
    # Import the module dynamically
    model_module = importlib.import_module(f"{models_folder}.{model_name}")
    # Get the function from the module
    model_func = getattr(model_module, model_name)
    # Call the function to apply the model_name model
    model_stats, y_train_predicted, y_test_predicted, y_predicted, current_model = model_func(X_train, y_train, X_test, y_test,X_std_values,model_name,settings['pollutant'],output_folder_name,model_stats)

    #save out the model and the y predicted 
    y_predicted = pd.DataFrame(data = y_predicted, columns = [settings['pollutant']], index = X_std.index)
    y_predicted.to_csv(os.path.join('Outputs', output_folder_name, f'{model_name}_colo_y_predicted.csv'))
    joblib.dump(current_model, os.path.join('Outputs', output_folder_name, f'{model_name}_model.joblib'))
     
#plotting of modelled data
    if 'colo_timeseries' in settings['colo_plot_list']:
        from Python_Functions.plots import colo_timeseries
        colo_timeseries.colo_timeseries(y_train, y_train_predicted, y_test, y_test_predicted, settings['pollutant'], model_name, output_folder_name)
        
    if 'colo_scatter' in settings['colo_plot_list']:
        from Python_Functions.plots import colo_scatter
        colo_scatter.colo_scatter(y_train, y_train_predicted, y_test, y_test_predicted, settings['pollutant'], model_name, output_folder_name)
        
    if 'colo_residual' in settings['colo_plot_list']:
       from Python_Functions.plots import colo_residual
       colo_residual.colo_residual(y_train, y_train_predicted, y_test, y_test_predicted, X_train, X_test, settings['pollutant'], model_name, output_folder_name,X_std.columns) 

#save out the model for later analysis and use in field data
model_stats.to_csv(os.path.join('Outputs', output_folder_name, 'colo_model_stats.csv'), index = True)

#stats_plot plots the R2, RMSE, and MBE of train and test data as a bar graph
if "colo_stats_plot" in settings['colo_plot_list']:
    from Python_Functions.plots import colo_stats_plot 
    colo_stats_plot.colo_stats_plot(settings['models'], model_stats, settings['pollutant'],output_folder_name)
    
#save out settings for future reference
joblib.dump(settings, os.path.join('Outputs', output_folder_name, 'run_settings.joblib'))