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
from sklearn.preprocessing import StandardScaler  #pip install scikit-learn
from datetime import datetime
import importlib
import joblib

# Other packages that must be installed prior to running:
# atmos (for humidity conversion)
# numpy
# xlsxwriter
# seaborn

#Begin code
print('Beginning "MPC Colocation"')

#packages from python_functions folder
from Python_Functions import preprocessing_func
from Python_Functions import test_train_split_func
from Python_Functions import data_loading_func
from Python_Functions import plotting_func
from Python_Functions import weighting_grid


#########
#initiate settings
settings={}

#close previous figures
plt.close('all')

#Variable to change for your analysis
settings['colo_run_name'] = ''  #Name of the outputs file. If you leave it blank inside the quotes, the output folder will be named with current time
# ^^ If you want the outputs folder to just be named with current datetime, set settings['run_name'] = '' (YOU NEED THE APOSTROPHES/QUOTES)
settings['ref_file_name'] = 'InnerPort_101023_031824_voc' #Name of the reference CSV or XLSX file you are using (do not type .csv for the name)
settings['ref_timezone'] = 'PST'
settings['pollutant']='TVOC' #make sure this matches the column name in the reference data file (CSV or XLSX)
settings['unit'] = 'ppb' #concentration units of the target pollutant (for plot labels)
settings['time_interval'] = 60 #time averaging in minutes. needs to be at least as high as the time resolution of the reference data
settings['retime_calc'] = "median" #How the time averaging is calculated. Options are median and mean right now and are the same for pod and ref
settings['sensors_included'] = ["Fig2600",'Temperature','Humidity'] #list the sensors you want in the model (both pollutant and environmental, like temperature or humidity)
settings['scaler'] = StandardScaler() #How the data is scaled. StandardScaler is mean zero and st dev 1
settings['t_warmup'] = 120 #warm up period in minutes
settings['test_percentage'] = 0.2 #what percentage of data goes into the test set, usually 0.2 or 0.3
settings['traintest_split_type'] = 'mid_end_split' #how the data is split into train and test
#start_end_split takes the % of the data at the start and % of data at the end to form test set
# 'mid_end_split' takes % of middle data and % of data at end to form test set
# 'end_test' takes % of end data to form test set

settings['colo_plot_list'] = ['colo_timeseries','colo_scatter', 'colo_stats_plot','colo_residual'] # plots to plot and save
#plot options:
# colo_timeseries: timeseries of predicted Y and reference Y
# colo_scatter: scatter plot of predicted vs. reference Y
# colo_stats_plot: bar chart of the R2, RMSE, MBE of train and test data
# colo_residual: residuals plotted over each sensor used in model
# corr_heatmap: heat map of the correlations between each sensor column and the reference data
# feature_importance: bar plot of the relative importance of the features used in machine learning models. does not work for linear models.


settings['models']=['lin_reg','random_forest'] #which models are run on the data
#regular options: 'lin_reg','lasso','ridge','random_forest','adaboost', 'gradboost', 'svr_'
#svr takes a long time
#adaboost is usually a classification model, so it doesn't work great
#lasso, ridge, random forest, adaboost, gradboost and svr are all common machine learning models.

#peak weighting model options: 'rf_qw_tuned', 'svr_qw_tuned' (i haven't made the others because it doesn't seem helpful, can add easily though lmk.

settings['preprocess'] = ["rmv_warmup",'temp_C_2_K','hum_rel_2_abs']
# temp_C_2_K": converts temperature from C to K, required for HumRel2Abs to run
#hum_rel_2_abs: converts humidity from relative to absolute
#rmv_warmup: Removes the first settings['t_warmup'] minutes of data, as well as settings['t_warmup'] minutes of data after the pod cuts out for more than 10 min
#add_time_elapsed: Adds a column to the X dataframe of how much time has elapsed since the first data point. Helpful for drift
#'fig2600_2602_ratio', 'fig2600_3_ratio','fig4_2602_ratio','fig4_3_ratio': adds a column to the X dataframe that is fig# divided by fig # (figaro numbers listed in the name)
    #### ^^ these figaros will only work if you name the columns as fig2600, fig 2602, fig3, and fig4 in settings['column_names']
#binned_resample: resample the data so that there are the same number of data points per "bin" (number of bins set with settings['n_bins'])
#resample_quartile: removes a percentage of values from the reference data in the quartile listed in settings['quartiles_to_resample'] (percentage based on settings['quartiles_downsampling_rate'] )
#"rmv_negative_CO_aux": Filters out negative CO values (not clear why they are occuring)
#interaction_terms: add interaction terms to the X matrix. a column is added for each feature multiplied with each other feature.
#Preprocessing time sort, remove NaN, remove 999 are done automatically to avoid errors.


#Sub settings for resampling/weighting preprocessing functions
settings['quartiles_to_resample'] = ['first']   ##which quantiles you want to downsample from if applying 'resample_quartile' in 'preprocess
settings['quartiles_downsampling_rate'] = 0.6   ## If using 'resample_quartile', choose a downsampling rate between 0-1 (e.g., keeping 70% of instances within the lower quartile)
settings['n_bins']= 5   ## If using binned_resample, choose how many bins to split the data into.
settings['binned_resample_binnum_multiplier'] = 2  #The number of samples per bin after resampling is set as 1/n_bins. However, this can be adjusted by mulitiplying 1/n by 'binned_resample_binnum_multiplier' if you don't want to remove so much data
settings['weighting_percentile'] = [99.5,99.9] #list which percentiles to test for weighting. All data points in that percentile or higher will be weighted higher than those below.
settings['weighting_weight'] = [10, 15, 20] #list what weights to test for a weighted model. All points above the percentile will be given this weight. Those below the percentile will ahve a weight of 1.

#Column_names is a dictionary of column names lists that will be applied to pod data.
# The name of the list corresponds to the deployment log "header_type" column
# besides datetime (or date and time), ALL columns must be numeric.
#if you want to have data columns that are strings (text), discuss with caroline.
settings['column_names'] = {'3.1.0':
                                ["datetime", "Volts", "Fig2600", "Fig2602","Fig3","Fig3heater", "Fig4","Fig4heater",
                                 "PID","Mics2611", "CO_aux","CO_main", "CO2", "Temperature", "Pressure", "Humidity",
                                "Quad1C1", "Quad1C2","Quad1C3","Quad1C4", "Quad2C1", "Quad2C2","Quad2C3","Quad2C4",
                                "MQ131","PM 10 ENV", "PM 25 ENV", "PM 100 ENV", "PM 03 um", "PM 05 um", "PM 10 um",
                                "PM 25 um", "PM 50 um", "PM 100 um",'OPC1','OPC2','OPC3','OPC4','OPC5','WS_mph','WD','unk'],

                            '3.1.2_opc':
                                ["datetime", "Volts", "Fig2600", "Fig2602", "Fig3", "Fig3heater", "Fig4", "Fig4heater",
                                 "PID","Mics2611", "CO_aux", "CO_main", "CO2",
                                 "Temperature", "Pressure", "Humidity", "QS1_Aux", "QS1_Main", "QS2_Aux", "QS2_Main",
                                 "QS3_Aux", "QS3_Main", "QS4_Aux", "QS4_Main", 'WS_mph', 'WD',
                                 "MQ131", "PM 10 ENV", "PM 25 ENV", "PM 100 ENV", "PM 03 um", "PM 05 um", "PM 10 um",
                                 "PM 25 um", "PM 50 um", "PM 100 um", 'OPC_Bin1_', 'OPC_Bin2', 'OPC_Bin3', 'OPC_Bin4',
                                 'OPC_Bin5', 'OPC_Bin6', 'OPC_Bin7', 'OPC_Bin8',
                                 'OPC_Bin9', 'OPC_Bin10', 'OPC_Bin11', 'OPC_Bin12', 'OPC_Bin13', 'OPC_Bin14',
                                 'OPC_Bin15', 'OPC_Bin16', 'OPC_SampPer',
                                 'OPC_FlowRate', 'OPC_PM10_', 'OPC_PM25', 'OPC_PM100', 'unk']
    ,
                            '3.1.2':
                                ["datetime","Volts", "Fig2600", "Fig2602","Fig3","Fig3heater", "Fig4","Fig4heater",
                                "PID", "Mics2611", "CO_aux","CO_main", "CO2", "Temperature", "Pressure", "Humidity", "Quad1C1", "Quad1C2","Quad1C3","Quad1C4",
                                "Quad2C1", "Quad2C2","Quad2C3","Quad2C4",'WS_mph','WD',
                                "MQ131","PM 10 ENV", "PM 25 ENV", "PM 100 ENV", "PM 03 um", "PM 05 um", "PM 10 um",
                                "PM 25 um", "PM 50 um", "PM 100 um",'OPC1','OPC2','OPC3','OPC4','OPC5', 'unk']

                            }

###############

## Begin actual code

#check for contradictions in preprocessing
if 'resample_quartile' in settings['preprocess'] and 'binned_resample' in settings['preprocess']:
    raise ValueError("binned_resample and resample_quartile are both present in settings['preprocess']."
                     "Only one resample technique is allowed.")

### if rf_qw_tuned in models, then it should be the only model!
if 'rf_qw_tuned' in settings['models'] or 'svr_qw_tuned' in settings['models'] and len(settings['models']) != 1:
    raise ValueError("rf_qw_tuned must be used alone in settings['models']. You must test other models separately.")


#Create output folder
if settings['colo_run_name'] == '':
    # Get the current time as YYMMDDHHss
    current_time = datetime.now().strftime('%y%m%d%H%M%S')
    # Create the output folder name
    output_folder_name = f'Output_{current_time}'
    del current_time
else: output_folder_name = f'Output_{settings["colo_run_name"]}'

# Check if the directory already exists
if os.path.exists(os.path.join('Outputs', output_folder_name)):
    raise FileExistsError(f"The Output folder '{output_folder_name}' already exists. Please choose a different output folder name or use the current time option (settings['colo_run_name'] == '').")
else:
    # If the directory doesn't exist, create it
    os.makedirs(os.path.join('Outputs', output_folder_name))

# Load deployment log
print('Loading deployment log...')
deployment_log = data_loading_func.load_deployment_log()

#get the earliest start time (for time elapsed)
if "add_time_elapsed" in settings['preprocess']:
    #get the earliest time in the deployment log
    settings['earliest_time'] = deployment_log['start'].min()

#Load colocation data
colo_file_list = deployment_log[(deployment_log['deployment']=='C') & (deployment_log['pollutant']==settings['pollutant'])]['file_name'] #list of all colo files to combine

#make sure there just one colocation pod
settings['colo_pod_name'] = list(set([string.split('_')[0] for string in colo_file_list]))
if len(settings['colo_pod_name']) == 0:
    raise KeyError('No colocation files listed in deployment log for given pollutant.')   
if len(settings['colo_pod_name']) != 1:
    raise KeyError('Run cannot continue because there is more than one unique colocation pod listed in the deployment log for the pollutant of interest.')

#load pod data
print('Loading colocation pod data...')
colo_pod_data, deployment_log = data_loading_func.load_data(colo_file_list,deployment_log,settings['column_names'], 'C',settings['pollutant'], settings['ref_timezone'])

if colo_pod_data.empty:
    raise AssertionError("No colocation pod data was found in the Colocation Pod folder that matched the deployment log. Stopping execution.")

# Load reference data from either CSV or Excel file
# Check if the CSV file exists, and if not, try loading the Excel file
print('Loading reference data...')
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

#colo pod preprocessing
# FYI: scaling happens in the ML section instead of the preprocess section here BECAUSE WE WANT TO SCALE ONLY THE TRAINING DATA
print('Preprocessing colocation pod and reference data...')
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

#remove any duplicate rows (based on time). This line will keep the first instance.
colo_pod_data = colo_pod_data[~colo_pod_data.index.duplicated(keep='first')]
ref_data = ref_data[~ref_data.index.duplicated(keep='first')]

#time average and align the colocation and reference data
print('Re-timing colocation pod and reference data...')
if settings['retime_calc']=='median':
    data_combined= pd.concat([colo_pod_data, ref_data], axis=1).resample(settings['time_interval']+'T').median()
if settings['retime_calc']=='mean':
    data_combined= pd.concat([colo_pod_data, ref_data], axis=1).resample(settings['time_interval']+'T').mean()
  
#rename the reference column to the pollutant name    
data_combined.rename(columns={data_combined.columns[-1]:settings['pollutant']+'_ref'},inplace=True)
data_combined.dropna(inplace=True)

#add some preprocessing that has to happen after the data is aligned, and therefore can't happen in the "preprocessing" function
if "add_time_elapsed" in settings['preprocess']:
    data_combined = preprocessing_func.add_time_elapsed(data_combined, settings['earliest_time'])

#create figaro ratios if using:
if 'fig2600_2602_ratio' in settings['preprocess']:
    data_combined = preprocessing_func.fig2600_2602_ratio(data_combined)

if 'fig2600_3_ratio' in settings['preprocess']:
    data_combined = preprocessing_func.fig2600_3_ratio(data_combined)

if 'fig3_2602_ratio' in settings['preprocess']:
    data_combined = preprocessing_func.fig3_2602_ratio(data_combined)

if 'fig4_2602_ratio' in settings['preprocess']:
    data_combined = preprocessing_func.fig4_2602_ratio(data_combined)

if 'fig4_3_ratio' in settings['preprocess']:
    data_combined = preprocessing_func.fig4_3_ratio(data_combined)

#create the correlation heat map here is if is listed in the colo plot list:
if 'corr_heatmap' in settings['colo_plot_list']:
    plotting_func.corr_heatmap(data_combined, output_folder_name)


#begin ML
print('Initializing models...')
#create X and y dataframes
X=data_combined.drop([settings['pollutant'] + '_ref'],axis=1)
y=data_combined[settings['pollutant'] + '_ref']

#if using interaction terms in the model, this is where you add them in:
if "interaction_terms" in settings['preprocess']:
    X = preprocessing_func.interaction_terms(X)

#delete data_combined
#del data_combined

#Train and Test split
if settings['traintest_split_type'] == 'end_test':
    X_train, y_train, X_test, y_test = test_train_split_func.end_test(settings['test_percentage'], X, y)

elif settings['traintest_split_type'] == 'mid_end_split':
    X_train, y_train, X_test, y_test = test_train_split_func.mid_end_split(settings['test_percentage'], X, y)

elif settings['traintest_split_type'] == 'start_end_split':
    X_train, y_train, X_test, y_test = test_train_split_func.start_end_split(settings['test_percentage'], X, y)

else: 
    raise KeyError('Invalid traintest_split_type, run is ended')

#Bin-based downsampling happens here, only on training data, if included in preprocessing
if "binned_resample" in settings['preprocess']:
    X_train, y_train = preprocessing_func.binned_resample(X_train, y_train, settings['n_bins'], settings['binned_resample_binnum_multiplier'] )

#Resampling based on quartiles happens here, only on training data, if included in preprocessing
if "resample_quartile" in settings['preprocess']:
    for quartile in settings['quartiles_to_resample']:
        X_train, y_train = preprocessing_func.resample_quartile(X_train, y_train, quartile, settings['quartiles_downsampling_rate'])


#Scale the data using the technique specified in "scaler"
X_train_std = settings['scaler'].fit_transform(X_train)
X_test_std = settings['scaler'].transform(X_test)
X_std = pd.DataFrame(data=settings['scaler'].fit_transform(X),columns=X.columns,index=X.index)
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

#if using weighted models, proceed with the following code:
if settings['models']== ['rf_qw_tuned'] or settings['models']== ['svr_qw_tuned']:  #if using a weighted tuning model
    # first, establish a dataframe to save model statistics in
    model_stats = pd.DataFrame(columns=['Training_R2','Testing_R2', 'Training_RMSE', 'Testing_RMSE', 'Training_MBE', 'Testing_MBE'])

    for p in settings['weighting_percentile']:
        for w in settings['weighting_weight']:
            p_str = str(p)
            w_str = str(w)
            model_name = f'rf_p_{p_str}_w_{w_str}'
            # Appending an empty row to model stats with the model name index
            model_stats.loc[model_name] = [None] * len(model_stats.columns)

            if settings['models'] == ['rf_qw_tuned']:
                print(f'Fitting colocation pod data to reference data using weighted random forest with p= {p_str} and w= {w_str}...')
                model_stats, y_train_predicted, y_test_predicted, y_predicted, current_model = weighting_grid.rf_qw_tuned(X_train, y_train,
                                                                                                       X_test, y_test,
                                                                                                       X_std, p,
                                                                                                       w, model_name,
                                                                                                       model_stats)
            elif settings['models'] == ['svr_qw_tuned']:
                print(f'Fitting colocation pod data to reference data using weighted SVR with p= {p_str} and w= {w_str}...')
                model_stats, y_train_predicted, y_test_predicted, y_predicted, current_model = weighting_grid.svr_qw_tuned(
                    X_train, y_train,
                    X_test, y_test,
                    X_std, p,
                    w, model_name,
                    model_stats)

            # save out the model and the y predicted
            y_predicted = pd.DataFrame(data=y_predicted, columns=[settings['pollutant']], index=X_std.index)
            y_predicted.to_csv(os.path.join('Outputs', output_folder_name, f'{model_name}_colo_y_predicted.csv'))
            joblib.dump(current_model, os.path.join('Outputs', output_folder_name, f'{model_name}_model.joblib'))

            plotting_func.colo_plots_series(settings['colo_plot_list'], y_train, y_train_predicted, y_test, y_test_predicted, settings['pollutant'], model_name,
                      output_folder_name, settings['colo_run_name'], settings['unit'], current_model, list(X_std.columns), X_train, X_test)

    #reset the models list so it includes each combo of p and w
    settings['models'] = list(model_stats.index)

else:
    #if not using weighted models, proceed with this code:
    #first, establish a dataframe to save model statistics in
    model_stats=pd.DataFrame(index=settings['models'], columns = ['Training_R2','Testing_R2','Training_RMSE','Testing_RMSE','Training_MBE','Testing_MBE'])
    #models_folder = "Python_Functions." + "models"
    for i, model_name in enumerate(settings['models']):
        print(f'Fitting colocation pod data to reference data using model {model_name}...')
        # Import the module dynamically
        model_module = importlib.import_module('Python_Functions.colo_model_func')
        # Get the function from the module
        model_func = getattr(model_module, model_name)
        # Call the function to apply the model_name model
        model_stats, y_train_predicted, y_test_predicted, y_predicted, current_model = model_func(X_train, y_train, X_test, y_test,X_std_values,model_name,model_stats)

        #save out the model and the y predicted
        y_predicted = pd.DataFrame(data = y_predicted, columns = [settings['pollutant']], index = X_std.index)
        y_predicted.to_csv(os.path.join('Outputs', output_folder_name, f'{model_name}_colo_y_predicted.csv'))
        joblib.dump(current_model, os.path.join('Outputs', output_folder_name, f'{model_name}_model.joblib'))

    #plotting of modelled data
        plotting_func.colo_plots_series(settings['colo_plot_list'], y_train, y_train_predicted, y_test,
                                        y_test_predicted, settings['pollutant'], model_name,
                                        output_folder_name, settings['colo_run_name'], settings['unit'], current_model,
                                        list(X_std.columns), X_train, X_test)



#save out the model for later analysis and use in field data
model_stats.to_csv(os.path.join('Outputs', output_folder_name, 'colo_model_stats.csv'), index = True)

#stats_plot plots the R2, RMSE, and MBE of train and test data as a bar graph
if "colo_stats_plot" in settings['colo_plot_list']:
    plotting_func.colo_stats_plot(settings['models'], model_stats, settings['pollutant'],output_folder_name, settings['colo_run_name'])
    
#save out settings for future reference
joblib.dump(settings, os.path.join('Outputs', output_folder_name, 'run_settings.joblib'))
