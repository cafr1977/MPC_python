import pandas as pd
import os
import numpy as np

def float_converter(value):
    try:
        return float(value)
    except ValueError:
        return pd.NaT

def load_deployment_log():
    # Load deployment from either CSV or Excel file
    try:
        deployment_log = pd.read_csv("deployment_log.csv")
    except FileNotFoundError:
        try:
            deployment_log = pd.read_excel("deployment_log.xlsx")
        except FileNotFoundError:
            # Handle the case when neither CSV nor Excel file is found
            raise FileNotFoundError("Deployment log file not found.")

    expected_columns = ['file_name', 'deployment', 'location', 'pollutant', 'timezone', 'start', 'end', 'header_type']
    extra_columns = [column for column in deployment_log.columns if column not in expected_columns]

    if extra_columns:
        deployment_log.drop(columns=extra_columns, inplace=True)

    if not all(column in deployment_log.columns for column in expected_columns):
        raise KeyError("Deployment log is missing essential columns. Please change the columns to include the following: 'file_name', 'deployment', 'location', 'pollutant', 'timezone', 'start', 'end', 'header_type'")

    # Specify data types for columns
    dl_dtype = {
        'file_name': 'str',
        'deployment': 'str',
        'location': 'str',
        'pollutant': 'str',
        'timezone': 'str',
        'start': 'datetime64[ns]',
        'end': 'datetime64[ns]',
        'header_type': 'str'
    }

    # Convert columns to specified data types
    for col, dtype in dl_dtype.items():
        deployment_log[col] = deployment_log[col].astype(dtype)

    deployment_log['timezone_change_from_ref'] = np.nan

    return deployment_log

def load_data(data_file_list, deployment_log, column_names, deployment_type, pollutant, ref_timezone):

    tz = {
        'MST': -7,
        'PST': -8,
        'CST': -6,
        'EST': -5,
        'MDT': -6,
        'PDT': -7,
        'CDT': -5,
        'EDT': -4,
        'UTC': 0,
        'UTC+1': 1,
        'UTC+2': 2,
        'UTC+3':3,
        'UTC+4': 4,
        'UTC+5': 5,
        'UTC+6': 6,
        'UTC+7': 7,
        'UTC+8': 8,
        'UTC+9': 9,
        'UTC+10': 10,
        'UTC+11': 11,
        'UTC+12': 12,
        'UTC+13': 13,
        'UTC+14': 14,
        'UTC-1': -1,
        'UTC-2': -2,
        'UTC-3': -3,
        'UTC-4': -4,
        'UTC-5': -5,
        'UTC-6': -6,
        'UTC-7': -7,
        'UTC-8': -8,
        'UTC-9': -9,
        'UTC-10': -10,
        'UTC-11': -11,
        'UTC-12': -12
    }

    if deployment_type == 'H':
        # get a list of all the pods (not full file names)
        pod_list = [string.split('_')[0] for string in data_file_list]
        # make a dictionary to put loaded data into, separated by pod names
        pod_data = dict.fromkeys(pod_list)
        data_path = 'Harmonization'

    if deployment_type == 'F':
        # get a list of all the pods (not full file names)
        pod_list = [string.split('_')[0] for string in data_file_list]
        # make a dictionary to put loaded data into, separated by pod names
        pod_data = dict.fromkeys(pod_list)
        data_path = 'Field'

    if deployment_type == 'C':
        data_path = os.path.join('Colocation', 'Pods')
        pod_data = pd.DataFrame()

    for i, file in enumerate(data_file_list):
        print(f'Importing {file}')
        # get the correct column names list based on the "header_type" in deployment log
        if deployment_type == 'C':
            header_type = deployment_log[(deployment_log['file_name'] == file) & (deployment_log['pollutant'] == pollutant)]['header_type'].to_string(index=False)
        else:
            header_type = deployment_log[deployment_log['file_name'] == file]['header_type'].to_string(index=False)
        if header_type not in column_names:
            raise KeyError(f"Header type {header_type} in deployment log does not match any column names options")

        # read the individual data file (to be combined after correcting the datetime)
        if os.path.exists(os.path.join(data_path, f'{file}.txt')):

            #first, check the datetime columns (if they exist)
            if 'datetime' in column_names[header_type]:
                # datetime columns to exclude from the converters
                datetime_columns = ['datetime']
            elif 'date' in column_names[header_type] and 'time' in column_names[header_type]:
                datetime_columns = ['date', 'time']
            else:
                raise KeyError(
                    f"File {file}  does not include datetime column OR date column and time column. Fix 'column_names' variable")

            # Create a dictionary with column names as keys and the custom converter function as values
            converter_dict = {column: float_converter for column in column_names[header_type] if
                              column not in datetime_columns}

            temp = pd.read_csv(
                os.path.join(data_path, f'{file}.txt'),
                header=None, names=column_names[header_type],
                parse_dates=datetime_columns, converters=converter_dict)

            if len(column_names[header_type]) != len(temp.columns):
                raise KeyError("Number of column names does not match the number of columns in the colocation data.")

            if 'date' in temp and 'time' in temp:
                temp['datetime'] = temp['date'] + 'T' + temp['time']
                temp = temp.drop(['date', 'time'], axis=1)

            temp['datetime'] = pd.to_datetime(temp['datetime'])
            temp.set_index('datetime', inplace=True)

            # crop data based on deployment log
            start = deployment_log[(deployment_log['file_name'] == file)]['start'].iat[0]
            end = deployment_log[(deployment_log['file_name'] == file)]['end'].iat[0]
            time_removed = (temp.index < start) | (temp.index > end)
            temp = temp[~time_removed]

            # correct datetime to the reference data timezone
            pod_timezone = deployment_log[(deployment_log['file_name'] == file)]['timezone'].iloc[0]

            timezone_change_from_ref = tz[pod_timezone] - tz[ref_timezone]
            deployment_log.loc[(deployment_log['file_name'] == file),'timezone_change_from_ref'] =  timezone_change_from_ref
            temp.index = temp.index - pd.to_timedelta(timezone_change_from_ref, unit='h')

            #old way of doing timezone correction that wasn't based on actual timezones:
            #timezone_change_from_ref = deployment_log[(deployment_log['file_name'] == file)]['timezone_change_from_ref'].iloc[0]
            #temp.index = temp.index - pd.to_timedelta(timezone_change_from_ref, unit='h')

            if deployment_type == 'C':
                # merge multiple colo files (if applicable)
                if i == 0:
                    pod_data = temp
                else:
                    pod_data = pd.concat([pod_data, temp], axis=0, join='outer')

            elif deployment_type == 'H' or deployment_type == 'F':
                pod_name = pod_list[i]
                # either save the pod data in a new dataframe in the field dictionary, or add the data to the preexisting dataframe for that pod
                # (if there is multiple field files for a pod)
                if isinstance(pod_data[pod_name], pd.DataFrame):
                    pod_data[pod_name] = pd.concat([pod_data[pod_name], temp], axis=0, join='outer')
                else:
                    pod_data[pod_name] = temp


        else:
            print()
            print(f"File {file} listed in deployment log does not exist in folder. This data will be skipped!")
            print()

    # Remove None values from the dictionary in place
    if isinstance(pod_data, dict):
        pod_data = {key: value for key, value in pod_data.items() if value is not None}

    return pod_data, deployment_log

def field_location(Y_field_df, deployment_log):
    Y_field_df['location'] = str()
    field_deployment_log = pd.DataFrame(deployment_log[deployment_log['deployment'] == 'F']).set_index('file_name')
    field_deployment_log['pod'] = [filename.split('_')[0] for filename in field_deployment_log.index]

    for file in field_deployment_log.index:
        # unconvert the pod datetime so that we can compare it to start/stop before the reference timezone change
        timezone_change_from_ref = field_deployment_log.loc[file, 'timezone_change_from_ref']

        mask = (Y_field_df['pod'] == field_deployment_log.loc[file, 'pod']) & \
               (Y_field_df['datetime'] + pd.to_timedelta(timezone_change_from_ref, unit='h') > field_deployment_log.loc[
                   file, 'start']) & \
               (Y_field_df['datetime'] + pd.to_timedelta(timezone_change_from_ref, unit='h') < field_deployment_log.loc[
                   file, 'end'])
        Y_field_df.loc[mask, 'location'] = field_deployment_log.loc[file, 'location']

    return Y_field_df