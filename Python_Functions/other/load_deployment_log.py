import pandas as pd

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

    expected_columns = ['file_name', 'deployment', 'location', 'pollutant', 'timezone_change_from_ref', 'start', 'end', 'header_type']
    if not all(deployment_log.columns == expected_columns) or len(deployment_log.columns) != len(expected_columns):
        raise KeyError('Deployment log column names are incorrect. Please change the columns to the following: file_name, deployment, location, pollutant,timezone_change_from_ref, start, end, header_type')

    # Specify data types for columns
    dl_dtype = {
        'file_name': 'str',
        'deployment': 'str',
        'location': 'str',
        'pollutant': 'str',
        'timezone_change_from_ref': 'int64',
        'start': 'datetime64[ns]',
        'end': 'datetime64[ns]',
        'header_type': 'str'
    }

    # Convert columns to specified data types
    for col, dtype in dl_dtype.items():
        deployment_log[col] = deployment_log[col].astype(dtype)

    deployment_log['pollutant'][deployment_log['pollutant']=='nan']=''
    deployment_log['location'][deployment_log['location']=='nan']=''

    return deployment_log