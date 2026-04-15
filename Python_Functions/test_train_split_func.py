import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import random

def check_data(X,y):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    return X, y

def mid_end_split(test_percentage, X, y):
    X, y = check_data(X, y)

    total_length = len(X)
    # Define the size of the middle and end portions of the test set (half of the total test percentage each)
    chunk_percentage = test_percentage / 2
    middle_test_size = end_test_size = round(chunk_percentage * total_length)

    # Calculate the start and end indices for the middle portion of the test set
    middle_test_start = total_length // 2 - middle_test_size // 2
    middle_test_end = middle_test_start + middle_test_size

    # Calculate the start and end indices for the end portion of the test set
    end_test_start = total_length - end_test_size
    end_test_end = total_length

    # Define the indices for the test set
    test_indices = np.concatenate([np.arange(middle_test_start, middle_test_end),
                                   np.arange(end_test_start, end_test_end)])

    # Define the indices for the train set (complement of the test set)
    train_indices = np.setdiff1d(np.arange(total_length), test_indices)

    # Use the indices to create the train and test sets
    X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
    X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]

    TT = pd.DataFrame(columns=['Test/Train'], index=X.index)

    TT.iloc[train_indices, 0] = 'Train'
    TT.iloc[test_indices, 0] = 'Test'

    return X_train, y_train, X_test, y_test, TT

def mid_test(test_percentage, X, y):
    X, y = check_data(X, y)

    total_length = len(X)

    middle_test_size = round(test_percentage * total_length)

    # Calculate the start and end indices for the middle portion of the test set
    middle_test_start = total_length // 2 - middle_test_size // 2
    middle_test_end = middle_test_start + middle_test_size

    # Define the indices for the test set
    test_indices = np.arange(middle_test_start, middle_test_end)

    # Define the indices for the train set (complement of the test set)
    train_indices = np.setdiff1d(np.arange(total_length), test_indices)

    # Use the indices to create the train and test sets
    X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
    X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]

    TT = pd.DataFrame(columns=['Test/Train'], index=X.index)

    TT.iloc[train_indices, 0] = 'Train'
    TT.iloc[test_indices, 0] = 'Test'

    return X_train, y_train, X_test, y_test, TT

def end_test(test_percentage, X, y):
    X, y = check_data(X, y)

    #for 'end' traintest_split_type, we split the train and test so that the test data is the last part of the data
    split_index = int((1-test_percentage) * len(X))
    # Use train_test_split with shuffle=False so that it takes the last chunk as test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=len(X) - split_index, shuffle=False)

    TT = pd.DataFrame(columns=['Test/Train'], index=X.index)

    TT.iloc[0:len(X) - split_index, 0] = 'Train'
    TT.iloc[len(X) - split_index:, 0] = 'Test'

    return X_train, y_train, X_test, y_test, TT

def start_test(test_percentage, X, y):
    X, y = check_data(X, y)

    # Calculate the split index, taking the first 'test_percentage' part as the test set
    split_index = int(test_percentage * len(X))
    # Split the data without shuffling: first part is the test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_index, shuffle=False)

    TT = pd.DataFrame(columns=['Test/Train'], index=X.index)

    TT.iloc[split_index:, 0] = 'Train'
    TT.iloc[0:split_index, 0] = 'Test'

    return X_train, y_train, X_test, y_test, TT

def start_end_split(test_percentage, X, y):
    X, y = check_data(X, y)

    total_length = len(X)
    # Define the size of the middle and end portions of the test set (half of the total test percentage each)
    chunk_percentage = test_percentage / 2
    start_test_size = end_test_size = round(chunk_percentage * total_length)

    # Calculate the start and end indices for the middle portion of the test set
    start_test_start = 0
    start_test_end = start_test_size

    # Calculate the start and end indices for the end portion of the test set
    end_test_start = total_length - end_test_size
    end_test_end = total_length

    # Define the indices for the test set
    test_indices = np.concatenate([np.arange(start_test_start, start_test_end),
                                   np.arange(end_test_start, end_test_end)])

    # Define the indices for the train set (complement of the test set)
    train_indices = np.setdiff1d(np.arange(total_length), test_indices)

    # Use the indices to create the train and test sets
    X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
    X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]

    TT = pd.DataFrame(columns=['Test/Train'], index=X.index)

    TT.iloc[train_indices, 0] = 'Train'
    TT.iloc[test_indices, 0] = 'Test'

    return X_train, y_train, X_test, y_test, TT

def start_mid_split(test_percentage, X, y):
    X, y = check_data(X, y)

    total_length = len(X)
    # Define the size of the middle and end portions of the test set (half of the total test percentage each)
    chunk_percentage = test_percentage / 2
    start_test_size = end_test_size = round(chunk_percentage * total_length)

    # Calculate the start and end indices for the middle portion of the test set
    start_test_start = 0
    start_test_end = start_test_size

    # Calculate the start and end indices for the end portion of the test set
    end_test_start = total_length // 2 - end_test_size // 2
    end_test_end = end_test_start + end_test_size

    # Define the indices for the test set
    test_indices = np.concatenate([np.arange(start_test_start, start_test_end),
                                   np.arange(end_test_start, end_test_end)])

    # Define the indices for the train set (complement of the test set)
    train_indices = np.setdiff1d(np.arange(total_length), test_indices)

    # Use the indices to create the train and test sets
    X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
    X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]

    TT = pd.DataFrame(columns=['Test/Train'], index=X.index)

    TT.iloc[train_indices, 0] = 'Train'
    TT.iloc[test_indices, 0] = 'Test'

    return X_train, y_train, X_test, y_test, TT

def split_in_thirds(test_percentage, X, y):
    X, y = check_data(X, y)

    total_length = len(X)
    # Define the size of the middle and end portions of the test set (half of the total test percentage each)
    chunk_percentage = test_percentage / 2
    middle_test_size = end_test_size = round(chunk_percentage * total_length)

    # Calculate the start and end indices for the middle portion of the test set
    #middle_test_start = total_length // 4 - end_test_size // 2
    #middle_test_end = middle_test_start + middle_test_size

    middle_test_start = 0
    middle_test_end = middle_test_size

    # Calculate the start and end indices for the end portion of the test set
    end_test_start = total_length // 1.5
    end_test_end = end_test_start + end_test_size

    # Define the indices for the test set
    test_indices = np.concatenate([np.arange(middle_test_start, middle_test_end),
                                   np.arange(end_test_start, end_test_end)])

    # Define the indices for the train set (complement of the test set)
    train_indices = np.setdiff1d(np.arange(total_length), test_indices)

    # Use the indices to create the train and test sets
    X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
    X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]

    TT = pd.DataFrame(columns = ['Test/Train'], index=X.index)

    TT.iloc[train_indices, 0] = 'Train'
    TT.iloc[test_indices, 0] = 'Test'

    return X_train, y_train, X_test, y_test, TT

#takes a random chunk of 10% length, then the other 10% is taken from the data after half way through
def random_twochunk_split(test_percentage, X, y):
    X, y = check_data(X, y)

    total_length = len(X)
    # Define the size of the middle and end portions of the test set (half of the total test percentage each)
    chunk_percentage = test_percentage / 2
    test_size = round(chunk_percentage * total_length)
    '''
    # Calculate the start and end indices for the first test chunk
    b = round((total_length / 2)) - test_size
    first_test_start = random.randint(0, b)
    first_test_end = first_test_start + test_size

    # Calculate the start and end indices for the end portion of the test set
    a = round(total_length / 2)
    b = total_length - test_size
    second_test_start = random.randint(a, b)
    second_test_end = second_test_start + test_size
    '''
    # Calculate the start and end indices for the first test chunk
    first_test_start = random.randint(0, total_length)
    if first_test_start > total_length - test_size:
        first_test_end = test_size - (total_length - first_test_start)
        first_test_indices = np.concatenate([np.arange(first_test_start, total_length), np.arange(0, first_test_end)])
    else:
        first_test_end = first_test_start + test_size
        first_test_indices = np.concatenate([np.arange(first_test_start, first_test_end)])

    second_test_start = first_test_end + round((total_length - test_size)/2)
    if second_test_start > total_length:
        second_test_start = second_test_start - total_length

    if second_test_start > total_length - test_size:
        second_test_end = test_size - (total_length - second_test_start)
        second_test_indices = np.concatenate([np.arange(second_test_start, total_length), np.arange(0, second_test_end)])
    else:
        second_test_end = second_test_start + test_size
        second_test_indices = np.concatenate([np.arange(second_test_start, second_test_end)])

    # Define the indices for the test set
    test_indices = np.concatenate([first_test_indices,
                                   second_test_indices])

    # Define the indices for the train set (complement of the test set)
    train_indices = np.setdiff1d(np.arange(total_length), test_indices)

    # Use the indices to create the train and test sets
    X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
    X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]

    TT = pd.DataFrame(columns = ['Test/Train'], index=X.index)

    TT.iloc[train_indices, 0] = 'Train'
    TT.iloc[test_indices, 0] = 'Test'

    return X_train, y_train, X_test, y_test, TT

#the ones below this were very specific to Caroline's work...
def testingforvoc(test_percentage, X, y):
    X, y = check_data(X, y)

    total_length = len(X)
    # Define the size of the middle and end portions of the test set (half of the total test percentage each)
    chunk_percentage = test_percentage / 2
    middle_test_size = end_test_size = round(chunk_percentage * total_length)

    # Calculate the start and end indices for the middle portion of the test set
    middle_test_start = total_length // 4
    middle_test_end = middle_test_start + middle_test_size

    # Calculate the start and end indices for the end portion of the test set
    end_test_start = total_length - end_test_size - total_length*0.05
    end_test_end = end_test_start + end_test_size

    # Define the indices for the test set
    test_indices = np.concatenate([np.arange(middle_test_start, middle_test_end),
                                   np.arange(end_test_start, end_test_end)])

    # Define the indices for the train set (complement of the test set)
    train_indices = np.setdiff1d(np.arange(total_length), test_indices)

    # Use the indices to create the train and test sets
    X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
    X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]

    TT = pd.DataFrame(columns=['Test/Train'], index=X.index)

    TT.iloc[train_indices, 0] = 'Train'
    TT.iloc[test_indices, 0] = 'Test'

    return X_train, y_train, X_test, y_test, TT

def testingformethane(test_percentage, X, y):
    X, y = check_data(X, y)

    total_length = len(X)
    # Define the size of the middle and end portions of the test set (half of the total test percentage each)
    chunk_percentage = test_percentage / 2
    middle_test_size = end_test_size = round(chunk_percentage * total_length)

    # Calculate the start and end indices for the middle portion of the test set
    middle_test_start = total_length // 2
    middle_test_end = middle_test_start + middle_test_size

    # Calculate the start and end indices for the end portion of the test set
    end_test_start = total_length - end_test_size - total_length*0.2
    end_test_end = end_test_start + end_test_size

    # Define the indices for the test set
    test_indices = np.concatenate([np.arange(middle_test_start, middle_test_end),
                                   np.arange(end_test_start, end_test_end)])

    # Define the indices for the train set (complement of the test set)
    train_indices = np.setdiff1d(np.arange(total_length), test_indices)

    # Use the indices to create the train and test sets
    X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
    X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]

    TT = pd.DataFrame(columns=['Test/Train'], index=X.index)

    TT.iloc[train_indices, 0] = 'Train'
    TT.iloc[test_indices, 0] = 'Test'

    return X_train, y_train, X_test, y_test, TT

def winter_summer_split(test_percentage, X, y):
    X, y = check_data(X, y)

    total_length = len(X)

    # Define the size of the middle and end portions of the test set
    chunk_percentage = test_percentage / 2
    middle_test_size = end_test_size = round(chunk_percentage * total_length)

    # Ensure index is a DatetimeIndex
    if not isinstance(X.index, pd.DatetimeIndex):
        raise ValueError("X.index must be a DatetimeIndex for this operation.")

    # --- Find the exact positions of the start timestamps ---
    middle_start_ts = pd.Timestamp("2024-01-15 00:00")
    end_start_ts = pd.Timestamp("2025-07-15 00:00")

    # Locate integer positions
    middle_test_start = X.index.get_loc(middle_start_ts)
    end_test_start = X.index.get_loc(end_start_ts)

    # Calculate the end positions using the same lengths as before
    middle_test_end = middle_test_start + middle_test_size
    end_test_end = end_test_start + end_test_size

    # --- Define test and train indices ---
    test_indices = np.concatenate([
        np.arange(middle_test_start, middle_test_end),
        np.arange(end_test_start, end_test_end)
    ])

    # Train indices are everything else
    train_indices = np.setdiff1d(np.arange(total_length), test_indices)

    # --- Create splits ---
    X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
    X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]

    TT = pd.DataFrame(columns=['Test/Train'], index=X.index)

    TT.iloc[train_indices, 0] = 'Train'
    TT.iloc[test_indices, 0] = 'Test'

    return X_train, y_train, X_test, y_test, TT

def winter_summer_NO2_split(test_percentage, X, y):
    X, y = check_data(X, y)

    total_length = len(X)

    # Define the size of the middle and end portions of the test set
    chunk_percentage = test_percentage / 2
    middle_test_size = end_test_size = round(chunk_percentage * total_length)

    # Ensure index is a DatetimeIndex
    if not isinstance(X.index, pd.DatetimeIndex):
        raise ValueError("X.index must be a DatetimeIndex for this operation.")

    # --- Find the exact positions of the start timestamps ---
    #middle_start_ts = pd.Timestamp("2023-12-25 00:00")
    #end_start_ts = pd.Timestamp("2024-04-25 00:00")

    middle_start_ts = pd.Timestamp("2023-11-10 00:00")
    end_start_ts = pd.Timestamp("2024-04-25 00:00")

    # Locate integer positions
    middle_test_start = X.index.get_loc(middle_start_ts)
    end_test_start = X.index.get_loc(end_start_ts)

    # Calculate the end positions using the same lengths as before
    middle_test_end = middle_test_start + middle_test_size
    end_test_end = end_test_start + end_test_size

    # --- Define test and train indices ---
    test_indices = np.concatenate([
        np.arange(middle_test_start, middle_test_end),
        np.arange(end_test_start, end_test_end)
    ])

    # Train indices are everything else
    train_indices = np.setdiff1d(np.arange(total_length), test_indices)

    # --- Create splits ---
    X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
    X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]

    TT = pd.DataFrame(columns=['Test/Train'], index=X.index)

    TT.iloc[train_indices, 0] = 'Train'
    TT.iloc[test_indices, 0] = 'Test'

    return X_train, y_train, X_test, y_test, TT

def winter_summer_C16_split(test_percentage, X, y):
    X, y = check_data(X, y)

    total_length = len(X)

    # Define the size of the middle and end portions of the test set
    chunk_percentage = test_percentage / 2
    middle_test_size = end_test_size = round(chunk_percentage * total_length)

    # Ensure index is a DatetimeIndex
    if not isinstance(X.index, pd.DatetimeIndex):
        raise ValueError("X.index must be a DatetimeIndex for this operation.")

    # --- Find the exact positions of the start timestamps ---
    middle_start_ts = pd.Timestamp("2025-01-15 00:00")
    end_start_ts = pd.Timestamp("2025-05-01 00:00")

    # Locate integer positions
    middle_test_start = X.index.get_loc(middle_start_ts)
    end_test_start = X.index.get_loc(end_start_ts)

    # Calculate the end positions using the same lengths as before
    middle_test_end = middle_test_start + middle_test_size
    end_test_end = end_test_start + end_test_size

    # --- Define test and train indices ---
    test_indices = np.concatenate([
        np.arange(middle_test_start, middle_test_end),
        np.arange(end_test_start, end_test_end)
    ])

    # Train indices are everything else
    train_indices = np.setdiff1d(np.arange(total_length), test_indices)

    # --- Create splits ---
    X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
    X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]

    TT = pd.DataFrame(columns=['Test/Train'], index=X.index)

    TT.iloc[train_indices, 0] = 'Train'
    TT.iloc[test_indices, 0] = 'Test'

    return X_train, y_train, X_test, y_test, TT

def winter_summer_C24_split(test_percentage, X, y):
    X, y = check_data(X, y)

    total_length = len(X)

    # Define the size of the middle and end portions of the test set
    chunk_percentage = test_percentage / 2
    middle_test_size = end_test_size = round(chunk_percentage * total_length)

    # Ensure index is a DatetimeIndex
    if not isinstance(X.index, pd.DatetimeIndex):
        raise ValueError("X.index must be a DatetimeIndex for this operation.")

    # --- Find the exact positions of the start timestamps ---
    middle_start_ts = pd.Timestamp("2023-12-01 00:00")
    end_start_ts = pd.Timestamp("2024-04-15 00:00")

    # Locate integer positions
    middle_test_start = X.index.get_loc(middle_start_ts)
    end_test_start = X.index.get_loc(end_start_ts)

    # Calculate the end positions using the same lengths as before
    middle_test_end = middle_test_start + middle_test_size
    end_test_end = end_test_start + end_test_size

    # --- Define test and train indices ---
    test_indices = np.concatenate([
        np.arange(middle_test_start, middle_test_end),
        np.arange(end_test_start, end_test_end)
    ])

    # Train indices are everything else
    train_indices = np.setdiff1d(np.arange(total_length), test_indices)

    # --- Create splits ---
    X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
    X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]

    TT = pd.DataFrame(columns=['Test/Train'], index=X.index)

    TT.iloc[train_indices, 0] = 'Train'
    TT.iloc[test_indices, 0] = 'Test'

    return X_train, y_train, X_test, y_test, TT

def winter_summer_CO_split(test_percentage, X, y):
    X, y = check_data(X, y)

    total_length = len(X)

    # Define the size of the middle and end portions of the test set
    chunk_percentage = test_percentage / 2
    middle_test_size = end_test_size = round(chunk_percentage * total_length)

    # Ensure index is a DatetimeIndex
    if not isinstance(X.index, pd.DatetimeIndex):
        raise ValueError("X.index must be a DatetimeIndex for this operation.")

    # --- Find the exact positions of the start timestamps ---
    middle_start_ts = pd.Timestamp("2024-12-01 00:00")
    end_start_ts = pd.Timestamp("2025-05-01 00:00")

    # Locate integer positions
    middle_test_start = X.index.get_loc(middle_start_ts)
    end_test_start = X.index.get_loc(end_start_ts)

    # Calculate the end positions using the same lengths as before
    middle_test_end = middle_test_start + middle_test_size
    end_test_end = end_test_start + end_test_size

    # --- Define test and train indices ---
    test_indices = np.concatenate([
        np.arange(middle_test_start, middle_test_end),
        np.arange(end_test_start, end_test_end)
    ])

    # Train indices are everything else
    train_indices = np.setdiff1d(np.arange(total_length), test_indices)

    # --- Create splits ---
    X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
    X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]

    TT = pd.DataFrame(columns=['Test/Train'], index=X.index)

    TT.iloc[train_indices, 0] = 'Train'
    TT.iloc[test_indices, 0] = 'Test'

    return X_train, y_train, X_test, y_test, TT
