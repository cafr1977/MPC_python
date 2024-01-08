# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 12:20:48 2023

@author: cfris
"""

import numpy as np

def mid_end_split(test_percentage, X, y):
    total_length = len(X)
    # Define the size of the middle and end portions of the test set (half of the total test percentage each)
    chunk_percentage = test_percentage/2
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
    
    return X_train, y_train, X_test, y_test