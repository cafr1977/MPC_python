# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 12:16:39 2023

@author: cfris
"""

from sklearn.model_selection import train_test_split

def end_test(test_percentage, X, y):
    #for 'end' traintest_split_type, we split the train and test so that the test data is the last part of the data
    split_index = int((1-test_percentage) * len(X))
    # Use train_test_split with shuffle=False so that it takes the last chunk as test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=len(X) - split_index, shuffle=False)
    return X_train, y_train, X_test, y_test