# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 14:06:04 2023

@author: cfris
"""
import matplotlib.pyplot as plt
import os

def colo_timeseries(y_train, y_train_predicted, y_test, y_test_predicted, pollutant, model_name, output_folder_name):
    # Set the figure size
    plt.figure(figsize=(12, 4))
    
    # Scatter plot for actual (training) values
    plt.scatter(y_train.index, y_train, label='Actual (Training)', color='blue', marker='.')
    
    # Scatter plot for predicted (training) values
    plt.scatter(y_train.index, y_train_predicted, label='Predicted (Training)', color='darkorange', marker='.')
    
    # Scatter plot for actual (test) values
    plt.scatter(y_test.index, y_test, label='Actual (Test)', color='lightblue', marker='.')
    
    # Scatter plot for predicted (test) values
    plt.scatter(y_test.index, y_test_predicted, label='Predicted (Test)', color='gold', marker='.')
    
    # Set the title of the plot
    plt.title(pollutant + ' ' + model_name + ' - Actual vs Predicted')
    
    # Add a legend to the plot
    plt.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()

    # Save the plot as an image file
    plt.savefig(os.path.join('Outputs', output_folder_name, model_name + 'ref_colo_timeseries.png'))

# Example usage:
# model_reference_timeseries(y_train, y_train_predicted, y_test, y_test_predicted, 'Pollutant', 'ModelName', 'OutputFolderName')
