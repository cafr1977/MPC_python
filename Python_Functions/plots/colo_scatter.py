# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 15:17:14 2023

@author: cfris
"""
import os
import matplotlib.pyplot as plt

def colo_scatter(y_train, y_train_predicted, y_test, y_test_predicted, pollutant, model_name, output_folder_name):
    # Set the figure size
    plt.figure(figsize=(8, 8))
    
    # Scatter plot for training data
    plt.scatter(y_train, y_train_predicted, label='Training', marker='.', color='blue')
    
    # Scatter plot for testing data
    plt.scatter(y_test, y_test_predicted, label='Testing', marker='.', color='lightblue')
    
    # Set the title of the plot
    plt.title(pollutant + ' ' + model_name + ' - Actual vs Predicted')
    
    # Plot a dashed 1:1 line
    plt.plot(
        [min([min(y_train_predicted), min(y_test_predicted), min(y_train), min(y_test)]),
         max([max(y_train_predicted), max(y_test_predicted), max(y_train), max(y_test)])],
        [min([min(y_train_predicted), min(y_test_predicted), min(y_train), min(y_test)]),
         max([max(y_train_predicted), max(y_test_predicted), max(y_train), max(y_test)])],
        'k--', label='1:1 Line'
    )
    
    # Add a legend to the plot
    plt.legend()
    
    # Set labels for the x and y axes
    plt.xlabel('Reference')
    plt.ylabel('Predicted')
    
    # Adjust the layout to prevent clipping of labels
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
    # Save the plot as an image file
    plt.savefig(os.path.join('Outputs', output_folder_name, model_name + 'ref_vs_pred_scatter.png'))
