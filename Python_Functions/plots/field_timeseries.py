# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 10:20:51 2023

@author: cfris
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns

def field_timeseries(data, model_name, output_folder_name, colo_output_folder, pollutant, unit):
    # Set the figure size
    plt.figure(figsize=(11, 6))
    
    # Create a scatter plot using Seaborn
    sns.scatterplot(x='datetime', y=pollutant, hue='location', data=data, marker='.')

    # Set labels for the x and y axes
    plt.xlabel('Datetime')
    plt.ylabel(pollutant + ' Concentration (' + unit + ')')

    plt.show()
    # Save the plot as an image file
    plt.savefig(os.path.join('Outputs', colo_output_folder, output_folder_name, model_name + '_field_prediction_timeseries.png'))

# Example usage:
# field_timeseries(data, 'ModelName', 'OutputFolderName', 'ColoOutputFolder', 'Pollutant', 'Unit')
