# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 10:20:51 2023

@author: cfris
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Define a function to create a boxplot
def field_boxplot(data, model_name, output_folder_name, colo_output_folder, pollutant, unit):
    # Set the figure size
    plt.figure(figsize=(11, 6))

    # Create a boxplot using Seaborn
    sns.boxplot(x='location', y=pollutant, data=data)

    # Set labels for the x and y axes
    plt.xlabel('Pod site')
    plt.ylabel(pollutant + ' Concentration (' + unit + ')')

    plt.show()
    # Save the boxplot as an image file
    plt.savefig(os.path.join('Outputs', colo_output_folder, output_folder_name, model_name + '_field_prediction_boxplot.png'))

# Example usage:
# field_boxplot(data, 'ModelName', 'OutputFolder', 'ColoOutputFolder', 'Pollutant', 'Unit')