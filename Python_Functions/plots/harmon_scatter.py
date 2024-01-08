# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 15:17:14 2023

@author: cfris
"""
import os
import matplotlib.pyplot as plt
import pandas as pd

def harmon_scatter(colo_pod_harmon_data, pod_fitted, colo_output_folder, output_folder_name):
    # Create subplots for each sensor in colo_pod_harmon_data
    fig, axs = plt.subplots(nrows=round(colo_pod_harmon_data.shape[1]/2), ncols=2, figsize=(10, 3*round(colo_pod_harmon_data.shape[1]/2)))
    
    # Flatten the axs array to iterate over it
    axs_flat = axs.flatten()

    # Iterate over each sensor
    for i, sensor in enumerate(colo_pod_harmon_data):
        # Iterate over each key in pod_fitted
        for key in pod_fitted:
            # Scatter plot of colo_pod_harmon_data vs fitted values for each key
            temp = pd.merge(colo_pod_harmon_data[sensor], pod_fitted[key][sensor], how='outer', left_index=True, right_index=True)
            axs_flat[i].scatter(temp[sensor+ '_x'], temp[sensor + '_y'], label=key, marker='.')

        # Set title for the subplot
        axs_flat[i].set_title(sensor)

        # Plot a dashed 1:1 line
        axs_flat[i].plot([min(colo_pod_harmon_data[sensor]), max(colo_pod_harmon_data[sensor])],
                         [min(colo_pod_harmon_data[sensor]), max(colo_pod_harmon_data[sensor])], 'k--', label='1:1 Line')

        # Set labels for the x and y axes
        axs_flat[i].set_xlabel('Secondary standard pod')
        axs_flat[i].set_ylabel('Fitted pod')

    # Add a legend for keys
    fig.legend(list(pod_fitted), loc='lower center', bbox_to_anchor=(0.5, 0), ncol=i+1)

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()

    # Save the plot as an image file
    fig.subplots_adjust(bottom=0.1)
    plt.savefig(os.path.join('Outputs', colo_output_folder, output_folder_name, 'harmon_scatter.png'))

# Example usage:
# harmon_scatter(colo_pod_harmon_data, pod_fitted, 'ColoOutputFolder', 'OutputFolderName')
