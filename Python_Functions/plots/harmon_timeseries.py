# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 15:17:14 2023

@author: cfris
"""
import os
import matplotlib.pyplot as plt

def harmon_timeseries(colo_pod_harmon_data, pod_fitted, colo_output_folder, output_folder_name):
    # Create subplots for each sensor in colo_pod_harmon_data
    fig, axs = plt.subplots(nrows=colo_pod_harmon_data.shape[1], ncols=1, figsize=(15, 8.5))

    # Iterate over each sensor
    for i, sensor in enumerate(colo_pod_harmon_data):
        # Iterate over each key in pod_fitted
        for key in pod_fitted:
            # Scatter plot for fitted values for each key
            axs[i].scatter(colo_pod_harmon_data.index, pod_fitted[key][sensor], label=key, marker='.')

        # Plot the actual values for the sensor
        axs[i].plot(colo_pod_harmon_data.index, colo_pod_harmon_data[sensor], color='k')

        # Set y-axis label and title for each subplot
        axs[i].set_ylabel('Sensor value')
        axs[i].set_title(sensor)

    # Add a legend for keys
    fig.legend(list(pod_fitted), loc='lower center', bbox_to_anchor=(0.5, 0), ncol=i+1)

    # Set x-axis label for the last subplot
    axs[-1].set_xlabel('Time')

    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.1)

    # Show the plot
    plt.show()

    # Save the plot as an image file
    plt.savefig(os.path.join('Outputs', colo_output_folder, output_folder_name, '_harmon_timeseries.png'))

# Example usage:
# harmon_timeseries(colo_pod_harmon_data, pod_fitted, 'ColoOutputFolder', 'OutputFolderName')
