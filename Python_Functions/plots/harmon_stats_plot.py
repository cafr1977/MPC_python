# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 12:06:08 2023

@author: cfris
"""
import seaborn as sns
import copy
import pandas as pd
import os

def harmon_stats_plot(model_stats, output_folder_name, colo_output_folder, sensors_included):
    # Create a deep copy of the model_stats dictionary to avoid modifying the original data
    model_stats_melted = copy.deepcopy(model_stats)

    # Melt the dataframes in model_stats for each stat and sensor
    temp = dict.fromkeys(model_stats)
    for stat in model_stats:
        for sensor in model_stats[stat]:
           model_stats_melted[stat][sensor] = pd.melt(model_stats_melted[stat][sensor], var_name='pod', value_name='value')
        temp[stat] = pd.concat([df.assign(sensor=name) for name, df in model_stats_melted[stat].items()])      

    # Concatenate the melted dataframes into a single dataframe
    stats_df = pd.concat([df.assign(stat=name) for name, df in temp.items()])

    # Split the 'stat' column into two columns using '_' so that testing or training is in one column and the stat is in the other
    stats_df[['data_type', 'stat']] = stats_df['stat'].str.split('_', 1, expand=True)

    # Create a FacetGrid for visualizing the melted data
    stat_plot = sns.FacetGrid(stats_df, row='sensor', col='stat', sharey=False, hue='data_type')

    # Map a scatter plot for each combination of 'sensor', 'stat', and 'data_type'
    stat_plot.map(sns.scatterplot, "pod", "value")

    # Set y-axis limits for the first column of plots
    for ax in stat_plot.axes[:, 0]:
        ax.set_ylim(0, 1.2)

    # Save the FacetGrid as an image file
    stat_plot.savefig(os.path.join('Outputs', colo_output_folder, output_folder_name, 'harmonization_stats.png'))

# Example usage:
# harmon_stats_plot(model_stats, 'OutputFolderName', 'ColoOutputFolder', 'SensorsIncluded')
       