# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 11:41:34 2023

@author: cfris
"""
import numpy as np
import matplotlib.pyplot as plt
import os

def colo_stats_plot(models, model_stats, pollutant,output_folder_name):
    #plot RMSE, MBE, and R2 of each model
    #x = np.arange(len(models))  # the label locations
    
    fig, ax = plt.subplots(1, 3, figsize=(11, 4))
    bar_width = 0.35
    opacity = 0.7
    
    # Bar plot for training RMSE
    rects1 = ax[0].bar(np.arange(len(model_stats)), model_stats['Training_RMSE'], bar_width,
                       label='Training RMSE', color='blue', alpha=opacity)
    
    # Bar plot for testing RMSE
    rects2 = ax[0].bar(np.arange(len(model_stats)) + bar_width, model_stats['Testing_RMSE'], bar_width,
                       label='Testing RMSE', color='orange', alpha=opacity)
    
    #ax[0].set_title('Training and Testing RMSE for Each Model')
    ax[0].set_ylabel('RMSE')
    ax[0].set_xticks(np.arange(len(model_stats)) + bar_width / 2)
    ax[0].set_xticklabels(models)
    
    # Add value labels on top of each bar
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax[0].text(rect.get_x() + rect.get_width() / 2, height,
                   f'{height:.2f}', ha='center', va='bottom')
        
    #ax[0].set_title('Training and Testing RMSE for Each Model')
    ax[0].set_ylabel('RMSE')
    ax[0].set_xticks(np.arange(len(model_stats)) + bar_width / 2)
    ax[0].set_xticklabels(models)
    
    # next subplot: Bar plot of training R2
    rects3 = ax[1].bar(np.arange(len(model_stats)), model_stats['Training_R2'], bar_width,
                       label='Training R2', color='blue', alpha=opacity)
    
    ax[1].set_ylabel('R2')
    ax[1].set_xticks(np.arange(len(model_stats)))
    ax[1].set_xticklabels(models)
    
    # Add value labels on top of each bar
    for rect in rects3:
        height = rect.get_height()
        ax[1].text(rect.get_x() + rect.get_width() / 2, height,
                   f'{height:.2f}', ha='center', va='bottom')
      
    # Bar plot for training MBE
    rects4 = ax[2].bar(np.arange(len(model_stats)), model_stats['Training_MBE'], bar_width,
                       label='Training MBE', color='blue', alpha=opacity)
    
    # Bar plot for testing MBE
    rects5 = ax[2].bar(np.arange(len(model_stats)) + bar_width, model_stats['Testing_MBE'], bar_width,
                       label='Testing MBE', color='orange', alpha=opacity)
    
    # Add value labels on top of each bar
    for rect in rects4 + rects5:
        height = rect.get_height()
        ax[2].text(rect.get_x() + rect.get_width() / 2, height,
                   f'{height:.2f}', ha='center', va='bottom')   
     
    
    ax[2].set_ylabel('MBE')
    ax[2].set_xticks(np.arange(len(model_stats)) + bar_width / 2)
    ax[2].set_xticklabels(models)   
    
    #set the y axis so that it is symmetrically around zero
    # Get the current y-axis limits
    y_min, y_max = ax[2].get_ylim()
    abs_max = max(abs(y_min), abs(y_max))
    ax[2].set_ylim(-abs_max, abs_max)
        
    # Move the legend outside the plot
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),ncol=2)
    ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
    ax[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),ncol=2)
    
    #set overall figure title 
    fig.suptitle(pollutant)
    
    # Adjust layout for better spacing
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
    #save plot in outputs folder
    plt.savefig(os.path.join('Outputs', output_folder_name, 'colo_model_statistics.png'))
    
