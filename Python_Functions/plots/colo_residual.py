# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 15:45:01 2023

@author: cfris
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns

def colo_residual(y_train, y_train_predicted, y_test, y_test_predicted, X_train, X_test, pollutant, model_name, output_folder_name, X_features):
    # Calculate residuals
    training_residuals = y_train_predicted - y_train
    testing_residuals = y_test_predicted - y_test
    
    # Create subplots for each feature in X
    fig, axs = plt.subplots(nrows=1, ncols=X_train.shape[1]+1, figsize=(18, 5))
    
    # Iterate over each feature
    for i in range(X_train.shape[1]):
        # Scatter plot of residuals vs the i-th feature
        axs[i+1].scatter(X_train[:, i], training_residuals, color='blue', label='training', marker='.')
        axs[i+1].scatter(X_test[:, i], testing_residuals, color='lightblue', label='testing', marker='.')

        # Add a dashed line at y=0
        axs[i+1].axhline(y=0, color='gray', linestyle='--')
        
        # Set labels and title
        axs[i+1].set_xlabel(X_features[i])
        axs[i+1].set_ylabel('Residuals')
    
    # Plot KDE (Kernel Density Estimate) for residuals
    sns.kdeplot(y=training_residuals, ax=axs[0], color='blue', label='training')
    sns.kdeplot(y=testing_residuals, ax=axs[0], color='lightblue', label='testing')
    
    # Set labels and title for the PDF subplot
    axs[0].set_xlabel('Residuals')
    axs[0].set_ylabel('Probability Density')
    axs[0].set_title('Residuals PDF')
    axs[0].axhline(y=0, color='gray', linestyle='--')

    axs[0].legend()
    
    # Set the main title
    fig.suptitle(pollutant + ' ' + model_name + ' Residuals Analysis')
        
    # Adjust layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
    # Save the plot as an image file
    plt.savefig(os.path.join('Outputs', output_folder_name, model_name + 'colo_residual.png'))


# Example usage:
# residual_plot(y_train, y_train_predicted, y_test, y_test_predicted, X_train, X_test, 'Pollutant', 'ModelName', 'OutputFolderName', X_features)
