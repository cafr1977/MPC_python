import matplotlib
matplotlib.use('Qt5Agg')

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import copy
import pandas as pd


#Colocation plots
def colo_timeseries(y_train, y_train_predicted, y_test, y_test_predicted, pollutant, model_name, output_folder_name,run_name, unit):
    # Set the figure size
    plt.figure(figsize=(12, 4))

    # Scatter plot for actual (training) values
    plt.scatter(y_train.index, y_train, label='Actual (Training)', color='gray', marker='.')

    # Scatter plot for predicted (training) values
    plt.scatter(y_train.index, y_train_predicted, label='Predicted (Training)', color='blue', marker='.')

    # Scatter plot for actual (test) values
    plt.scatter(y_test.index, y_test, label='Actual (Test)', color='gray', marker='.')

    # Scatter plot for predicted (test) values
    plt.scatter(y_test.index, y_test_predicted, label='Predicted (Test)', color='lightblue', marker='.')

    # Set the title of the plot
    plt.title(run_name + ' ' + pollutant + ' ' + model_name + ' - Actual vs Predicted')
    plt.ylabel(pollutant + ' conc. (' + unit + ')')
    plt.xlabel('Date')
    # Add a legend to the plot
    plt.legend()

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show(block=False)

    # Save the plot as an image file
    plt.savefig(os.path.join('Outputs', output_folder_name, model_name + 'ref_colo_timeseries.png'))

    # Example usage:
    # model_reference_timeseries(y_train, y_train_predicted, y_test, y_test_predicted, 'Pollutant', 'ModelName', 'OutputFolderName')
def colo_stats_plot(models, model_stats, pollutant, output_folder_name, run_name):
    #settings['models'], model_stats, settings['pollutant'],output_folder_name, settings['colo_run_name']
    # plot RMSE, MBE, and R2 of each model
    # x = np.arange(len(models))  # the label locations

    fig, ax = plt.subplots(1, 3, figsize=(11, 4))
    bar_width = 0.35
    opacity = 0.7

    # Bar plot for training RMSE
    rects1 = ax[0].bar(np.arange(len(model_stats)), model_stats['Training_RMSE'], bar_width,
                       label='Training RMSE', color='blue', alpha=opacity)

    # Bar plot for testing RMSE
    rects2 = ax[0].bar(np.arange(len(model_stats)) + bar_width, model_stats['Testing_RMSE'], bar_width,
                       label='Testing RMSE', color='lightblue', alpha=opacity)

    # ax[0].set_title('Training and Testing RMSE for Each Model')
    ax[0].set_ylabel('RMSE')
    ax[0].set_xticks(np.arange(len(model_stats)) + bar_width / 2)
    ax[0].set_xticklabels(models, rotation=45, ha='right')

    # Add value labels on top of each bar
    for rect in rects2:
        height = rect.get_height()
        ax[0].text(rect.get_x() + rect.get_width() / 2, height,
                    f'{height:.1f}', ha='center', va='bottom')

    for rect in rects1:
        height = rect.get_height()
        ax[0].text(rect.get_x() + rect.get_width() / 2, height/2,
                    f'{height:.1f}', ha='center', va='bottom')

    # next subplot: Bar plot of training R2
    rects3 = ax[1].bar(np.arange(len(model_stats)), model_stats['Training_R2'], bar_width,
                       label='Training R2', color='blue', alpha=opacity)

    # next subplot: Bar plot of training R2
    rects6 = ax[1].bar(np.arange(len(model_stats)) + bar_width, model_stats['Testing_R2'], bar_width,
                       label='Testing R2', color='lightblue', alpha=opacity)


    ax[1].set_ylabel('R2')
    ax[1].set_xticks(np.arange(len(model_stats)))
    ax[1].set_xticklabels(models, rotation=45, ha='right')

    # Add value labels on top of each bar
    for rect in rects3:
        height = rect.get_height()
        ax[1].text(rect.get_x() + rect.get_width() / 2, height,
                   f'{height:.2f}', ha='center', va='bottom')

    # Add value labels on top of each bar
    for rect in rects6:
        height = rect.get_height()
        ax[1].text(rect.get_x() + rect.get_width() / 2, height,
                   f'{height:.2f}', ha='center', va='bottom')

    # Bar plot for training MBE
    rects4 = ax[2].bar(np.arange(len(model_stats)), model_stats['Training_MBE'], bar_width,
                       label='Training MBE', color='blue', alpha=opacity)

    # Bar plot for testing MBE
    rects5 = ax[2].bar(np.arange(len(model_stats)) + bar_width, model_stats['Testing_MBE'], bar_width,
                       label='Testing MBE', color='lightblue', alpha=opacity)

    # Add value labels on top of each bar
    for rect in rects5:
        height = rect.get_height()
        ax[2].text(rect.get_x() + rect.get_width() / 2, height,
                    f'{height:.1f}', ha='center', va='bottom')

    for rect in rects4:
        height = rect.get_height()
        ax[2].text(rect.get_x() + rect.get_width() / 2, height/2,
                    f'{height:.1f}', ha='center', va='bottom')

    ax[2].set_ylabel('MBE')
    ax[2].set_xticks(np.arange(len(models)) + bar_width / 2)
    ax[2].set_xticklabels(models, rotation=45, ha='right')

    # set the y axis so that it is symmetrically around zero
    # Get the current y-axis limits
    y_min, y_max = ax[2].get_ylim()
    abs_max = max(abs(y_min), abs(y_max))
    ax[2].set_ylim(-abs_max, abs_max)

    # Move the legend outside the plot
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2)
    ax[1].legend(loc='upper center',bbox_to_anchor=(0.5, 1.25), ncol=2)
    ax[2].legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2)

    # set overall figure title
    fig.suptitle(run_name + ' ' + pollutant, y=0.9)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the plot
    plt.show(block=False)

    # save plot in outputs folder
    plt.savefig(os.path.join('Outputs', output_folder_name, 'colo_model_statistics.png'))
def colo_scatter(y_train, y_train_predicted, y_test, y_test_predicted, pollutant, model_name, output_folder_name,run_name):
    # Set the figure size
    plt.figure(figsize=(8, 8))

    # Scatter plot for training data
    plt.scatter(y_train, y_train_predicted, label='Training', marker='.', color='blue', s = 50)

    # Scatter plot for testing data
    plt.scatter(y_test, y_test_predicted, label='Testing', marker='.', color='lightblue', s = 50)

    # Set the title of the plot
    plt.title(run_name + ' ' + pollutant + ' ' + model_name + ' - Actual vs Predicted')

    # Plot a dashed 1:1 line
    plt.plot(
        [min([min(y_train_predicted), min(y_test_predicted), min(y_train), min(y_test)]),
         max([max(y_train_predicted), max(y_test_predicted), max(y_train), max(y_test)])],
        [min([min(y_train_predicted), min(y_test_predicted), min(y_train), min(y_test)]),
         max([max(y_train_predicted), max(y_test_predicted), max(y_train), max(y_test)])],
        'k--', label='1:1 Line'
    )

    # Add a legend to the plot
    plt.legend(fontsize = 16)

    # Set labels for the x and y axes
    plt.xlabel('Reference')
    plt.ylabel('Predicted')

    # Set the font size of x and y tick labels
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Adjust the layout to prevent clipping of labels
    plt.tight_layout()

    # Show the plot
    plt.show(block=False)

    # Save the plot as an image file
    plt.savefig(os.path.join('Outputs', output_folder_name, model_name + 'ref_vs_pred_scatter.png'))
def colo_residual(y_train, y_train_predicted, y_test, y_test_predicted, pollutant, model_name,output_folder_name, X_features, run_name, X_train, X_test):
    # Calculate residuals
    training_residuals = y_train_predicted - y_train
    testing_residuals = y_test_predicted - y_test

    # Create subplots for each feature in X
    fig, axs = plt.subplots(nrows=1, ncols=X_train.shape[1] + 1, figsize=(18, 5))

    # Iterate over each feature
    for i in range(X_train.shape[1]):
        # Scatter plot of residuals vs the i-th feature
        axs[i + 1].scatter(X_train[:, i], training_residuals, color='blue', label='training', marker='.')
        axs[i + 1].scatter(X_test[:, i], testing_residuals, color='lightblue', label='testing', marker='.')

        # Add a dashed line at y=0
        axs[i + 1].axhline(y=0, color='gray', linestyle='--')

        # Set labels and title
        axs[i + 1].set_xlabel(X_features[i])
        axs[i + 1].set_ylabel('Residuals')

    # Plot KDE (Kernel Density Estimate) for residuals
    sns.kdeplot(data=training_residuals, ax=axs[0], color='blue', label='training')
    sns.kdeplot(data=testing_residuals, ax=axs[0], color='lightblue', label='testing')

    print(
        "Ignore FutureWarning about mode.use_inf_as_na for now. It is from kdeplot in colo_residual. Cannot figure out why")

    # Set labels and title for the PDF subplot
    axs[0].set_xlabel('Residuals')
    axs[0].set_ylabel('Probability Density')
    axs[0].set_title('Residuals PDF')
    axs[0].axhline(y=0, color='gray', linestyle='--')

    axs[0].legend()

    # Set the main title
    fig.suptitle(run_name + ' ' + pollutant + ' ' + model_name + ' Residuals Analysis')

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show(block=False)

    # Save the plot as an image file
    plt.savefig(os.path.join('Outputs', output_folder_name, model_name + 'colo_residual.png'))

    # Example usage:
    # residual_plot(y_train, y_train_predicted, y_test, y_test_predicted, X_train, X_test, 'Pollutant', 'ModelName', 'OutputFolderName', X_features)

def corr_heatmap(data_combined, output_folder_name):
    plt.figure(figsize=(12, 10))
    sns.heatmap(data=data_combined.corr(), annot=True, cmap="coolwarm",cbar=False,vmin=-1,vmax=1)
    plt.title('')
    plt.show(block=False)
    # Save the plot as an image file
    plt.savefig(os.path.join('Outputs', output_folder_name, 'corr_heatmap.png'))

def feature_importance(current_model, output_folder_name, run_name, pollutant, model_name, features):
    if hasattr(current_model, 'feature_importances_'):
        # Plot feature importance for Random Forest
        rf_feature_importance = current_model.feature_importances_
        sorted_idx = np.argsort(rf_feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + 0.5
        plt.figure(figsize=(14, 6))
        plt.bar(pos, rf_feature_importance[sorted_idx], width=0.8, align='center')
        plt.xticks(pos, [features[i] for i in sorted_idx], rotation=90)
        plt.ylabel('Relative Importance')
        plt.xlabel('Feature')
        plt.title(run_name + ' ' + pollutant + ' ' + model_name + ' - Feature Importance')
        plt.tight_layout()
        plt.show(block=False)

        # Save the plot as an image file
        plt.savefig(os.path.join('Outputs', output_folder_name, model_name + '_feature_important.png'))

#Field plots
def field_boxplot(data, model_name, output_folder_name, colo_output_folder, pollutant, unit):
    # Set the figure size
    plt.figure(figsize=(11, 6))

    # Create a boxplot using Seaborn
    sns.boxplot(x='location', y=pollutant, data=data)

    # Set labels for the x and y axes
    plt.xlabel('Pod site')
    plt.ylabel(pollutant + ' Concentration (' + unit + ')')

    plt.show(block=False)
    # Save the boxplot as an image file
    plt.savefig(os.path.join('Outputs', colo_output_folder, output_folder_name, model_name + '_field_prediction_boxplot.png'))

    # Example usage:
    # field_boxplot(data, 'ModelName', 'OutputFolder', 'ColoOutputFolder', 'Pollutant', 'Unit')
def field_timeseries(data, model_name, output_folder_name, colo_output_folder, pollutant, unit):
    # Set the figure size
    plt.figure(figsize=(11, 6))

    # Create a scatter plot using Seaborn
    sns.scatterplot(x='datetime', y=pollutant, hue='location', data=data, marker='.', palette='tab20')

    # Set labels for the x and y axes
    plt.xlabel('Datetime')
    plt.ylabel(pollutant + ' Concentration (' + unit + ')')

    plt.show(block=False)
    # Save the plot as an image file
    plt.savefig(os.path.join('Outputs', colo_output_folder, output_folder_name,
                             model_name + '_field_prediction_timeseries.png'))

    # Example usage:
    # field_timeseries(data, 'ModelName', 'OutputFolderName', 'ColoOutputFolder', 'Pollutant', 'Unit')

def field_histogram(data, model_name, output_folder_name, colo_output_folder, pollutant, unit):
    # Set the figure size
    # Determine the range of values for the pollutant
    min_value = data[pollutant].min()
    max_value = data[pollutant].max()

    # Calculate the number of bins based on the desired bin width
    num_bins = 30
    #bin_width = int((max_value - min_value)/ num_bins)  # Adjust as needed

    # Create a histogram for each location using Seaborn
    g = sns.FacetGrid(data=data, col='location', col_wrap=4)
    g.map(sns.histplot, pollutant, bins=num_bins)  # Adjust the bins as needed , binwidth=bin_width

    # Set labels for the x and y axes
    g.set_axis_labels(pollutant + ' Concentration (' + unit + ')', 'Frequency')

    # Show the plot
    plt.show(block=False)

    # Save the histogram as an image file
    plt.savefig(os.path.join('Outputs', colo_output_folder, output_folder_name, model_name + '_field_prediction_histogram.png'))

def harmonized_field_hist(data, output_folder_name, colo_output_folder, sensors_included):
    # Set the figure size
    for sensor in sensors_included:
        filtered_data = data[data['Sensor'].str.contains(sensor)]

        # Determine the range of values for the pollutant
        #min_value = filtered_data['Reading'].min()
        #max_value = filtered_data['Reading'].max()

        # Calculate the number of bins based on the desired bin width
        #num_bins = 30
        #bin_width = int((max_value - min_value) / num_bins)

        # Create a histogram for each location using Seaborn
        g = sns.FacetGrid(data=filtered_data, col='location', col_wrap=4)
        g.map(sns.histplot, 'Reading', bins=20)  # Adjust the bins as needed, binwidth = bin_width

        # Set labels for the x and y axes
        g.set_axis_labels(sensor, 'Frequency')

        # Show the plot
        plt.show(block=False)

        # Save the histogram as an image file
        plt.savefig(
            os.path.join('Outputs', colo_output_folder, output_folder_name, 'Harmonized_field_' + sensor + '_histogram.png'))

#Harmonization plots
def harmon_timeseries(colo_pod_harmon_data, pod_fitted, colo_output_folder, output_folder_name):
    # Create subplots for each sensor in colo_pod_harmon_data
    fig, axs = plt.subplots(nrows=colo_pod_harmon_data.shape[1], ncols=1, figsize=(15, 8.5))

    # Get the tab20 colormap
    tab20_cmap = plt.cm.get_cmap('tab20')

    # Generate a list of 30 colors from the tab20 colormap
    tab20_colors = [tab20_cmap(i) for i in range(30)]

    # Iterate over each sensor
    for i, sensor in enumerate(colo_pod_harmon_data):
        # Iterate over each key in pod_fitted
        for j, key in enumerate(pod_fitted):
            # Scatter plot for fitted values for each key
            axs[i].scatter(pod_fitted[key].index, pod_fitted[key][sensor], label=key, marker='.', color=tab20_colors[j])

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
    plt.show(block=False)

    # Save the plot as an image file
    plt.savefig(os.path.join('Outputs', colo_output_folder, output_folder_name, '_harmon_timeseries.png'))

    # Example usage:
    # harmon_timeseries(colo_pod_harmon_data, pod_fitted, 'ColoOutputFolder', 'OutputFolderName')
def harmon_scatter(colo_pod_harmon_data, pod_fitted, colo_output_folder, output_folder_name):
    # Create subplots for each sensor in colo_pod_harmon_data
    fig, axs = plt.subplots(nrows=round(colo_pod_harmon_data.shape[1] / 2), ncols=2,
                            figsize=(10, 3 * round(colo_pod_harmon_data.shape[1] / 2)))

    # Flatten the axs array to iterate over it
    axs_flat = axs.flatten()

    # Iterate over each sensor
    for i, sensor in enumerate(colo_pod_harmon_data):
        # Iterate over each key in pod_fitted
        for key in pod_fitted:
            # Scatter plot of colo_pod_harmon_data vs fitted values for each key
            temp = pd.merge(colo_pod_harmon_data[sensor], pod_fitted[key][sensor], how='outer', left_index=True,
                            right_index=True)
            axs_flat[i].scatter(temp[sensor + '_x'], temp[sensor + '_y'], label=key, marker='.')

        # Set title for the subplot
        axs_flat[i].set_title(sensor)

        # Plot a dashed 1:1 line
        axs_flat[i].plot([min(colo_pod_harmon_data[sensor]), max(colo_pod_harmon_data[sensor])],
                         [min(colo_pod_harmon_data[sensor]), max(colo_pod_harmon_data[sensor])], 'k--',
                         label='1:1 Line')

        # Set labels for the x and y axes
        axs_flat[i].set_xlabel('Secondary standard pod')
        axs_flat[i].set_ylabel('Fitted pod')

    # Add a legend for keys
    fig.legend(list(pod_fitted), loc='lower center', bbox_to_anchor=(0.5, 0), ncol=i + 1)

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show(block=False)

    # Save the plot as an image file
    fig.subplots_adjust(bottom=0.1)
    plt.savefig(os.path.join('Outputs', colo_output_folder, output_folder_name, 'harmon_scatter.png'))

    # Example usage:
    # harmon_scatter(colo_pod_harmon_data, pod_fitted, 'ColoOutputFolder', 'OutputFolderName')
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
    stats_df[['data_type', 'stat']] = stats_df['stat'].str.split('_', n=1, expand=True)

    # Create a FacetGrid for visualizing the melted data
    stat_plot = sns.FacetGrid(stats_df, row='sensor', col='stat', sharey=False, hue='data_type', aspect=2)

    # Map a scatter plot for each combination of 'sensor', 'stat', and 'data_type'
    stat_plot.map(sns.scatterplot, "pod", "value")

    # Set y-axis limits for the first column of plots
    for ax in stat_plot.axes[:, 0]:
        ax.set_ylim(0, 1.2)

    plt.show(block=False)

    # Save the FacetGrid as an image file
    stat_plot.savefig(os.path.join('Outputs', colo_output_folder, output_folder_name, 'harmonization_stats.png'))

    plt.figure()
    # Example usage:
    # harmon_stats_plot(model_stats, 'OutputFolderName', 'ColoOutputFolder', 'SensorsIncluded')

def colo_plots_series(colo_plot_list, y_train, y_train_predicted, y_test, y_test_predicted, pollutant, model_name, output_folder_name, colo_run_name,unit, current_model, features, X_train, X_test):
# plotting of modelled data
    if 'colo_timeseries' in colo_plot_list:
        colo_timeseries(y_train, y_train_predicted, y_test, y_test_predicted, pollutant,
                                        model_name, output_folder_name, colo_run_name, unit)

    if 'colo_scatter' in colo_plot_list:
        colo_scatter(y_train, y_train_predicted, y_test, y_test_predicted, pollutant,
                                    model_name, output_folder_name, colo_run_name)

    if 'colo_residual' in colo_plot_list:
        colo_residual(y_train, y_train_predicted, y_test, y_test_predicted,
                                    pollutant, model_name, output_folder_name, features,
                                    colo_run_name, X_train, X_test)

    if 'feature_importance' in colo_plot_list:
        feature_importance(current_model, output_folder_name, colo_run_name, pollutant, model_name, features)

