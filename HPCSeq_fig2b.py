import numpy as np
import pandas as pd
from scipy.stats import linregress, ttest_1samp, sem
import pdb
import matplotlib.pyplot as plt
import seaborn as sns
import os

import matplotlib as mpl
# Set font type to 42 (TrueType)
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['pdf.fonttype'] = 42

NUM_TRIALS=3
NUM_TRIALS=1

# Configuration Constants
INPUT_PKL = 'unit_condition_pair_info_df_with_processed_rates_slopes.pkl'
OUTPUT_PKL = 'unit_condition_pair_info_df_with_firing_rate_dependence_folded.pkl'
OUTPUT_DIR = f'firing_rate_dependence_plots_{NUM_TRIALS}TrialsPerPhase'
DISTANCE_DEPN_PLOT_FILENAME = 'firing_rate_distance_dependence_comparison_folded2.pdf'
CHANGE_RATE_PLOT_FILENAME = 'change_in_firing_rate_vs_distance2.pdf'

# Define scene distances
SCENE_DISTANCES = np.arange(-4, 6)  # -4 to 5 inclusive
NEGATIVE_INDICES = SCENE_DISTANCES < 0
POSITIVE_INDICES = SCENE_DISTANCES > 0
NEGATIVE_DISTANCES = SCENE_DISTANCES[NEGATIVE_INDICES]
POSITIVE_DISTANCES = SCENE_DISTANCES[POSITIVE_INDICES]

def get_custom_palette():
    
    # Base colors
    red = (1.0, 0.0, 0.0, 1.0)       # Opaque red
    red_transparent = (1.0, 0.0, 0.0, 0.6)  # Semi-transparent red

    brown = (0.545, 0.271, 0.075, 1.0)      # Opaque brown
    brown_transparent = (0.545, 0.271, 0.075, 0.6)  # Semi-transparent brown


    # Define the set of groups
    groups = {
        'HPC-structured-response', 'non-HPC-structured-response', 'non-HPC-random-scene', 
        'non-HPC-random-response', 'HPC-structured-scene', 'non-HPC-structured-scene', 
        'HPC-random-scene', 'HPC-random-response'
    }

    # Initialize the palette dictionary
    palette = {}

    for group in groups:
        # Determine if the group is HPC or non-HPC
        if 'HPC' in group:
            base_color = red
            base_color_transparent = red_transparent
        else:
            base_color = brown
            base_color_transparent = brown_transparent
        
        # Determine if the group is structured or random
        if 'structured' in group:
            color = base_color  # Opaque
        elif 'random' in group:
            color = base_color_transparent  # Semi-transparent
        else:
            color = base_color  # Default to opaque if neither
        
        # Assign the color to the group in the palette
        palette[group] = color
    return palette

def categorize_region(row):
    """
    Categorizes the region as 'HPC' or 'non-HPC'.
    
    Parameters:
    - row (pd.Series): A row from the DataFrame.
    
    Returns:
    - category (str): 'HPC' if region_name is 'HPC', else 'non-HPC'.
    """
    if row['region_name'].strip().lower() == 'hpc':
        return 'HPC'
    else:
        return 'non-HPC'

def fold_firing_rate_matrix(scene_firing_rate_matrix, response_firing_rate_matrix):
    """
    Folds the firing rate matrices by averaging negative and positive scene distances.
    
    Parameters:
    - scene_firing_rate_matrix (np.ndarray): 2D array with shape (num_trials, num_distances).
    - response_firing_rate_matrix (np.ndarray): 2D array with shape (num_trials, num_distances).
    
    Returns:
    - folded_scene_distances (np.ndarray): 1D array of folded scene distances.
    - folded_scene_firing_rates (np.ndarray): 2D array with shape (num_trials, num_folded_distances).
    - folded_response_firing_rates (np.ndarray): 2D array with shape (num_trials, num_folded_distances).
    """
    def fold_matrix(firing_rate_matrix):
        if not isinstance(firing_rate_matrix, np.ndarray):
            return np.nan, np.nan
        
        unique_distances = np.unique(SCENE_DISTANCES)
        folded_scene_distances = []
        folded_firing_rates = []
        
        for dist in sorted(unique_distances):
            if dist < 0:
                continue  # Handled by its positive counterpart
            elif dist == 0:
                # No counterpart, keep as is
                folded_scene_distances.append(dist)
                rates = firing_rate_matrix[:, SCENE_DISTANCES == dist].flatten()
                folded_firing_rates.append(rates)
            else:
                neg_dist = -dist
                if neg_dist in unique_distances:
                    rates_neg = firing_rate_matrix[:, SCENE_DISTANCES == neg_dist].flatten()
                    rates_pos = firing_rate_matrix[:, SCENE_DISTANCES == dist].flatten()
                    averaged_rates = np.nanmean(np.vstack((rates_neg, rates_pos)), axis=0)
                    folded_scene_distances.append(dist)
                    folded_firing_rates.append(averaged_rates)
                else:
                    # No negative counterpart, keep as is
                    folded_scene_distances.append(dist)
                    rates = firing_rate_matrix[:, SCENE_DISTANCES == dist].flatten()
                    folded_firing_rates.append(rates)
        
        if not folded_firing_rates:
            return np.nan, np.nan
        
        folded_firing_rates = np.vstack(folded_firing_rates).T  # Shape: (num_trials, num_folded_distances)
        folded_scene_distances = np.array(folded_scene_distances)
        
        return folded_scene_distances, folded_firing_rates

    folded_scene_distances, folded_scene_firing_rates = fold_matrix(scene_firing_rate_matrix)
    _, folded_response_firing_rates = fold_matrix(response_firing_rate_matrix)
    
    return folded_scene_distances, folded_scene_firing_rates, folded_response_firing_rates

def compute_slope(distances, rates):
    """
    Computes the slope of the best-fit line between distances and firing rates.
    
    Parameters:
    - distances (np.ndarray): 1D array of scene distances.
    - rates (np.ndarray): 1D array of firing rates corresponding to the distances.
    
    Returns:
    - slope (float): Slope of the best-fit line. Returns np.nan if insufficient data.
    """
    # Remove NaN values
    valid_mask = ~np.isnan(rates) & ~np.isnan(distances)
    valid_distances = distances[valid_mask]
    valid_rates = rates[valid_mask]
    
    # Ensure there are at least two points to compute the slope
    if len(valid_distances) < 2:
        return np.nan
    
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(valid_distances, valid_rates)
    return slope

def compute_slopes_folded(firing_rate_matrix, folded_scene_distances):
    """
    Computes the slope of the best-fit line for each trial based on folded firing rates.
    
    Parameters:
    - firing_rate_matrix (np.ndarray): 2D array with shape (num_trials, num_folded_distances).
    - folded_scene_distances (np.ndarray): 1D array of folded scene distances.
    
    Returns:
    - slopes (list): List of slopes for each trial.
    """
    if not isinstance(firing_rate_matrix, np.ndarray):
        return [np.nan]
    
    num_trials, num_distances = firing_rate_matrix.shape
    slopes = []
    
    for trial in range(num_trials):
        rates = firing_rate_matrix[trial, :]
        
        # Check for NaNs
        valid_mask = ~np.isnan(rates) & ~np.isnan(folded_scene_distances)
        valid_distances = folded_scene_distances[valid_mask]
        valid_rates = rates[valid_mask]
        
        if len(valid_distances) < 2:
            slopes.append(np.nan)
            continue
        
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = linregress(valid_distances, valid_rates)
        slopes.append(slope)
    
    return slopes

def compute_firing_rate_dependence(slopes_list, early_trials=3, late_trials=3):
    """
    Computes firing rate distance-dependence defined as the average of the last few trials' slopes
    minus the average of the first few trials' slopes.
    
    Parameters:
    - slopes_list (list): List of slopes for each trial.
    - early_trials (int): Number of early trials to average.
    - late_trials (int): Number of late trials to average.
    
    Returns:
    - dependence (float): The computed firing rate distance-dependence.
                           Returns np.nan if data is insufficient.
    """
    if not isinstance(slopes_list, list):
        return np.nan
    
    # Remove NaN slopes
    slopes_clean = [s for s in slopes_list if not np.isnan(s)]
    
    total_trials = len(slopes_clean)
    
    if total_trials < (early_trials + late_trials):
        # Need enough trials to compute dependence
        return np.nan
    
    early_avg = np.mean(slopes_clean[:(early_trials+1)])
    late_avg = np.mean(slopes_clean[-late_trials:])
    
    dependence = late_avg - early_avg
    return dependence

def compute_change_firing_rate(firing_rate_matrix_folded, folded_scene_distances, early_trials=2, last_trial_idx=-1):
    """
    Computes the change in firing rate from the average of the first few trials to the last trial
    for each absolute scene distance.
    
    Parameters:
    - firing_rate_matrix_folded (np.ndarray): 2D array with shape (num_trials, num_folded_distances).
    - folded_scene_distances (np.ndarray): 1D array of folded scene distances.
    - early_trials (int): Number of early trials to average.
    - last_trial_idx (int): Index of the last trial.
    
    Returns:
    - change_rates (np.ndarray or np.nan): 1D array of change in firing rates per distance.
                                           Returns np.nan if data is insufficient.
    """
    if not isinstance(firing_rate_matrix_folded, np.ndarray) or not isinstance(folded_scene_distances, np.ndarray):
        return np.nan
    
    num_trials, num_distances = firing_rate_matrix_folded.shape
    
    if num_trials < (early_trials + 1):
        return np.nan
    
    # Compute average of the first few trials
    avg_early = np.nanmean(firing_rate_matrix_folded[:(early_trials+1), :], axis=0)  # Shape: (num_distances,)
    
    # Compute average of the last few trials
    avg_late = np.nanmean(firing_rate_matrix_folded[last_trial_idx:, :], axis=0)  # Shape: (num_distances,)
    
    # Compute change in firing rate
    change_rates = avg_late - avg_early  # Shape: (num_distances,)
    
    return change_rates

def z_score_matrix(firing_rate_matrix):
    """
    Z-scores the firing rate matrix for each trial across distances.
    
    Parameters:
    - firing_rate_matrix (np.ndarray): 2D array with shape (num_trials, num_distances).
    
    Returns:
    - z_scored (np.ndarray or np.nan): Z-scored firing rate matrix.
    """
    #return firing_rate_matrix
    
    if not isinstance(firing_rate_matrix, np.ndarray):
        return np.nan
    # Compute z-score across distances for each trial
    mean = np.nanmean(firing_rate_matrix, axis=1, keepdims=True)
    std = np.nanstd(firing_rate_matrix, axis=1, keepdims=True)
    # Avoid division by zero
    std[std == 0] = 1
    z_scored = (firing_rate_matrix - mean) / std
    return z_scored

def visualize_firing_rate_dependence(master_df, output_dir, plot_filename):
    """
    Visualizes the firing rate distance-dependence within each region_category-condition group
    for both scene and response rates. Adds p-value annotations indicating significance from 0.
    
    Parameters:
    - master_df (pd.DataFrame): The DataFrame with firing rate dependence columns.
    - output_dir (str): Directory to save the visualization plots.
    - plot_filename (str): Filename for the saved plot.
    
    Returns:
    - None
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare DataFrame for plotting
    plot_df_scene = master_df.dropna(subset=['firing_rate_distance_dependence_scene'])
    plot_df_response = master_df.dropna(subset=['firing_rate_distance_dependence_response'])
    
    if plot_df_scene.empty and plot_df_response.empty:
        print("No data available for plotting firing rate distance dependence. Please check the DataFrame.")
        return
    
    # Define the grouping
    plot_df_scene['group'] = plot_df_scene['region_category'] + '-' + plot_df_scene['task_type'] + '-scene'
    plot_df_response['group'] = plot_df_response['region_category'] + '-' + plot_df_response['task_type'] + '-response'
    
    # Combine scene and response data
    combined_plot_df = pd.concat([plot_df_scene, plot_df_response], ignore_index=True)
    
    # Define the order of groups for consistent plotting
    group_order = [
        'HPC-structured-scene', 'HPC-random-scene',
        'non-HPC-structured-scene', 'non-HPC-random-scene',
        'HPC-structured-response', 'HPC-random-response',
        'non-HPC-structured-response', 'non-HPC-random-response'
    ]
    
    # Ensure that all groups are present in the data
    existing_groups = combined_plot_df['group'].unique()
    group_order = [group for group in group_order if group in existing_groups]
    
    if not group_order:
        print("No valid groups available for plotting firing rate distance dependence. Please check the data.")
        return
    
    
    # Set the aesthetic style of the plots
    sns.set(style="ticks")
    
    # Initialize the matplotlib figure with size (12,8)
    plt.figure(figsize=(12, 8))
    
    '''
    # Compute mean and SEM for each group
    summary_df_response = combined_plot_df.groupby('group')['firing_rate_distance_dependence_response'].agg(['mean', 'sem']).reset_index()
    summary_df_scene = combined_plot_df.groupby('group')['firing_rate_distance_dependence_scene'].agg(['mean', 'sem']).reset_index()
  
    pdb.set_trace()
    custom_palette = ['red', 'red', 'brown', 'brown']
    '''
    # Extract 'response_type' from 'group' name
    combined_plot_df['response_type'] = combined_plot_df['group'].apply(
        lambda x: 'response' if x.endswith('response') else 'scene'
    )

    # Assign firing rate based on 'response_type'
    def assign_firing_rate(row):
        if row['response_type'] == 'response':
            return row['firing_rate_distance_dependence_response']
        else:
            return row['firing_rate_distance_dependence_scene']

    combined_plot_df['firing_rate_distance_dependence'] = combined_plot_df.apply(assign_firing_rate, axis=1)

    # Create a long-format DataFrame
    long_df = combined_plot_df[['group', 'response_type', 'firing_rate_distance_dependence']]

    # Calculate mean and SEM
    aggregated_stats = long_df.groupby(['group', 'response_type'])['firing_rate_distance_dependence'].agg(['mean', 'sem']).reset_index()

    # Define custom palette: 'response' as red and 'scene' as brown
    custom_palette = get_custom_palette()
    ax = sns.barplot(x='group', y='mean', data=aggregated_stats, order=group_order, palette=custom_palette,
                     capsize=0.1, ci=None)

    ''' 
    # Create a bar plot without error bars
    ax = sns.barplot(x='group', y='mean', data=summary_df, order=group_order, palette='Set2',
                     capsize=0.1, ci=None)
    
    # Manually add error bars
    for i, patch in enumerate(ax.patches):
        # Get the center x position of the bar
        bar_x = patch.get_x() + patch.get_width() / 2
        # Get the height of the bar (mean)
        bar_height = patch.get_height()
        # Get the SEM for this group
        sem_val = summary_df['sem'].iloc[i]
        # Plot the error bar
        ax.errorbar(bar_x, bar_height, yerr=sem_val, fmt='none', c='black', capsize=5, linewidth=2)
    
    # Add a dashed black line at y=0 with linewidth 3
    plt.axhline(0, linestyle='--', color='black', linewidth=3)
    
    # Remove grid lines
    sns.despine()
    '''

    # Overlay individual data points as dots
    sns.stripplot(x='group', y='firing_rate_distance_dependence_response', data=combined_plot_df, order=group_order, 
                  color='black', alpha=0.5, jitter=True, size=5)
    
    # Perform one-sample t-tests for each group against 0
    p_values = {}
    for group in group_order:
        group_data = combined_plot_df[combined_plot_df['group'] == group]['firing_rate_distance_dependence_response']
        # Perform one-sample t-test
        t_stat, p_val = ttest_1samp(group_data, popmean=0, nan_policy='omit')
        p_values[group] = p_val
    
    # Define a function to map p-values to asterisks
    def get_asterisks(p):
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return 'n.s.'  # not significant
    
    # Compute asterisks for each group
    asterisks = {group: get_asterisks(p_val) for group, p_val in p_values.items()}
    
    '''
    # Determine the y-position for annotations (slightly above the max mean + SEM)
    y_max = summary_df_scene['mean'].max() + summary_df_scene['sem'].max()
    y_min = summary_df_scene['mean'].min() - summary_df_scene['sem'].min()
    y_offset = 0.05 * (y_max - y_min)  # 5% of the data range
    
    # Add asterisks above each bar
    for i, group in enumerate(group_order):
        if 'scene' in group:
            summary_df = summary_df_scene
        else:
            summary_df = summary_df_response
            
        x = i
        y = summary_df.loc[summary_df['group'] == group, 'mean'].values[0] + summary_df.loc[summary_df['group'] == group, 'sem'].values[0] + y_offset
        ax.text(x, y, asterisks[group], ha='center', va='bottom', fontsize=14, color='black')
    
    # Set plot labels and title
    plt.ylabel('Change in (Firing Rate vs Scene Distance) Slope \n(Hz/scene, Last - Early)', fontsize=14)
    plt.xlabel('Group', fontsize=14)
    plt.title('Firing Rate Distance-Dependence Across Groups and Rates', fontsize=16)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    '''
    # Adjust layout for better fit
    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(output_dir, plot_filename)
    plt.savefig(save_path,format='pdf')
    plt.close()
    
    print(f"Saved firing rate distance-dependence plot with p-values at '{save_path}'.")

def visualize_change_in_firing_rate(master_df, output_dir, plot_filename):
    """
    Visualizes the change in firing rates from the average of the first few trials to the last trial
    as a function of absolute scene distance for both scene and response rates.
    
    Parameters:
    - master_df (pd.DataFrame): The DataFrame with change firing rate columns.
    - output_dir (str): Directory to save the visualization plots.
    - plot_filename (str): Filename for the saved plot.
    
    Returns:
    - None
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare DataFrame for plotting
    plot_df_scene = master_df.dropna(subset=['change_firing_rate_scene'])
    plot_df_response = master_df.dropna(subset=['change_firing_rate_response'])
    
    if plot_df_scene.empty and plot_df_response.empty:
        print("No data available for plotting change in firing rates. Please check the DataFrame.")
        return
    
    # Expand the 'change_firing_rate' into separate rows for scene and response
    def expand_change_rates(row, rate_type):
        if rate_type == 'scene':
            change_rates = row['change_firing_rate_scene']
            folded_distances = row['folded_scene_distances']
        else:
            change_rates = row['change_firing_rate_response']
            folded_distances = row['folded_scene_distances']  # Assuming distances are same for response
                
        if not isinstance(change_rates, np.ndarray) or not isinstance(folded_distances, np.ndarray):
            return []
        
        # Exclude distance 0
        mask = folded_distances > 0
        distances = folded_distances[mask]
        changes = change_rates[mask]
        
        # Create a list of dictionaries
        expanded = []
        for dist, change in zip(distances, changes):
            expanded.append({
                'group': row['region_category'] + '-' + row['task_type'] + f'-{rate_type}',
                'distance': dist,
                'change_firing_rate': change
            })
        return expanded
    
    # Apply the expansion
    expanded_data_scene = plot_df_scene.apply(lambda row: expand_change_rates(row, 'scene'), axis=1)
    expanded_data_response = plot_df_response.apply(lambda row: expand_change_rates(row, 'response'), axis=1)
    
    # Combine the expanded data
    expanded_data_scene = [item for sublist in expanded_data_scene for item in sublist]
    expanded_data_response = [item for sublist in expanded_data_response for item in sublist]
    combined_data = expanded_data_scene + expanded_data_response
    expanded_df = pd.DataFrame(combined_data)
    
    if expanded_df.empty:
        print("No valid data after expanding for plotting change in firing rates.")
        return
    
    # Define the grouping
    group_order = [
        'HPC-structured-scene', 'HPC-random-scene',
        'non-HPC-structured-scene', 'non-HPC-random-scene',
        'HPC-structured-response', 'HPC-random-response',
        'non-HPC-structured-response', 'non-HPC-random-response'
    ]
    
    existing_groups = expanded_df['group'].unique()
    group_order = [group for group in group_order if group in existing_groups]
    
    if not group_order:
        print("No valid groups available for plotting change in firing rates. Please check the data.")
        return
    
    # Set the aesthetic style of the plots
    sns.set(style="ticks")
    
    '''
    # Assuming expanded_df is your DataFrame
    groups = expanded_df['group'].unique()  # Get unique groups

    # Initialize the matplotlib figure with subplots, one for each group
    fig, axes = plt.subplots(len(groups), 1, figsize=(4, len(groups) * 4), sharex=True)  # Create a subplot for each group

    # If there is only one group, axes will not be an array, so handle that case
    if len(groups) == 1:
        axes = [axes]

    # Iterate over the groups and create a separate point plot for each group
    for i, group in enumerate(groups):
        sns.pointplot(
            data=expanded_df[expanded_df['group'] == group],
            x='distance',
            y='change_firing_rate',
            palette='Set2',
            errorbar='se',  # Standard Error
            dodge=0.3,
            markers='o',
            capsize=0.1,
            ax=axes[i]
        )
        axes[i].set_title(f'Group: {group}')
        # Add a horizontal line at y=0
        axes[i].axhline(0, linestyle='--', color='black', linewidth=2)
        axes[i].set_ylim([-3,3])
    '''
    # Assuming expanded_df is your DataFrame
    groups = expanded_df['group'].unique()  # Get unique groups
    groups=groups[:4] #just take scene rate changes
    # Set the number of rows and columns for the subplots
    num_groups = len(groups)
    rows = (num_groups + 1) // 2  # Number of rows needed (round up if odd number of groups)
    cols = 2  # Number of columns
    
    # Initialize the matplotlib figure with the appropriate number of subplots
    fig, axes = plt.subplots(rows, cols, figsize=(8, rows * 4), sharex=True)
    
    # Flatten the axes array if there is more than one row
    axes = axes.flatten()
   
    # Iterate over the groups and create a separate point plot for each group
    for i, group in enumerate(groups):
        sns.pointplot(
            data=expanded_df[expanded_df['group'] == group],
            x='distance',
            y='change_firing_rate',
            palette='viridis',
            errorbar='se',  # Standard Error
            dodge=0.3,
            markers='o',
            capsize=0.1,
            ax=axes[i]
        )
        axes[i].set_title(f'Group: {group}')
        axes[i].axhline(0, linestyle='--', color='black', linewidth=2)
        axes[i].set_ylim([-2.5,2.5])
        axes[i].set_ylim([-1,1])

        axes[i].set_xlabel('Absolute Scene Distance', fontsize=14)
        axes[i].set_ylabel('Change in Firing Rate (Z)', fontsize=14)
    
    # Hide any empty subplots if there are fewer groups than subplots
    for i in range(num_groups, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()  # Adjust layout to avoid overlap
    plt.suptitle(f'{NUM_TRIALS} trials per phase (+1 early baseline)')
    
    # Set plot labels and title
    
    # Adjust legend
    plt.legend(title='Group', fontsize=12, title_fontsize=12)
    
    # Adjust layout for better fit
    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(output_dir, plot_filename)
    plt.savefig(save_path,format='pdf')
    plt.close()
    
    print(f"Saved change in firing rate vs distance plot at '{save_path}'.")

def main():
    """
    Main function to load the DataFrame, fold firing rates, compute slopes based on folded data,
    compute firing rate distance-dependence, compute change in firing rates, visualize them,
    and save the updated DataFrame.
    """
    # Load the DataFrame
    try:
        master_df = pd.read_pickle(INPUT_PKL)
        print(f"Successfully loaded '{INPUT_PKL}'.")
    except FileNotFoundError:
        print(f"Error: '{INPUT_PKL}' not found.")
        return
    except Exception as e:
        print(f"An error occurred while loading the DataFrame: {e}")
        return
    
    # Verify necessary columns exist
    required_columns = [
        'region_name', 'task_type', 
        'smoothed_scene_rate_per_trial_per_distance', 
        'smoothed_response_rate_per_trial_per_distance',
        'slope_neg_scene', 'slope_pos_scene',
        'slope_neg_response', 'slope_pos_response'
    ]
    missing_columns = [col for col in required_columns if col not in master_df.columns]
    if missing_columns:
        print(f"Error: Required columns {missing_columns} not found in the DataFrame.")
        return
    
    # Categorize regions
    master_df['region_category'] = master_df.apply(categorize_region, axis=1)
    print("Added 'region_category' column.")
    
    # --- Begin Z-Score Normalization ---
    # Z-score non-folded firing rate matrices
    master_df['z_scored_scene_rate_per_trial_per_distance'] = master_df['smoothed_scene_rate_per_trial_per_distance'].apply(z_score_matrix)
    master_df['z_scored_response_rate_per_trial_per_distance'] = master_df['smoothed_response_rate_per_trial_per_distance'].apply(z_score_matrix)
    print("Z-scored non-folded firing rate matrices.")
    # --- End Z-Score Normalization ---
    
    # Fold the firing rate matrices using z-scored data
    print("Folding the firing rate matrices by averaging negative and positive scene distances...")
    folded_data = master_df.apply(
        lambda row: fold_firing_rate_matrix(
            row['z_scored_scene_rate_per_trial_per_distance'], 
            row['z_scored_response_rate_per_trial_per_distance']
        ),
        axis=1
    )
    master_df['folded_scene_distances'] = folded_data.apply(lambda x: x[0] if isinstance(x, tuple) else np.nan)
    master_df['folded_scene_firing_rates'] = folded_data.apply(lambda x: x[1] if isinstance(x, tuple) else np.nan)
    master_df['folded_response_firing_rates'] = folded_data.apply(lambda x: x[2] if isinstance(x, tuple) else np.nan)
    print("Completed folding of firing rate matrices.")
    
    # --- Begin Z-Score Normalization for Folded Data ---
    master_df['z_scored_folded_scene_firing_rates'] = master_df['folded_scene_firing_rates']#TEST.apply(z_score_matrix)
    master_df['z_scored_folded_response_firing_rates'] = master_df['folded_response_firing_rates']#TEST.apply(z_score_matrix)
    print("Z-scored folded firing rate matrices.")
    # --- End Z-Score Normalization for Folded Data ---
    
    # Compute slopes based on folded z-scored data
    print("Computing slopes for folded scene firing rates...")
    master_df['slopes_folded_scene'] = master_df.apply(
        lambda row: compute_slopes_folded(row['z_scored_folded_scene_firing_rates'], row['folded_scene_distances']) 
                    if isinstance(row['z_scored_folded_scene_firing_rates'], np.ndarray) else [np.nan],
        axis=1
    )
    print("Completed computing slopes for folded scene firing rates.")
    
    print("Computing slopes for folded response firing rates...")
    master_df['slopes_folded_response'] = master_df.apply(
        lambda row: compute_slopes_folded(row['z_scored_folded_response_firing_rates'], row['folded_scene_distances']) 
                    if isinstance(row['z_scored_folded_response_firing_rates'], np.ndarray) else [np.nan],
        axis=1
    )
    print("Completed computing slopes for folded response firing rates.")
    
    # Compute firing rate distance-dependence for scene and response
    print("Computing firing rate distance-dependence for scene rates...")
    master_df['firing_rate_distance_dependence_scene'] = master_df['slopes_folded_scene'].apply(
        lambda slopes: compute_firing_rate_dependence(slopes, early_trials=NUM_TRIALS, late_trials=NUM_TRIALS) 
                    if isinstance(slopes, list) else np.nan
    )
    print("Added 'firing_rate_distance_dependence_scene' column.")
    
    print("Computing firing rate distance-dependence for response rates...")
    master_df['firing_rate_distance_dependence_response'] = master_df['slopes_folded_response'].apply(
        lambda slopes: compute_firing_rate_dependence(slopes, early_trials=NUM_TRIALS, late_trials=NUM_TRIALS) 
                    if isinstance(slopes, list) else np.nan
    )
    print("Added 'firing_rate_distance_dependence_response' column.")
    
    # Compute change in firing rate from early trials to last trial for scene and response
    print("Computing change in firing rates for scene rates...")
    master_df['change_firing_rate_scene'] = master_df.apply(
        lambda row: compute_change_firing_rate(row['z_scored_folded_scene_firing_rates'], row['folded_scene_distances'], early_trials=NUM_TRIALS, last_trial_idx=-NUM_TRIALS) 
                    if isinstance(row['z_scored_folded_scene_firing_rates'], np.ndarray) else np.nan,
        axis=1
    )
    print("Added 'change_firing_rate_scene' column.")
    
    print("Computing change in firing rates for response rates...")
    master_df['change_firing_rate_response'] = master_df.apply(
        lambda row: compute_change_firing_rate(row['z_scored_folded_response_firing_rates'], row['folded_scene_distances'], early_trials=NUM_TRIALS, last_trial_idx=-NUM_TRIALS) 
                    if isinstance(row['z_scored_folded_response_firing_rates'], np.ndarray) else np.nan,
        axis=1
    )
    print("Added 'change_firing_rate_response' column.")
    
    # Save the updated DataFrame with all computations
    try:
        master_df.to_pickle(OUTPUT_PKL)
        print(f"Saved the updated DataFrame with firing rate dependence as '{OUTPUT_PKL}'.")
    except Exception as e:
        print(f"An error occurred while saving the DataFrame: {e}")
        return
    
    # Visualize the firing rate distance-dependence for scene and response
    visualize_firing_rate_dependence(master_df, output_dir=OUTPUT_DIR, plot_filename=DISTANCE_DEPN_PLOT_FILENAME)
    print("Completed visualization of firing rate distance-dependence.")
    
    # Visualize the change in firing rates vs distance for scene and response
    visualize_change_in_firing_rate(master_df, output_dir=OUTPUT_DIR, plot_filename=CHANGE_RATE_PLOT_FILENAME)
    print("Completed visualization of change in firing rates vs distance.")
    
    # Optional: Remove intermediate folded data columns to save space
    master_df.drop(['folded_scene_distances', 'folded_scene_firing_rates', 'folded_response_firing_rates',
                   'slopes_folded_scene', 'slopes_folded_response'], axis=1, inplace=True)
    
    # Save the cleaned DataFrame without intermediate columns
    try:
        master_df.to_pickle(OUTPUT_PKL)
        print(f"Saved the cleaned DataFrame without intermediate folded data as '{OUTPUT_PKL}'.")
    except Exception as e:
        print(f"An error occurred while saving the cleaned DataFrame: {e}")
        return

if __name__ == "__main__":
    main()

