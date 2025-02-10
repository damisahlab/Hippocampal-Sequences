import pdb
import numpy as np
import pandas as pd
from scipy.stats import linregress, wilcoxon, sem, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
import os

import matplotlib as mpl
# Set font type to 42 (TrueType)
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['pdf.fonttype'] = 42

N_TRIALS_PER_PHASE=1
#N_TRIALS_PER_PHASE=2
#N_TRIALS_PER_PHASE=3
TWO_SIDED_STATS=True

# Configuration Constants
INPUT_PKL = 'unit_condition_pair_info_df_with_slopes.pkl'
OUTPUT_PKL = 'unit_condition_pair_info_df_with_firing_rate_dependence.pkl'  # Updated to generalize
#OUTPUT_DIR = 'firing_rate_dependence_plots'
OUTPUT_DIR = f'firing_rate_dependence_plots_{N_TRIALS_PER_PHASE}TrialsPerPhase'
PLOT_FILENAME_FOLDED = 'firing_rate_distance_dependence_comparison_folded.pdf'
PLOT_FILENAME_POSITIVE = 'firing_rate_distance_dependence_comparison_positive.pdf'
PLOT_FILENAME_NEGATIVE = 'firing_rate_distance_dependence_comparison_negative.pdf'
CHANGE_PLOT_FILENAME = 'change_in_firing_rate_vs_distance.pdf'  # New Plot Filename
DISTRIBUTION_PLOT_FILENAME = 'firing_rate_distribution_folded.pdf'  # New Distribution Plot Filename

def get_custom_palette():
    # Base colors
    red = '#FF0000'                # Opaque red
    red_transparent = '#FF000099' # Semi-transparent red
    lightcoral = '#F08080'         # Light Coral
    brown = '#8B4513'               # Opaque brown
    brown_transparent = '#8B451399' # Semi-transparent brown
    light_brown = '#D2B48C'         # Light Brown (Tan)

    # Define the set of groups
    groups = {
        'HPC-structured', 'non-HPC-structured', 'non-HPC-random',
        'non-HPC-random', 'HPC-structured', 'non-HPC-structured',
        'HPC-random', 'HPC-random'
    }

    # Initialize the palette dictionary
    palette = {}

    for group in groups:
        # Determine if the group is HPC or non-HPC
        if 'non-HPC' not in group:
            base_color = red
            base_color_transparent = lightcoral
        else:
            base_color = brown
            base_color_transparent = light_brown

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

custom_palette = get_custom_palette()

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

def fold_firing_rate_matrix(firing_rate_matrix, scene_distances):
    """
    Folds the firing rate matrix by averaging negative and positive scene distances.

    Parameters:
    - firing_rate_matrix (np.ndarray): 2D array with shape (num_trials, num_distances).
    - scene_distances (np.ndarray): 1D array of scene distances corresponding to distances axis.

    Returns:
    - folded_scene_distances (np.ndarray): 1D array of folded scene distances.
    - folded_firing_rates (np.ndarray): 2D array with shape (num_trials, num_folded_distances).
    """
    if not isinstance(firing_rate_matrix, np.ndarray):
        return np.nan, np.nan

    # Define unique scene distances
    unique_distances = np.unique(scene_distances)

    # Initialize lists for folded distances and folded firing rates
    folded_scene_distances = []
    folded_firing_rates = []

    # Handle symmetric distances
    # Iterate over positive distances including zero
    for dist in sorted(unique_distances):
        if dist < 0:
            continue  # Will be handled by its positive counterpart
        elif dist == 0:
            # No counterpart, keep as is
            folded_scene_distances.append(dist)
            firing_rates = firing_rate_matrix[:, scene_distances == dist].flatten()
            folded_firing_rates.append(firing_rates)
        elif dist > 0:
            # Check if negative counterpart exists
            neg_dist = -dist
            if neg_dist in unique_distances:
                # Average negative and positive firing rates
                rates_neg = firing_rate_matrix[:, scene_distances == neg_dist].flatten()
                rates_pos = firing_rate_matrix[:, scene_distances == dist].flatten()
                averaged_rates = np.nanmean(np.vstack((rates_neg, rates_pos)), axis=0)
                folded_scene_distances.append(dist)
                folded_firing_rates.append(averaged_rates)
            else:
                # No negative counterpart, keep as is
                folded_scene_distances.append(dist)
                firing_rates = firing_rate_matrix[:, scene_distances == dist].flatten()
                folded_firing_rates.append(firing_rates)

    # Convert lists to arrays
    folded_firing_rates = np.vstack(folded_firing_rates)  # Shape: (num_folded_distances, num_trials)
    folded_firing_rates = folded_firing_rates.T  # Shape: (num_trials, num_folded_distances)
    folded_scene_distances = np.array(folded_scene_distances)

    return folded_scene_distances, folded_firing_rates

def compute_slopes_folded(row, folded_scene_distances):
    """
    Computes the slope of the best-fit line for each trial based on folded firing rates.

    Parameters:
    - row (pd.Series): A row from the DataFrame.
    - folded_scene_distances (np.ndarray): 1D array of folded scene distances.

    Returns:
    - slopes (list): List of slopes for each trial.
    """
    folded_firing_rates = row['firing_rate_matrix_folded']

    if not isinstance(folded_firing_rates, np.ndarray):
        return [np.nan]

    num_trials, num_distances = folded_firing_rates.shape
    slopes = []

    for trial in range(num_trials):
        rates = folded_firing_rates[trial, :]

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

def compute_firing_rate_dependence(row):
    """
    Computes the firing rate distance-dependence for a unit.

    Defined as the average of the last trial's slope minus the average of the first two trials' slopes.

    Parameters:
    - row (pd.Series): A row from the DataFrame.

    Returns:
    - dependence (float): The computed firing rate distance-dependence.
                           Returns np.nan if data is insufficient.
    """
    slopes = row['slopes_folded']

    if not isinstance(slopes, list):
        return np.nan

    # Remove NaN slopes
    slopes_clean = [s for s in slopes if not np.isnan(s)]

    if len(slopes_clean) < 15:
        # Need at least 15 trials to have first 2 and last 1
        return np.nan

    #late_avg = np.mean(slopes_clean[-N_TRIALS_PER_PHASE:])  # Averaging last trial
    #early_avg = np.mean(slopes_clean[:N_TRIALS_PER_PHASE])  # Note: Averaging first 3 trials as per user code
    late_avg = np.mean(slopes_clean[-N_TRIALS_PER_PHASE:])  # Averaging last trial
    early_avg = np.mean(slopes_clean[:(N_TRIALS_PER_PHASE+1)])  # Note: Averaging first 3 trials as per user code
   
    dependence = late_avg - early_avg
    return dependence

def compute_change_firing_rate(row):
    """
    Computes the change in firing rate from the average of the first two trials to the last trial
    for each absolute scene distance.

    Parameters:
    - row (pd.Series): A row from the DataFrame.

    Returns:
    - change_rates (np.ndarray or np.nan): 1D array of change in firing rates per distance.
                                           Returns np.nan if data is insufficient.
    """
    firing_rates = row['firing_rate_matrix_folded']
    folded_scene_distances = row['folded_scene_distances']

    if not isinstance(firing_rates, np.ndarray) or not isinstance(folded_scene_distances, np.ndarray):
        return np.nan

    if firing_rates.shape[0] < 15:
        return np.nan

    # Compute average of the first two trials
    avg_first_two = np.nanmean(firing_rates[0:2, :], axis=0)  # Shape: (num_distances,)
    last_trial = firing_rates[14, :]  # Shape: (num_distances,)

    # Compute change in firing rate
    change_rates = last_trial - avg_first_two  # Shape: (num_distances,)

    return change_rates

def format_pval_pairwise(p,stat,statNameStr):
    """
    Formats the p-value for pairwise comparisons.

    Parameters:
    - p (float): The p-value to format.

    Returns:
    - formatted_p (str): 'n.s.' if p >= 0.05, else 'p = x.xxxxxx'.
    """
    if np.isnan(p):
        return 'n.s.'
    elif p >= 0.05:
        return f'n.s. (p = {p:.4f}), {statNameStr} = {stat:.2f}'
    else:
        return f'p = {p:.5f}, {statNameStr} = {stat:.2f}'

def visualize_firing_rate_dependence(master_df, distance_type='folded', output_dir=OUTPUT_DIR, plot_filename=PLOT_FILENAME_FOLDED):
    """
    Visualizes the firing rate distance-dependence within each region_category-condition group.
    Adds p-value annotations indicating significance from 0 and pairwise comparisons.

    Parameters:
    - master_df (pd.DataFrame): The DataFrame with 'firing_rate_distance_dependence' column.
    - distance_type (str): Type of distance analysis ('folded', 'positive', 'negative').
    - output_dir (str): Directory to save the visualization plots.
    - plot_filename (str): Filename for the saved plot.

    Returns:
    - None
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define column names based on distance_type
    dependence_col = 'firing_rate_distance_dependence'
    if distance_type == 'folded':
        dependence_col = 'firing_rate_distance_dependence_folded'
        group_col = 'group_folded'
    elif distance_type == 'positive':
        dependence_col = 'firing_rate_distance_dependence_positive'
        group_col = 'group_positive'
    elif distance_type == 'negative':
        dependence_col = 'firing_rate_distance_dependence_negative'
        group_col = 'group_negative'
    else:
        raise ValueError("distance_type must be 'folded', 'positive', or 'negative'")

    # Drop rows with NaN firing_rate_distance_dependence
    plot_df = master_df.dropna(subset=[dependence_col])

    if plot_df.empty:
        print(f"No data available for plotting firing rate distance dependence ({distance_type}). Please check the DataFrame.")
        return

    # Define the grouping
    plot_df['group'] = plot_df['region_category'] + '-' + plot_df['task_type']

    # Define the order of groups for consistent plotting
    group_order = ['HPC-structured', 'HPC-random', 'non-HPC-structured', 'non-HPC-random']

    # Ensure that all groups are present in the data
    existing_groups = plot_df['group'].unique()
    group_order = [group for group in group_order if group in existing_groups]

    if not group_order:
        print(f"No valid groups available for plotting firing rate distance dependence ({distance_type}). Please check the data.")
        return

    # Set the aesthetic style of the plots
    sns.set(style="ticks")

    # Initialize the matplotlib figure with size (6,8)
    plt.figure(figsize=(6, 8))

    # Compute mean and SEM for each group
    summary_df = plot_df.groupby('group')[dependence_col].agg(['mean', 'sem']).reset_index()

    # Create a bar plot without error bars
    ax = sns.barplot(x='group', y='mean', data=summary_df, order=group_order, palette=custom_palette,
                    capsize=0.1, errorbar=None)

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

    # Add a dashed black line at y=0 with linewidth 4
    plt.axhline(0, linestyle='--', color='black', linewidth=4)

    # Remove grid lines
    sns.despine()

    # Overlay individual data points as dots
    sns.stripplot(x='group', y=dependence_col, data=plot_df, order=group_order, 
                color='black', alpha=0.5, jitter=True, size=5)

    '''
    # Perform one-sample Wilcoxon signed-rank tests for each group against 0
    p_values_one_sample = {}
    for group in group_order:
        group_data = plot_df[plot_df['group'] == group][dependence_col]
        # Perform Wilcoxon signed-rank test
        try:
            stat, p_val = wilcoxon(group_data - 0)  # Testing median difference from 0
        except ValueError:
            p_val = np.nan  # If all values are zero or insufficient data
        p_values_one_sample[group] = (p_val, stat)

    # Define a function to format p-values for one-sample tests
    def format_pval_one_sample(p):
        if np.isnan(p):
            return 'p = NaN'
        elif p < 0.0001:
            return 'p < 0.0001'
        else:
            return f'p = {p:.4f}'
    '''
    # Compute formatted p-values for each group
    #formatted_pvals_one_sample = {group: format_pval_one_sample(p_val) for group, p_val in p_values_one_sample.items()}

    # Perform one-sample Wilcoxon signed-rank tests for each group against 0
    p_values_one_sample = {}
    for group in group_order:
        group_data = plot_df[plot_df['group'] == group][dependence_col]
        # Perform Wilcoxon signed-rank test
        try:
            if TWO_SIDED_STATS:
                stat, p_val = wilcoxon(group_data - 0)  # Testing median difference from 0
            else:
                stat, p_val = wilcoxon(group_data - 0,alternative='less')  # Testing median difference from 0
        except ValueError:
            p_val = np.nan  # If all values are zero or insufficient data
            stat = np.nan  # Ensure stat is also set to NaN
        p_values_one_sample[group] = (p_val, stat)

    
    # Compute formatted p-values for each group using correct unpacking
    formatted_pvals_one_sample = {
        group: format_pval_pairwise(p_val, stat,'W') 
        for group, (p_val, stat) in p_values_one_sample.items()
        }

    # Determine the y-position for annotations (slightly above the max mean + SEM)
    y_max = summary_df['mean'].max() + summary_df['sem'].max()
    y_min = summary_df['mean'].min() - summary_df['sem'].min()
    y_offset = 0.05 * (y_max - y_min)  # 5% of the data range

    #MAX_Y = 0.6  
    MAX_Y = 1.7  
    MAX_Y = 1.5  

    # Add p-value annotations above each bar for one-sample tests
    for i, group in enumerate(group_order):
        x = i
        y = summary_df.loc[summary_df['group'] == group, 'mean'].values[0] + summary_df.loc[summary_df['group'] == group, 'sem'].values[0] + y_offset
        # Ensure that y does not exceed MAX_Y
        y = min(y, MAX_Y*0.9)
        # Add the formatted p-value text
        ax.text(x, y, formatted_pvals_one_sample[group], ha='center', va='bottom', fontsize=6, color='black')

    # ----- Begin Pairwise Mann-Whitney U Tests and Annotations -----
    # Define the pairs for comparison within each region
    pairwise_pairs = [
        ('HPC-structured', 'HPC-random'),
        ('non-HPC-structured', 'non-HPC-random')
    ]

    # Initialize a list to store pairwise p-values and their corresponding pairs
    pairwise_pvals = []

    for structured_group, random_group in pairwise_pairs:
        structured_data = plot_df[plot_df['group'] == structured_group][dependence_col]
        random_data = plot_df[plot_df['group'] == random_group][dependence_col]

        # Perform Mann-Whitney U test
        try:
            if TWO_SIDED_STATS:
                stat, p_val = mannwhitneyu(structured_data, random_data, alternative='two-sided')
            else:
                stat, p_val = mannwhitneyu(structured_data, random_data, alternative='less')
        except ValueError:
            p_val = np.nan  # If data is insufficient
        pairwise_pvals.append((structured_group, random_group, p_val,stat))

    # Function to convert p-value to formatted string with 'n.s.'
    # Already defined above as format_pval_pairwise

    # Annotate the pairwise p-values on the plot
    for pair in pairwise_pvals:
        group1, group2, p_val,stat = pair
        formatted_pval = format_pval_pairwise(p_val,stat,'U')

        # Get the indices of the groups
        try:
            x1 = group_order.index(group1)
            x2 = group_order.index(group2)
        except ValueError:
            continue  # If groups are not found, skip

        # Calculate the y position for the annotation
        height1 = summary_df.loc[summary_df['group'] == group1, 'mean'].values[0] + summary_df.loc[summary_df['group'] == group1, 'sem'].values[0]
        height2 = summary_df.loc[summary_df['group'] == group2, 'mean'].values[0] + summary_df.loc[summary_df['group'] == group2, 'sem'].values[0]
        y = max(height1, height2) + y_offset + 0.05  # Slightly above the higher bar

        # Draw the line
        ax.plot([x1, x1, x2, x2], [y, y+0.02, y+0.02, y], lw=1.5, c='black')

        # Add the formatted p-value text
        ax.text((x1 + x2) * 0.5, y + 0.02, formatted_pval, ha='center', va='bottom', color='black', fontsize=6)
    # ----- End Pairwise Mann-Whitney U Tests and Annotations -----

    if TWO_SIDED_STATS:
        statsStr=f'{N_TRIALS_PER_PHASE} trials per phase (+1 early baseline); two-sided wilcoxon, mannwhitney'
    else:
        statsStr=f'{N_TRIALS_PER_PHASE} trials per phase (+1 early baseline); one-sided wilcoxon, mannwhitney'
    
    # Set plot labels and title
    if distance_type == 'folded':
        plt.ylabel('Change in (Firing Rate vs Scene Distance) Slope \n(Hz/scene, Late - Early)', fontsize=11)
        plt.title(f'Firing Rate Distance Dependence (Folded)\n{statsStr}', fontsize=11)
    elif distance_type == 'positive':
        plt.ylabel('Change in (Firing Rate vs Positive Scene Distance) Slope \n(Hz/scene, Late - Early)', fontsize=11)
        plt.title(f'Firing Rate Distance Dependence (Positive)\n{statsStr}', fontsize=11)
    elif distance_type == 'negative':
        plt.ylabel('Change in (Firing Rate vs Negative Scene Distance) Slope \n(Hz/scene, Late - Early)', fontsize=11)
        plt.title(f'Firing Rate Distance Dependence (Negative)\n{statsStr}', fontsize=11)

    ax.set_ylim([-MAX_Y, MAX_Y])
    #ax.set_ylim([-0.575, 0.375])
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Adjust layout for better fit
    plt.tight_layout()

    # Save the plot
    save_path = os.path.join(output_dir, plot_filename)
    plt.savefig(save_path)
    plt.close()

    print(f"Saved firing rate distance-dependence ({distance_type}) plot with p-values at '{save_path}'.")

def visualize_firing_rate_distribution(master_df, distance_type='folded', output_dir=OUTPUT_DIR, plot_filename=DISTRIBUTION_PLOT_FILENAME):
    """
    Visualizes the distribution of firing rate distance-dependence as point clouds for each group.
    Adds a black horizontal line at y=0 and applies jitter to prevent overlapping.

    Parameters:
    - master_df (pd.DataFrame): The DataFrame with 'firing_rate_distance_dependence_folded' column.
    - distance_type (str): Type of distance analysis ('folded', 'positive', 'negative').
    - output_dir (str): Directory to save the distribution plot.
    - plot_filename (str): Filename for the saved distribution plot.

    Returns:
    - None
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define column names based on distance_type
    dependence_col = 'firing_rate_distance_dependence_folded'
    if distance_type == 'folded':
        dependence_col = 'firing_rate_distance_dependence_folded'
    elif distance_type == 'positive':
        dependence_col = 'firing_rate_distance_dependence_positive'
    elif distance_type == 'negative':
        dependence_col = 'firing_rate_distance_dependence_negative'
    else:
        raise ValueError("distance_type must be 'folded', 'positive', or 'negative'")

    # Drop rows with NaN firing_rate_distance_dependence
    plot_df = master_df.dropna(subset=[dependence_col])

    if plot_df.empty:
        print(f"No data available for plotting firing rate distribution ({distance_type}). Please check the DataFrame.")
        return

    # Define the grouping
    plot_df['group'] = plot_df['region_category'] + '-' + plot_df['task_type']

    # Define the order of groups for consistent plotting
    group_order = ['HPC-structured', 'HPC-random', 'non-HPC-structured', 'non-HPC-random']

    # Ensure that all groups are present in the data
    existing_groups = plot_df['group'].unique()
    group_order = [group for group in group_order if group in existing_groups]

    if not group_order:
        print(f"No valid groups available for plotting firing rate distribution ({distance_type}). Please check the DataFrame.")
        return

    # Set the aesthetic style of the plots
    sns.set(style="ticks")

    # Initialize the matplotlib figure with size (6, 8)
    plt.figure(figsize=(6, 8))
    plt.figure(figsize=(8, 10))

    # Create a strip plot with jitter
    ax = sns.stripplot(x='group', y=dependence_col, data=plot_df, order=group_order, palette=custom_palette,
                       jitter=0.5, dodge=True, alpha=0.6, size=10)

    # Add a black horizontal line at y=0
    plt.axhline(0, linestyle='--', color='black', linewidth=2)

    # Remove grid lines
    sns.despine()

    # Set plot labels and title
    plt.ylabel('Change in Firing Rate Distance Dependence (Hz)', fontsize=14)
    plt.title('Distribution of Firing Rate Distance Dependence (Folded)', fontsize=16)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Set y-axis limits to match the main plot
    plt.ylim([-1.5, 1.6])

    # Adjust layout for better fit
    plt.tight_layout()

    # Save the plot
    save_path = os.path.join(output_dir, plot_filename)
    plt.savefig(save_path)
    plt.close()

    print(f"Saved firing rate distribution plot at '{save_path}'.")

def visualize_change_firing_rate(master_df, output_dir=OUTPUT_DIR, plot_filename=CHANGE_PLOT_FILENAME):
    """
    Visualizes the change in firing rates from the average of the first two trials to the last trial
    as a function of absolute scene distance. The x-axis represents scene distances (1-5),
    and the y-axis represents the change in firing rate.

    Parameters:
    - master_df (pd.DataFrame): The DataFrame with 'change_firing_rate_per_distance' column.
    - output_dir (str): Directory to save the visualization plots.
    - plot_filename (str): Filename for the saved plot.

    Returns:
    - None
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Drop rows with NaN change_firing_rate_per_distance
    plot_df = master_df.dropna(subset=['change_firing_rate_per_distance'])

    if plot_df.empty:
        print("No data available for plotting change in firing rates. Please check the DataFrame.")
        return

    # Expand the 'change_firing_rate_per_distance' into separate rows
    # Assuming 'folded_scene_distances' includes distance 0, we exclude it for distances 1-5
    expanded_data = []
    for _, row in plot_df.iterrows():
        change_rates = row['change_firing_rate_per_distance']
        folded_distances = row['folded_scene_distances']
        # Exclude distance 0
        mask = folded_distances > 0
        distances = folded_distances[mask]
        changes = change_rates[mask]
        for dist, change in zip(distances, changes):
            expanded_data.append({
                'group': row['region_category'] + '-' + row['task_type'],
                'distance': dist,
                'change_firing_rate': change
            })

    expanded_df = pd.DataFrame(expanded_data)

    if expanded_df.empty:
        print("No valid data after expanding for plotting change in firing rates.")
        return

    # Define the grouping
    group_order = ['HPC-structured', 'HPC-random', 'non-HPC-structured', 'non-HPC-random']
    existing_groups = expanded_df['group'].unique()
    group_order = [group for group in group_order if group in existing_groups]

    if not group_order:
        print("No valid groups available for plotting change in firing rates. Please check the data.")
        return

    # Set the aesthetic style of the plots
    sns.set(style="ticks")

    # Initialize the matplotlib figure with size (8,6)
    plt.figure(figsize=(8, 6))

    # Create a line plot with error bars using the updated 'errorbar' parameter
    ax = sns.pointplot(
        data=expanded_df,
        x='distance',
        y='change_firing_rate',
        hue='group',
        palette=custom_palette,
        errorbar="se",  # Updated parameter
        dodge=0.3,
        markers='o',
        capsize=0.1
    )
   
    # Add a horizontal line at y=0
    plt.axhline(0, linestyle='--', color='black', linewidth=1)

    # Set plot labels and title
    plt.xlabel('Absolute Scene Distance', fontsize=14)
    plt.ylabel('Change in Firing Rate (Hz)', fontsize=14)
    plt.title('Change in Firing Rate from Early to Last Trial vs Scene Distance', fontsize=16)

    # Adjust legend
    plt.legend(title='Group', fontsize=12, title_fontsize=12)

    # Adjust layout for better fit
    plt.tight_layout()

    # Save the plot
    save_path = os.path.join(output_dir, plot_filename)
    plt.savefig(save_path)
    plt.close()

    print(f"Saved change in firing rate vs distance plot at '{save_path}'.")

def visualize_firing_rate_distribution(master_df, distance_type='folded', output_dir=OUTPUT_DIR, plot_filename=DISTRIBUTION_PLOT_FILENAME):
    """
    Visualizes the distribution of firing rate distance-dependence as point clouds for each group.
    Adds a black horizontal line at y=0 and applies jitter to prevent overlapping.

    Parameters:
    - master_df (pd.DataFrame): The DataFrame with 'firing_rate_distance_dependence_folded' column.
    - distance_type (str): Type of distance analysis ('folded', 'positive', 'negative').
    - output_dir (str): Directory to save the distribution plot.
    - plot_filename (str): Filename for the saved distribution plot.

    Returns:
    - None
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define column names based on distance_type
    dependence_col = 'firing_rate_distance_dependence_folded'
    if distance_type == 'folded':
        dependence_col = 'firing_rate_distance_dependence_folded'
    elif distance_type == 'positive':
        dependence_col = 'firing_rate_distance_dependence_positive'
    elif distance_type == 'negative':
        dependence_col = 'firing_rate_distance_dependence_negative'
    else:
        raise ValueError("distance_type must be 'folded', 'positive', or 'negative'")

    # Drop rows with NaN firing_rate_distance_dependence
    plot_df = master_df.dropna(subset=[dependence_col])

    if plot_df.empty:
        print(f"No data available for plotting firing rate distribution ({distance_type}). Please check the DataFrame.")
        return

    # Define the grouping
    plot_df['group'] = plot_df['region_category'] + '-' + plot_df['task_type']

    # Define the order of groups for consistent plotting
    group_order = ['HPC-structured', 'HPC-random', 'non-HPC-structured', 'non-HPC-random']

    # Ensure that all groups are present in the data
    existing_groups = plot_df['group'].unique()
    group_order = [group for group in group_order if group in existing_groups]

    if not group_order:
        print(f"No valid groups available for plotting firing rate distribution ({distance_type}). Please check the DataFrame.")
        return

    # Set the aesthetic style of the plots
    sns.set(style="ticks")

    # Initialize the matplotlib figure with size (6, 8)
    plt.figure(figsize=(6, 8))
    plt.figure(figsize=(8, 10))

    # Create a strip plot with jitter
    ax = sns.stripplot(x='group', y=dependence_col, data=plot_df, order=group_order, palette=custom_palette,
                       jitter=0.5, dodge=True, alpha=0.6, size=10)

    # Add a black horizontal line at y=0
    plt.axhline(0, linestyle='--', color='black', linewidth=2)

    # Remove grid lines
    sns.despine()

    # Set plot labels and title
    plt.ylabel('Change in Firing Rate Distance Dependence (Hz)', fontsize=14)
    plt.title('Distribution of Firing Rate Distance Dependence (Folded)', fontsize=16)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Set y-axis limits to match the main plot
    plt.ylim([-1.5, 1.6])

    # Adjust layout for better fit
    plt.tight_layout()

    # Save the plot
    save_path = os.path.join(output_dir, plot_filename)
    plt.savefig(save_path)
    plt.close()

    print(f"Saved firing rate distribution plot at '{save_path}'.")

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
    required_columns = ['region_name', 'task_type', 'firing_rate_matrix']
    for col in required_columns:
        if col not in master_df.columns:
            print(f"Error: Required column '{col}' not found in the DataFrame.")
            return

    # Categorize regions
    master_df['region_category'] = master_df.apply(categorize_region, axis=1)
    print("Added 'region_category' column.")

    # Define scene distances
    scene_distances = np.arange(-4, 6)  # -4 to 5 inclusive

    # Fold the firing rate matrix
    print("Folding the firing rate matrices by averaging negative and positive scene distances...")
    folded_data = master_df['firing_rate_matrix'].apply(lambda fm: fold_firing_rate_matrix(fm, scene_distances))
    master_df['folded_scene_distances'] = folded_data.apply(lambda x: x[0] if isinstance(x, tuple) else np.nan)
    master_df['folded_firing_rates'] = folded_data.apply(lambda x: x[1] if isinstance(x, tuple) else np.nan)
    print("Completed folding of firing rate matrices.")

    # Assign folded firing rates to a single column as a tuple
    master_df['firing_rate_matrix_folded'] = master_df.apply(
        lambda row: row['folded_firing_rates'] if not isinstance(row['folded_firing_rates'], float) else np.nan,
        axis=1
    )

    # Recompute slopes based on folded data
    print("Recomputing slopes based on folded firing rate matrices...")
    master_df['slopes_folded'] = master_df.apply(
        lambda row: compute_slopes_folded(row, row['folded_scene_distances']) if not isinstance(row['firing_rate_matrix_folded'], float) else [np.nan],
        axis=1
    )
    print("Completed recomputing slopes.")

    # Compute firing rate distance-dependence for folded data
    master_df['firing_rate_distance_dependence_folded'] = master_df.apply(compute_firing_rate_dependence, axis=1)
    print("Added 'firing_rate_distance_dependence_folded' column.")

    # Compute change in firing rate from first two trials to last trial
    print("Computing change in firing rates from first two trials to last trial...")
    master_df['change_firing_rate_per_distance'] = master_df.apply(compute_change_firing_rate, axis=1)
    print("Added 'change_firing_rate_per_distance' column.")

    # Process Positive Scene Distances
    print("Processing positive scene distances...")
    # Define positive scene distances
    positive_mask = scene_distances > 0
    positive_scene_distances = scene_distances[positive_mask]

    # Extract positive firing rates
    master_df['positive_scene_distances'] = master_df['firing_rate_matrix'].apply(lambda fm: scene_distances[scene_distances > 0] if isinstance(fm, np.ndarray) else np.nan)
    master_df['positive_firing_rates'] = master_df['firing_rate_matrix'].apply(lambda fm: fm[:, scene_distances > 0] if isinstance(fm, np.ndarray) else np.nan)

    # Compute slopes for positive distances
    def compute_slopes_positive(row):
        firing_rates = row['positive_firing_rates']
        pos_distances = row['positive_scene_distances']
        
        if not isinstance(firing_rates, np.ndarray) or not isinstance(pos_distances, np.ndarray):
            return [np.nan]
        
        num_trials, num_distances = firing_rates.shape
        slopes = []
        
        for trial in range(num_trials):
            rates = firing_rates[trial, :]
            
            # Check for NaNs
            valid_mask = ~np.isnan(rates) & ~np.isnan(pos_distances)
            valid_distances = pos_distances[valid_mask]
            valid_rates = rates[valid_mask]
            
            if len(valid_distances) < 2:
                slopes.append(np.nan)
                continue
            
            # Perform linear regression
            slope, intercept, r_value, p_value, std_err = linregress(valid_distances, valid_rates)
            slopes.append(slope)
        
        return slopes

    master_df['slopes_positive'] = master_df.apply(
        lambda row: compute_slopes_positive(row) if not isinstance(row['positive_firing_rates'], float) else [np.nan],
        axis=1
    )

    # Compute firing rate distance-dependence for positive distances
    def compute_firing_rate_dependence_positive(row):
        """
        Computes the firing rate distance-dependence for positive scene distances.
        """
        slopes = row['slopes_positive']

        if not isinstance(slopes, list):
            return np.nan

        # Remove NaN slopes
        slopes_clean = [s for s in slopes if not np.isnan(s)]

        if len(slopes_clean) < 15:
            # Need at least 15 trials to have first 2 and last 1
            return np.nan

        late_avg = np.mean(slopes_clean[-N_TRIALS_PER_PHASE:])  # Averaging last trial
        early_avg = np.mean(slopes_clean[:(N_TRIALS_PER_PHASE+1)])  # Note: Averaging first 3 trials as per user code

        dependence = late_avg - early_avg
        return dependence

    master_df['firing_rate_distance_dependence_positive'] = master_df.apply(compute_firing_rate_dependence_positive, axis=1)
    print("Added 'firing_rate_distance_dependence_positive' column.")

    # Compute change in firing rate for positive distances
    def compute_change_firing_rate_positive(row):
        """
        Computes the change in firing rate from the average of the first two trials to the last trial
        for each positive scene distance.
        """
        firing_rates = row['positive_firing_rates']
        pos_scene_distances = row['positive_scene_distances']

        if not isinstance(firing_rates, np.ndarray) or not isinstance(pos_scene_distances, np.ndarray):
            return np.nan

        if firing_rates.shape[0] < 15:
            return np.nan

        # Compute average of the first two trials
        avg_first_two = np.nanmean(firing_rates[:(N_TRIALS_PER_PHASE+1), :], axis=0)  # Shape: (num_distances,)
        last_trial = firing_rates[(-N_TRIALS_PER_PHASE), :]  # Shape: (num_distances,)

        # Compute change in firing rate
        change_rates = last_trial - avg_first_two  # Shape: (num_distances,)

        return change_rates

    master_df['change_firing_rate_positive'] = master_df.apply(compute_change_firing_rate_positive, axis=1)
    print("Added 'change_firing_rate_positive' column.")

    # Process Negative Scene Distances
    print("Processing negative scene distances...")
    # Define negative scene distances
    negative_mask = scene_distances < 0
    negative_scene_distances = scene_distances[negative_mask]

    # Extract negative firing rates
    master_df['negative_scene_distances'] = master_df['firing_rate_matrix'].apply(lambda fm: scene_distances[scene_distances < 0] if isinstance(fm, np.ndarray) else np.nan)
    master_df['negative_firing_rates'] = master_df['firing_rate_matrix'].apply(lambda fm: fm[:, scene_distances < 0] if isinstance(fm, np.ndarray) else np.nan)

    # Compute slopes for negative distances
    def compute_slopes_negative(row):
        firing_rates = row['negative_firing_rates']
        neg_distances = row['negative_scene_distances']

        if not isinstance(firing_rates, np.ndarray) or not isinstance(neg_distances, np.ndarray):
            return [np.nan]

        num_trials, num_distances = firing_rates.shape
        slopes = []

        for trial in range(num_trials):
            rates = firing_rates[trial, :]

            # Check for NaNs
            valid_mask = ~np.isnan(rates) & ~np.isnan(neg_distances)
            valid_distances = neg_distances[valid_mask]
            valid_rates = rates[valid_mask]

            if len(valid_distances) < 2:
                slopes.append(np.nan)
                continue

            # Perform linear regression
            slope, intercept, r_value, p_value, std_err = linregress(-valid_distances, valid_rates)
            slopes.append(slope)

        return slopes

    master_df['slopes_negative'] = master_df.apply(
        lambda row: compute_slopes_negative(row) if not isinstance(row['negative_firing_rates'], float) else [np.nan],
        axis=1
    )

    # Compute firing rate distance-dependence for negative distances
    def compute_firing_rate_dependence_negative(row):
        """
        Computes the firing rate distance-dependence for negative scene distances.
        """
        slopes = row['slopes_negative']

        if not isinstance(slopes, list):
            return np.nan

        # Remove NaN slopes
        slopes_clean = [s for s in slopes if not np.isnan(s)]

        if len(slopes_clean) < 15:
            # Need at least 15 trials to have first 2 and last 1
            return np.nan

        late_avg = np.mean(slopes_clean[-N_TRIALS_PER_PHASE:])  # Averaging last trial
        early_avg = np.mean(slopes_clean[:(N_TRIALS_PER_PHASE+1)])  # Note: Averaging first 3 trials as per user code

        dependence = late_avg - early_avg
        return dependence

    master_df['firing_rate_distance_dependence_negative'] = master_df.apply(compute_firing_rate_dependence_negative, axis=1)
    print("Added 'firing_rate_distance_dependence_negative' column.")

    # Compute change in firing rate for negative distances
    def compute_change_firing_rate_negative(row):
        """
        Computes the change in firing rate from the average of the first two trials to the last trial
        for each negative scene distance.
        """
        firing_rates = row['negative_firing_rates']
        neg_scene_distances = row['negative_scene_distances']

        if not isinstance(firing_rates, np.ndarray) or not isinstance(neg_scene_distances, np.ndarray):
            return np.nan

        if firing_rates.shape[0] < 15:
            return np.nan

        # Compute average of the first two trials
        avg_first_two = np.nanmean(firing_rates[:(N_TRIALS_PER_PHASE+1), :], axis=0)  # Shape: (num_distances,)
        last_trial = firing_rates[-N_TRIALS_PER_PHASE, :]  # Shape: (num_distances,)

        # Compute change in firing rate
        change_rates = last_trial - avg_first_two  # Shape: (num_distances,)

        return change_rates

    master_df['change_firing_rate_negative'] = master_df.apply(compute_change_firing_rate_negative, axis=1)
    print("Added 'change_firing_rate_negative' column.")

    # Save the updated DataFrame before visualization
    try:
        master_df.to_pickle(OUTPUT_PKL)
        print(f"Saved the updated DataFrame with firing rate distance-dependence as '{OUTPUT_PKL}'.")
    except Exception as e:
        print(f"An error occurred while saving the DataFrame: {e}")
        return

    # Visualize the firing rate distance-dependence for folded data
    visualize_firing_rate_dependence(master_df, distance_type='folded', output_dir=OUTPUT_DIR, plot_filename=PLOT_FILENAME_FOLDED)
    print("Completed visualization of firing rate distance-dependence (folded).")

    # Visualize the firing rate distance-dependence for positive distances
    visualize_firing_rate_dependence(master_df, distance_type='positive', output_dir=OUTPUT_DIR, plot_filename=PLOT_FILENAME_POSITIVE)
    print("Completed visualization of firing rate distance-dependence (positive).")

    # Visualize the firing rate distance-dependence for negative distances
    visualize_firing_rate_dependence(master_df, distance_type='negative', output_dir=OUTPUT_DIR, plot_filename=PLOT_FILENAME_NEGATIVE)
    print("Completed visualization of firing rate distance-dependence (negative).")

    # Visualize the change in firing rates vs distance
    visualize_change_firing_rate(master_df, output_dir=OUTPUT_DIR, plot_filename=CHANGE_PLOT_FILENAME)
    print("Completed visualization of change in firing rates vs distance.")

    # Visualize the distribution of firing rate distance-dependence as a separate PDF
    visualize_firing_rate_distribution(master_df, distance_type='folded', output_dir=OUTPUT_DIR, plot_filename=DISTRIBUTION_PLOT_FILENAME)
    print("Completed visualization of firing rate distribution (folded).")

    # Optional: Remove intermediate folded data columns to save space
    master_df.drop(['folded_scene_distances', 'folded_firing_rates', 'firing_rate_matrix_folded',
                   'positive_scene_distances', 'positive_firing_rates',
                   'negative_scene_distances', 'negative_firing_rates'], axis=1, inplace=True)

    # Save the cleaned DataFrame without intermediate columns
    try:
        master_df.to_pickle(OUTPUT_PKL)
        print(f"Saved the cleaned DataFrame without intermediate folded data as '{OUTPUT_PKL}'.")
    except Exception as e:
        print(f"An error occurred while saving the cleaned DataFrame: {e}")
        return

if __name__ == "__main__":
    main()

