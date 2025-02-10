import pdb
import os
import re
import pickle
from collections import defaultdict
from utils import pdfTextSet
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wilcoxon, mannwhitneyu

SMOOTH_MATRIX_FIRST = True
# SMOOTH_MATRIX_FIRST = False

def clean_string(input_string):
    """
    Cleans the input string by removing or replacing characters 
    that are not suitable for filenames.
    
    Args:
        input_string (str): The string to clean.
    
    Returns:
        str: A cleaned version of the input string.
    """
    return re.sub(r'\W+', '_', input_string)

def load_aggregated_slope_changes(directory, region_types):
    """
    Loads aggregated overall_slope_changes data and session names from pickle files for specified region types.
    
    Args:
        directory (str): Path to the directory containing pickle files.
        region_types (list of str): List of region types to load (e.g., ['hpc', 'aic']).
    
    Returns:
        tuple: Two nested dictionaries:
               data[region_type][condition_type] = list of slope changes
               sess_names[region_type][condition_type] = list of session names
    """
    data = defaultdict(lambda: defaultdict(list))  # data[region_type][condition_type] = list
    sess_names = defaultdict(lambda: defaultdict(list))  # sess_names[region_type][condition_type] = list

    for region in region_types:
        cleaned_region = clean_string(region)
        filename = f"All_Patients_allChannelSpectralCorrelationMatrix_CorrVsDistSlopes_Sept9_24_{cleaned_region}_SMOOTH_MATRIX_FIRST_{SMOOTH_MATRIX_FIRST}.pkl"
        filepath = os.path.join(directory, filename)
        if not os.path.isfile(filepath):
            print(f"Warning: File {filepath} does not exist. Skipping.")
            continue
        try:
            with open(filepath, 'rb') as file:
                slope_changes, slope_changes_sessNames = pickle.load(file)  # [Modified to load both]
            for condition, values in slope_changes.items():
                data[region][condition].extend(values)
            for condition, sess in slope_changes_sessNames.items():
                sess_names[region][condition].extend(sess)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")

    return data, sess_names  # [Modified to return both]

def compute_statistics(data):
    """
    Computes mean and SEM for each condition type within each region type.

    Args:
        data (dict): Nested dictionary as returned by load_aggregated_slope_changes.

    Returns:
        dict: Nested dictionary with structure:
              stats[region_type][condition_type] = {'mean': value, 'sem': value}
    """
    stats_dict = defaultdict(dict)

    for region, conditions in data.items():
        for condition, values in conditions.items():
            if len(values) == 0:
                mean = np.nan
                sem = np.nan
                print(f"Warning: No data for region '{region}', condition '{condition}'.")
            else:
                mean = np.mean(values)
                sem = stats.sem(values)
            stats_dict[region][condition] = {'mean': mean, 'sem': sem}

    return stats_dict

def format_pval_pairwise(p):
    """
    Formats p-values for display on the plot.

    Args:
        p (float): p-value to format.

    Returns:
        str: Formatted p-value string.
    """
    if np.isnan(p):
        return 'p = NaN'
    elif p < 0.0001:
        return 'p < 0.0001'
    elif p < 0.001:
        return 'p < 0.001'
    elif p < 0.01:
        return f'**(p = {p:.4f})'
    elif p < 0.05:
        return f'*(p = {p:.3f})'
    else:
        return f'n.s. (p = {p:.3f})'

import re

def pair_sessions(structured_sessions, random_sessions):
    """
    and pairing the remaining sessions in order. Assumes that after removal, both lists

    Args:
        structured_sessions (list of str): List of structured session names.
        random_sessions (list of str): List of random session names.

    Returns:
        tuple: Two lists containing paired structured and random sessions.
    """
    # Remove unpaired prefixed sessions from structured_sessions
    filtered_structured = [s for s in structured_sessions if not s.startswith('pt3')]
    del random_sessions[9:11]
    # Check if the number of sessions matches after filtering
    if len(filtered_structured) != len(random_sessions):
        raise ValueError(
            f"Mismatch in session counts after filtering: "
            f"{len(filtered_structured)} structured vs {len(random_sessions)} random."
        )

    # Since the lists are aligned, pair them directly
    paired_structured = filtered_structured
    paired_random = random_sessions.copy()

    return paired_structured, paired_random


def plot_overall_slope_changes(stats_dict, data, sess_names, region_types, output_dir, plot_filename):
    """
    Plots bar graphs of mean slope changes with SEM, matching the style of visualize_firing_rate_dependence.

    Args:
        stats_dict (dict): Nested dictionary as returned by compute_statistics.
        data (dict): Nested dictionary with raw data.
        sess_names (dict): Nested dictionary with session names.
        region_types (list of str): List of region types.
        output_dir (str): Directory to save the plot.
        plot_filename (str): Filename for the saved plot (should have .pdf extension).
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Prepare data for plotting
    plot_data = []
    for region, conditions in stats_dict.items():
        for condition, stats_vals in conditions.items():
            plot_data.append({
                'Region': region,
                'Condition': condition,
                'Mean': stats_vals['mean'],
                'SEM': stats_vals['sem']
            })

    # Convert to DataFrame for easier plotting with seaborn
    df = pd.DataFrame(plot_data)

    # Define the order of groups for consistent plotting
    group_order = []
    for region in region_types:
        for condition in ['structured', 'random']:
            group_order.append(f"{region.capitalize()}-{condition}")

    # Create a mapping for grouping
    df['Group'] = df['Region'].str.capitalize() + '-' + df['Condition']

    # Ensure that all groups are present in the data
    existing_groups = df['Group'].unique()
    group_order = [group for group in group_order if group in existing_groups]

    # Define custom color palette: red for hpc, brown for aic
    # Within each region, light for random and dark for structured
    custom_palette = {
        'Hpc-structured': '#8B0000',   # Dark Red
        'Hpc-random': '#FF6347',       # Tomato (Light Red)
        'Aic-structured': '#8B4513',   # SaddleBrown (Dark Brown)
        'Aic-random': '#DEB887'        # BurlyWood (Light Brown)
    }

    # Initialize the matplotlib figure with size (6,8)
    plt.figure(figsize=(6, 8))

    # Set the aesthetic style of the plots
    sns.set(style="ticks")

    # Create a bar plot without error bars
    ax = sns.barplot(
        x='Group',
        y='Mean',
        data=df,
        order=group_order,
        palette=custom_palette,
        capsize=0.1,
        ci=None  # Disable built-in confidence intervals
    )

    # Manually add error bars
    for i, patch in enumerate(ax.patches):
        # Get the center x position of the bar
        bar_x = patch.get_x() + patch.get_width() / 2
        # Get the height of the bar (mean)
        bar_height = patch.get_height()
        # Get the SEM for this group
        sem_val = df['SEM'].iloc[i]
        # Plot the error bar
        ax.errorbar(
            bar_x,
            bar_height,
            yerr=sem_val,
            fmt='none',
            c='black',
            capsize=5,
            linewidth=2
        )

    # Add a dashed black line at y=0 with linewidth 4
    plt.axhline(0, linestyle='--', color='black', linewidth=2)

    # Remove grid lines
    sns.despine()

    # Perform one-sample Wilcoxon signed-rank tests for each group against 0
    p_values_one_sample = {}
    for index, row in df.iterrows():
        group = row['Group']
        condition = row['Condition']
        region = row['Region']
        # Extract raw data
        raw_values = data[region.lower()][condition]
        if len(raw_values) == 0:
            p_val = np.nan
        else:
            # Wilcoxon test requires at least 5 non-zero differences
            non_zero = np.count_nonzero(raw_values)
            if non_zero < 5:
                p_val = np.nan
            else:
                try:
                    #stat, p_val = wilcoxon(raw_values, zero_method='wilcox', alternative='two-sided')
                    stat, p_val = wilcoxon(raw_values, zero_method='wilcox', alternative='less')
                    print(f'{region} {condition}:  W={stat:.2f}, p = {p_val:.8f}')
                except ValueError:
                    p_val = np.nan  # If all values are zero or insufficient data
        p_values_one_sample[group] = p_val

    # Compute formatted p-values for each group
    formatted_pvals_one_sample = {group: format_pval_pairwise(p_val) for group, p_val in p_values_one_sample.items()}

    # Initialize a list to store pairwise p-values and their corresponding pairs
    pairwise_pvals = []

    # Perform paired tests based on session names for each region
    for region in region_types:
        region_cap = region.capitalize()
        structured_sessions = sess_names[region]['structured']
        random_sessions = sess_names[region]['random'][:]
        structured_data = data[region]['structured']
        random_data = data[region]['random']

        # Pair sessions based on the updated rules
        paired_structured_sess, paired_random_sess = pair_sessions(structured_sessions, random_sessions)

        # Extract the slope changes corresponding to the paired sessions
        paired_structured_values = []
        paired_random_values = []
        for s_sess, r_sess in zip(paired_structured_sess, paired_random_sess):
            # Retrieve the slope change values
            try:
                s_val = data[region]['structured'][structured_sessions.index(s_sess)]
                r_val = data[region]['random'][random_sessions.index(r_sess)]
                paired_structured_values.append(s_val)
                paired_random_values.append(r_val)
            except ValueError:
                # If session names are not found, skip
                continue

        # Perform paired Wilcoxon signed-rank test if there are paired samples
        if len(paired_structured_values) > 0:
            try:
                #stat, p_val = wilcoxon(paired_structured_values, paired_random_values, zero_method='wilcox', alternative='two-sided')
                stat, p_val = wilcoxon(paired_structured_values, paired_random_values, zero_method='wilcox', alternative='less')
                print(f'{region} {condition}:  W={stat:.2f}, p = {p_val:.8f}')
            except ValueError:
                p_val = np.nan  # If data is insufficient
        else:
            p_val = np.nan

        #if np.isnan(p_val):
        #    pdb.set_trace()
        # Append the result
        pairwise_pvals.append((f"{region_cap}-structured", f"{region_cap}-random", p_val))

    # Annotate the p-values on the plot
    y_max = df['Mean'].max() + df['SEM'].max()
    y_min = df['Mean'].min() - df['SEM'].min()
    y_offset = 0.05 * (y_max - y_min)  # 5% of the data range
    MAX_Y = 0.6  # As per user request

    # Add p-value annotations above each bar for one-sample tests
    for i, group in enumerate(group_order):
        mean = df.loc[df['Group'] == group, 'Mean'].values[0]
        sem = df.loc[df['Group'] == group, 'SEM'].values[0]
        y = mean + sem + y_offset
        y = min(y, MAX_Y * 0.9)  # Ensure y does not exceed MAX_Y
        ax.text(i, y, formatted_pvals_one_sample[group], ha='center', va='bottom', fontsize=10, color='black')

    # Annotate pairwise p-values
    for pair in pairwise_pvals:
        group1, group2, p_val = pair
        formatted_pval = format_pval_pairwise(p_val)
        try:
            x1 = group_order.index(group1)
            x2 = group_order.index(group2)
        except ValueError:
            continue  # If groups are not found, skip

        # Calculate the y position for the annotation
        height1 = df.loc[df['Group'] == group1, 'Mean'].values[0] + df.loc[df['Group'] == group1, 'SEM'].values[0]
        height2 = df.loc[df['Group'] == group2, 'Mean'].values[0] + df.loc[df['Group'] == group2, 'SEM'].values[0]
        y = max(height1, height2) + y_offset + 0.05  # Slightly above the higher bar
        y = min(y, MAX_Y * 0.9)  # Ensure y does not exceed MAX_Y

        # Draw the line
        ax.plot([x1, x1, x2, x2], [y, y + 0.02, y + 0.02, y], lw=1.5, c='black')

        # Add the formatted p-value text
        ax.text((x1 + x2) * 0.5, y + 0.02, formatted_pval, ha='center', va='bottom', color='black', fontsize=10)

    # Set plot labels and title
    plt.ylabel('Mean Slope Change', fontsize=14)
    plt.title('Overall Slope Changes by Region and Condition', fontsize=16)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Adjust layout for better fit
    plt.tight_layout()

    # Save the plot as PDF
    save_path = os.path.join(output_dir, plot_filename)
    plt.savefig(save_path, format='pdf')
    plt.close()

    print(f"Saved overall slope changes plot at '{save_path}'.")

def plot_individual_data_points(data, sess_names, output_dir, plot_filename):
    """
    Plots individual data points corresponding to each bar and draws lines between paired points.

    Args:
        data (dict): Nested dictionary with raw data.
        sess_names (dict): Nested dictionary with session names.
        output_dir (str): Directory to save the plot.
        plot_filename (str): Filename for the saved plot (should have .pdf extension).
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Prepare data for plotting
    plot_data = []
    for region, conditions in data.items():
        for condition, values in conditions.items():
            for value in values:
                plot_data.append({
                    'Region': region.capitalize(),
                    'Condition': condition,
                    'Value': value
                })

    # Convert to DataFrame
    df = pd.DataFrame(plot_data)

    # Create 'Group' column
    df['Group'] = df['Region'] + '-' + df['Condition']

    # Define the order of groups for consistent plotting
    group_order = ['Hpc-structured', 'Hpc-random', 'Aic-structured', 'Aic-random']

    # Define custom color palette: red for hpc, brown for aic
    # Within each region, light for random and dark for structured
    custom_palette = {
        'Hpc-structured': '#8B0000',   # Dark Red
        'Hpc-random': '#FF6347',       # Tomato (Light Red)
        'Aic-structured': '#8B4513',   # SaddleBrown (Dark Brown)
        'Aic-random': '#DEB887'        # BurlyWood (Light Brown)
    }

    # Initialize the matplotlib figure with size (6,8)
    plt.figure(figsize=(6, 8))

    # Set the aesthetic style of the plots
    sns.set(style="ticks")

    # Create a boxplot for better visualization of data distribution
    ax = sns.boxplot(
        x='Group',
        y='Value',
        data=df,
        order=group_order,
        palette=custom_palette,
        showcaps=True,
        boxprops={'facecolor':'None'},
        showfliers=False,
        whiskerprops={'linewidth':0},
        saturation=1
    )

    # Overlay individual data points with colors matching the bars
    sns.stripplot(
        x='Group',
        y='Value',
        data=df,
        order=group_order,
        hue='Group',
        palette=custom_palette,
        dodge=False,
        alpha=0.7,
        jitter=False,
        size=5,
        edgecolor='none'
    )

    # Remove the duplicate legend from stripplot
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend_.remove()

    # Add a dashed black line at y=0 with linewidth 4
    plt.axhline(0, linestyle='--', color='black', linewidth=2)

    # Remove grid lines
    sns.despine()

    # Initialize a list to store pairwise p-values and their corresponding pairs
    pairwise_pvals = []

    # Perform paired tests based on session names for each region
    for region in ['hpc', 'aic']:
        region_cap = region.capitalize()
        structured_sessions = sess_names[region]['structured']
        random_sessions = sess_names[region]['random'][:]
        structured_data = data[region]['structured']
        random_data = data[region]['random']

        # Pair sessions based on the updated rules
        paired_structured_sess, paired_random_sess = pair_sessions(structured_sessions, random_sessions)

        # Extract the slope changes corresponding to the paired sessions
        paired_structured_values = []
        paired_random_values = []
        for s_sess, r_sess in zip(paired_structured_sess, paired_random_sess):
            # Retrieve the slope change values
            try:
                s_val = data[region]['structured'][structured_sessions.index(s_sess)]
                r_val = data[region]['random'][random_sessions.index(r_sess)]
                paired_structured_values.append(s_val)
                paired_random_values.append(r_val)
            except ValueError:
                # If session names are not found, skip
                continue

        # Perform paired Wilcoxon signed-rank test if there are paired samples
        if len(paired_structured_values) > 0:
            try:
                #stat, p_val = wilcoxon(paired_structured_values, paired_random_values, zero_method='wilcox', alternative='two-sided')
                stat, p_val = wilcoxon(paired_structured_values, paired_random_values, zero_method='wilcox', alternative='less')
                print(f'{region} {condition}:  W={stat:.2f}, p = {p_val:.8f}')
            except ValueError:
                p_val = np.nan  # If data is insufficient
        else:
            p_val = np.nan

        # Append the result
        pairwise_pvals.append((f"{region_cap}-structured", f"{region_cap}-random", p_val))

    # Annotate pairwise p-values
    y_max = df['Value'].max()
    y_min = df['Value'].min()
    y_offset = 0.05 * (y_max - y_min)  # 5% of the data range
    MAX_Y = 0.6  # As per user request

    # Compute formatted p-values for each pair
    formatted_pairwise_pvals = {f"{pair[0]}-{pair[1]}": format_pval_pairwise(pair[2]) for pair in pairwise_pvals}

    # Annotate pairwise p-values on the plot
    for pair in pairwise_pvals:
        group1, group2, p_val = pair
        formatted_pval = format_pval_pairwise(p_val)
        try:
            x1 = group_order.index(group1)
            x2 = group_order.index(group2)
        except ValueError:
            continue  # If groups are not found, skip

        # Calculate the y position for the annotation
        height1 = df[df['Group'] == group1]['Value'].max()
        height2 = df[df['Group'] == group2]['Value'].max()
        y = max(height1, height2) + y_offset + 0.05  # Slightly above the higher bar
        y = min(y, MAX_Y * 0.9)  # Ensure y does not exceed MAX_Y

        # Draw the line
        ax.plot([x1, x1, x2, x2], [y, y + 0.02, y + 0.02, y], lw=1.5, c='black')

        # Add the formatted p-value text
        ax.text((x1 + x2) * 0.5, y + 0.02, formatted_pval, ha='center', va='bottom', color='black', fontsize=10)

    # Draw lines between paired individual points
    for region in ['hpc', 'aic']:
        region_cap = region.capitalize()
        structured_sessions = sess_names[region]['structured']
        random_sessions = sess_names[region]['random'][:]
        paired_structured_sess, paired_random_sess = pair_sessions(structured_sessions, random_sessions)
        for s_sess, r_sess in zip(paired_structured_sess, paired_random_sess):
            try:
                # Get the y-values
                s_idx = structured_sessions.index(s_sess)
                r_idx = random_sessions.index(r_sess)
                s_val = data[region]['structured'][s_idx]
                r_val = data[region]['random'][r_idx]

                # Get the x positions
                group1 = f"{region_cap}-structured"
                group2 = f"{region_cap}-random"
                x1 = group_order.index(group1)
                x2 = group_order.index(group2)

                # Since stripplot adds jitter, approximate x positions by adding random jitter around the group center
                jitter_strength = 0  # Adjust as needed
                np.random.seed(s_idx + r_idx)  # For reproducibility per session
                x1_jitter = x1 + np.random.uniform(-jitter_strength, jitter_strength)
                x2_jitter = x2 + np.random.uniform(-jitter_strength, jitter_strength)

                # Determine the color based on the direction of change
                if s_val > r_val:
                    line_color = 'red'
                elif s_val < r_val:
                    line_color = 'blue'
                else:
                    line_color = 'gray'  # Neutral color if no change

                # Draw the line
                plt.plot([x1_jitter, x2_jitter], [s_val, r_val], c=line_color, alpha=0.7, linewidth=0.7)
            except ValueError:
                # If session names are not found, skip
                continue

    # Set plot labels and title
    plt.ylabel('Slope Change', fontsize=14)
    plt.title('Individual Slope Changes by Region and Condition', fontsize=16)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Adjust layout for better fit
    plt.tight_layout()

    # Save the plot as PDF
    save_path = os.path.join(output_dir, plot_filename)
    plt.savefig(save_path, format='pdf')
    plt.close()

    print(f"Saved individual data points plot at '{save_path}'.")

def main():
    """
    Example usage of the above functions.
    """
    # Parameters
    directory = '.'
    region_types = ['hpc', 'aic']       # The two region types
    output_dir = 'slope_change_analysis_LFP_Jan10_2025'  # Output directory as per user request
    plot_filename_bar = 'overall_slope_changes_plot.pdf'     # Save as PDF
    plot_filename_individual = 'individual_slope_changes_plot.pdf'  # Save as PDF

    # Load data and session names [Modified to load both]
    data, sess_names = load_aggregated_slope_changes(directory, region_types)

    # Compute statistics
    stats_dict = compute_statistics(data)

    # Plot bar graph with p-values
    plot_overall_slope_changes(
        stats_dict=stats_dict,
        data=data,
        sess_names=sess_names,  # [Modified to pass session names]
        region_types=region_types,
        output_dir=output_dir,
        plot_filename=plot_filename_bar
    )

    # Plot individual data points
    plot_individual_data_points(
        data=data,
        sess_names=sess_names,  # [Modified to pass session names]
        output_dir=output_dir,
        plot_filename=plot_filename_individual
    )

if __name__ == "__main__":
    main()

