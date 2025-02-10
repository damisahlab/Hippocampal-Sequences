import pdb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import sem, linregress, ttest_1samp
from matplotlib.colors import Normalize, ListedColormap

import matplotlib as mpl
# Set font type to 42 (TrueType)
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['pdf.fonttype'] = 42

N_TRIALS_PER_PHASE=1
#N_TRIALS_PER_PHASE=3
#N_TRIALS_PER_PHASE=2

# Configuration Constants
USE_RAW_RATE = False
# USE_RAW_RATE = True
PLOT_INDIVIDUAL_POINTS=False
USE_SMOOTH = True #for all trial heatmap
TRIAL_SMOOTH_WIND = 3
TRIAL_SMOOTH_WIND = 2
# sns.set(style="whitegrid")

# Create the plasma colormap
plasma_cmap = plt.cm.plasma
cutoff = 0.85  # 85% cutoff
#truncated_cmap = ListedColormap(plasma_cmap(np.linspace(0, cutoff, 256)))
if USE_RAW_RATE:
    BASELINE_SCENE_VMIN = -1  # Corrected duplicate assignment
    BASELINE_SCENE_VMAX = 2  # Hz
else:
    cutoff = 0.8
    BASELINE_SCENE_VMIN = -1  # Corrected duplicate assignment
    BASELINE_SCENE_VMAX = 2  # Hz

# Create the truncated colormap
truncated_cmap = ListedColormap(plasma_cmap(np.linspace(0, cutoff, 256)))

BASELINE_SCENE_RATE_CMAP = 'coolwarm'
BASELINE_SCENE_RATE_CMAP = 'plasma'
BASELINE_SCENE_RATE_CMAP = truncated_cmap
# BASELINE_SCENE_VMIN = None  # Corrected duplicate assignment
# BASELINE_SCENE_VMAX = None  # Hz

RESPONSE_VMIN = -0.75
RESPONSE_VMAX = 0.75

ONLY_SUMMARY = False  # Set to False to include individual unit plots
ONLY_SUMMARY = True   # Set to False to include individual unit plots
N_TRIALS_EARLY = 3   
N_TRIALS_EARLY = 1   

REGION_COLOR = {
    'HPC': {
        'structured': 'red',            # HPC-structured is red
        'random': 'lightcoral'          # HPC-random is lightcoral
    },
    'non-HPC': {
        'structured': 'brown',          # non-HPC-structured is brown
        'random': '#D2B48C'             # non-HPC-random is light brown (hex code)
    }
}


def moving_average_smooth(data, window_size):
    """
    Applies a moving average filter to smooth the data across trials.

    Parameters:
    - data (np.ndarray): 2D array with shape (scenes, trials).
    - window_size (int): Size of the moving window.

    Returns:
    - smoothed_data (np.ndarray): Smoothed data array.
    """
    smoothed_data = np.copy(data)
    half_window = window_size // 2
    for i in range(data.shape[1]):
        start = max(0, i - half_window)
        end = min(data.shape[1], i + half_window + 1)
        smoothed_data[:, i] = np.mean(data[:, start:end], axis=1)
    return smoothed_data


def plot_individual_unit_slopes(ax, slopes, cmap='coolwarm', color='black', center=0):
    """
    Plots individual unit slopes as points colored by their slope values.
    Additionally, overlays the mean slope with SEM as a line and shaded region on a secondary y-axis.
    Adds circular continuity by duplicating scene distances and adding shaded regions.

    Parameters:
    - ax (matplotlib.axes.Axes): The primary axes to plot on.
    - slopes (dict): Dictionary with scene distances as keys and lists of slopes as values.
    - cmap (str or Colormap): Colormap to use for coloring the points.
    - color (str): Color for the mean and SEM overlays.
    - center (float): The center value for the colormap normalization.

    Returns:
    - None
    """
    scene_distances = sorted(slopes.keys())

    # Define how many points to duplicate for circular continuity
    duplication_count = 2  # Number of points to duplicate on each side
    duplication_count = 1  # Number of points to duplicate on each side

    # Duplicate scene distances and slopes
    extended_scene_distances = scene_distances.copy()
    extended_slopes = {dist: slopes[dist].copy() for dist in scene_distances}

    # Duplicate the first 'duplication_count' scene distances at the end
    for i in range(duplication_count):
        original_dist = scene_distances[i]
        new_dist = original_dist + (scene_distances[-1] - scene_distances[0] + 1)
        extended_scene_distances.append(new_dist)
        extended_slopes[new_dist] = slopes[original_dist].copy()

    # Duplicate the last 'duplication_count' scene distances at the beginning
    for i in range(duplication_count):
        original_dist = scene_distances[-(i+1)]
        new_dist = original_dist - (scene_distances[-1] - scene_distances[0] + 1)
        extended_scene_distances.insert(0, new_dist)
        extended_slopes[new_dist] = slopes[original_dist].copy()

    # Update scene distances for plotting
    scene_distances = extended_scene_distances
    slopes = extended_slopes

    # Flatten all slope values to find min and max for normalization
    all_slopes = [slope for sublist in slopes.values() for slope in sublist if not np.isnan(slope)]
    if not all_slopes:
        print("No slope data available to plot individual unit slopes.")
        return

    
    # Calculate mean and SEM for each scene distance
    mean_slopes = []
    sem_slopes = []
    for dist in scene_distances:
        valid_slopes = [s for s in slopes[dist] if not np.isnan(s)]
        if len(valid_slopes) > 0:
            mean = np.mean(valid_slopes)
            sem_val = sem(valid_slopes)
        else:
            mean = np.nan
            sem_val = np.nan
        mean_slopes.append(mean)
        sem_slopes.append(sem_val)

    # Convert to numpy arrays for plotting
    mean_slopes = np.array(mean_slopes)
    sem_slopes = np.array(sem_slopes)


    ax.axhline(0,color='k',linestyle='--')
    ax.plot(scene_distances, mean_slopes, color=color, linewidth=3, marker='o',label='Mean Slope')

    ax.errorbar(scene_distances, mean_slopes,sem_slopes,linewidth=3,capsize=4, capthick=5,
            color=color, alpha=0.8, label='SEM')
    ax.set_ylim([-0.18,0.18])
    ax.set_ylim([-2.1,2.1])
    ax.set_ylim([-2.5,2.5])
    #ax.set_xlim([-5.5,5.5])



def plot_mean_slope_vs_trial(ax, slopes_per_trial_pos, slopes_per_trial_neg,
                             label='Mean (Pos - Neg) Slope', line_color='purple'):
    """
    Plots the mean of positive scene slopes with the negative of the negative scene slopes
    along with the Standard Error of the Mean (SEM) across units for each trial.

    Parameters:
    - ax (matplotlib.axes.Axes): The axes to plot on.
    - slopes_per_trial_pos (dict): Dictionary with trial numbers as keys and lists of slopes 
                                    (positive scene distances) as values.
    - slopes_per_trial_neg (dict): Dictionary with trial numbers as keys and lists of slopes 
                                    (negative scene distances) as values.
    - label (str): Label for the combined slope plot.
    - line_color (str): Color for the combined slope plot.

    Returns:
    - None
    """
    # Ensure both dictionaries have the same set of trials
    trials_pos = set(slopes_per_trial_pos.keys())
    trials_neg = set(slopes_per_trial_neg.keys())
    common_trials = sorted(trials_pos.intersection(trials_neg))  # Corrected variable name

    if not common_trials:
        print("Error: No common trials found between positive and negative slopes.")
        return

    mean_combined_slopes = []
    sem_combined_slopes = []
    valid_trials = []

    for trial in common_trials:
        slopes_pos = slopes_per_trial_pos.get(trial, [])
        slopes_neg = slopes_per_trial_neg.get(trial, [])

        # Check if both lists have the same number of slopes
        if len(slopes_pos) != len(slopes_neg):
            print(f"Warning: Trial {trial} has {len(slopes_pos)} positive slopes and {len(slopes_neg)} negative slopes.")
            # Proceed with the minimum length to avoid index errors
            min_length = min(len(slopes_pos), len(slopes_neg))
            slopes_pos = slopes_pos[:min_length]
            slopes_neg = slopes_neg[:min_length]

        # Combine slopes: positive slopes - negative slopes
        combined_slopes = []
        for p, n in zip(slopes_pos, slopes_neg):
            if np.isnan(p) or np.isnan(n):
                combined_slopes.append(np.nan)
            else:
                combined_slopes.append(p - n)

        # Remove NaN values for accurate mean and SEM calculations
        combined_slopes_clean = [s for s in combined_slopes if not np.isnan(s)]

        if combined_slopes_clean:
            mean = np.mean(combined_slopes_clean)
            sem_val = sem(combined_slopes_clean)
            mean_combined_slopes.append(mean)
            sem_combined_slopes.append(sem_val)
            valid_trials.append(trial)
        else:
            mean_combined_slopes.append(np.nan)
            sem_combined_slopes.append(np.nan)
            valid_trials.append(trial)
            print(f"Warning: All combined slopes are NaN for trial {trial}.")

    # Convert lists to numpy arrays for masking NaNs
    mean_combined_slopes = np.array(mean_combined_slopes)
    sem_combined_slopes = np.array(sem_combined_slopes)
    valid_trials = np.array(valid_trials)

    # Mask trials where mean_combined_slopes is NaN
    valid_mask = ~np.isnan(mean_combined_slopes)
    valid_trials = valid_trials[valid_mask]
    mean_combined_slopes = mean_combined_slopes[valid_mask]
    sem_combined_slopes = sem_combined_slopes[valid_mask]

    if len(valid_trials) == 0:
        print("Error: No valid trials with non-NaN combined slopes.")
        return

    # Plot the combined mean slopes with SEM
    ax.errorbar(valid_trials, mean_combined_slopes, yerr=sem_combined_slopes, fmt='o-',
                color=line_color, ecolor='lightcoral' if line_color == 'red' else 'sandybrown',
                capsize=5, label=label)

    # Add a horizontal line at y=0 for reference
    ax.axhline(0, linewidth=1, color='grey', linestyle='--')

    # Set plot labels and title
    ax.set_xlabel("Trial Number")
    ax.set_ylabel("Mean Combined Slope (Pos - Neg)")
    ax.set_title("Mean Combined Slope ± SEM Across Units by Trial")
    # ax.legend()
    # ax.set_ylim([-0.1, 0.18])
    ax.set_ylim([-1, 0.5])


def calculate_slope(data):
    """
    Calculates the slope and correlation coefficient using linear regression.

    Parameters:
    - data (np.ndarray): 1D array of data points.

    Returns:
    - slope (float): Slope of the regression line.
    - r_value (float): Correlation coefficient.
    """
    trials = np.arange(data.shape[0])
    slope, _, r_value, _, _ = linregress(trials, data)
   
    #TEST
    slope=np.mean(data[-N_TRIALS_PER_PHASE:])-np.mean(data[:(N_TRIALS_PER_PHASE+1)])

    return slope, r_value

def SMOOTHcalculate_slope(data, window_size=3):
    """
    Smooths the input data using a running average and then calculates the slope and correlation coefficient using linear regression.

    Parameters:
    - data (np.ndarray): 1D array of data points.
    - window_size (int): Size of the moving window for the running average. Must be a positive integer.

    Returns:
    - slope (float): Slope of the regression line on the smoothed data.
    - r_value (float): Correlation coefficient of the regression on the smoothed data.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a NumPy array.")
    if data.ndim != 1:
        raise ValueError("Input data must be a 1D array.")
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("window_size must be a positive integer.")

    # Create the running average window
    window = np.ones(window_size) / window_size

    # Apply convolution to smooth the data
    # 'same' ensures the output array has the same length as the input
    smoothed_data = np.convolve(data, window, mode='same')

    # Prepare the trial indices
    trials = np.arange(smoothed_data.shape[0])

    # Perform linear regression on the smoothed data
    slope, intercept, r_value, p_value, std_err = linregress(trials, smoothed_data)

    return slope, r_value


#TEST JAN 8 2025
#slope=data[-1]-data[0]
#slope=data[-1]-np.mean(data[:2])

def shift_colormap(cmap, start=0.0):
    """
    Shifts a colormap to start at a different point.

    Parameters:
    - cmap (Colormap): The original colormap.
    - start (float): The fraction of the colormap length to shift by (between 0.0 and 1.0).

    Returns:
    - new_cmap (ListedColormap): The shifted colormap.
    """
    if not 0.0 <= start <= 1.0:
        raise ValueError("start must be between 0.0 and 1.0")

    # Get the list of colors from the original colormap
    n_colors = cmap.N
    colors = cmap(np.linspace(0, 1, n_colors))

    # Shift the colors by the specified fraction
    shift_idx = int(start * n_colors)
    shifted_colors = np.roll(colors, -shift_idx, axis=0)

    # Create a new colormap from the shifted colors
    new_cmap = ListedColormap(shifted_colors)
    return new_cmap


def plot_response_line(ax, data, indices, cmap, vmin, vmax):
    """
    Plots the scene response curve across trials.

    Parameters:
    - ax (matplotlib.axes.Axes): The axes to plot on.
    - data (np.ndarray): 2D array with shape (scenes, trials).
    - indices (list): List of scene indices to plot.
    - cmap (Colormap): Colormap for different scene distances.
    - vmin (float): Minimum value for normalization.
    - vmax (float): Maximum value for normalization.

    Returns:
    - None
    """
    trials = np.arange(data.shape[1])
    norm = Normalize(vmin=vmin, vmax=vmax)
    for idx in indices:
        if idx < data.shape[0]:
            values = data[idx, :]
            scene_distance = idx - 4  # Map index to scene distance (-4 to 5)
            color = cmap(norm(scene_distance))
            ax.plot(trials, values - np.mean(values[:N_TRIALS_EARLY]), color=color)
    # Create ScalarMappable for colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Scene Distance from Center')
    ax.set_title("Scene Response Curve Across Trials")
    ax.set_ylim([RESPONSE_VMIN, RESPONSE_VMAX])
    ax.set_xlabel("Trial Number")
    ax.set_ylabel("Response Value")
    ax.set_xticks(np.arange(len(trials)))
    ax.set_xticklabels(np.arange(1, len(trials) + 1))  # Added 1 to trial numbers


def plot_violin_bar(delta_slopes_df):
    """
    Plots a violin bar plot for the four conditions:
    HPC-structured, HPC-random, non-HPC-structured, non-HPC-random.
    Each point represents the delta slope for a unit.
    The plot follows the specified color scheme and includes SEM on the bar plots.

    Parameters:
    - delta_slopes_df (pd.DataFrame): DataFrame containing 'Region', 'Task', 'Delta_Slope' columns.

    Returns:
    - None
    """
    # Create a 'Condition' column combining 'Region' and 'Task'
    delta_slopes_df['Condition'] = delta_slopes_df['Region'] + '-' + delta_slopes_df['Task']

    # Define the order of conditions
    condition_order = ['HPC-structured', 'HPC-random', 'non-HPC-structured', 'non-HPC-random']

    # Define colors with transparency for random conditions
    color_mapping = {
        'HPC-structured': (1.0, 0, 0, 1.0),           # Red, opaque
        'HPC-random': (1.0, 0, 0, 0.6),              # Red, transparent
        'non-HPC-structured': (165/255, 42/255, 42/255, 1.0),  # Brown, opaque
        'non-HPC-random': (165/255, 42/255, 42/255, 0.6)      # Brown, transparent
    }
    palette = [color_mapping[cond] for cond in condition_order]

    plt.figure(figsize=(4, 8))
    sns.violinplot(x='Condition', y='Delta_Slope', data=delta_slopes_df, order=condition_order,
                   palette=palette, inner=None, linewidth=1.5, bw=0.2)  # Smaller kernel with bw=0.2

    # Overlay individual data points
    sns.stripplot(x='Condition', y='Delta_Slope', data=delta_slopes_df, order=condition_order,
                  color='black', alpha=0.6, jitter=True, size=4)

    # Calculate means and SEMs
    summary = delta_slopes_df.groupby('Condition')['Delta_Slope'].agg(['mean', 'sem']).reindex(condition_order)

    # Plot bar plots with error bars
    plt.errorbar(x=np.arange(len(condition_order)),
                 y=summary['mean'],
                 yerr=summary['sem'],
                 fmt='none',
                 ecolor='black',
                 capsize=5,
                 linewidth=2,
                 label='Mean ± SEM')

    # Plot mean points
    plt.scatter(x=np.arange(len(condition_order)),
                y=summary['mean'],
                color='black',
                zorder=10)

    plt.axhline(0, linewidth=3, linestyle='--', color='black')
    # Perform single-sample t-tests and annotate p-values
    for idx, condition in enumerate(condition_order):
        data = delta_slopes_df[delta_slopes_df['Condition'] == condition]['Delta_Slope'].dropna()
        if len(data) < 2:
            p_value = np.nan
            print(f"Warning: Not enough data for t-test in condition '{condition}'.")
        else:
            t_stat, p_value = ttest_1samp(data, 0.0)
        # Determine significance asterisks
        if np.isnan(p_value):
            sig = 'ns'
        elif p_value <= 0.001:
            sig = '***'
        elif p_value < 0.01:
            sig = '**'
        elif p_value < 0.05:
            sig = '*'
        else:
            sig = 'ns'

        # Annotate p-value
        if not np.isnan(p_value):
            plt.text(idx, summary['mean'][condition] + summary['sem'][condition] + 0.05, f'{sig}, p={p_value:.8f}',
                     ha='center', va='bottom', fontsize=14, color='black')
        else:
            plt.text(idx, summary['mean'][condition] + 0.05, sig,
                     ha='center', va='bottom', fontsize=14, color='black')

    plt.xlabel("Condition", fontsize=14)
    plt.ylabel("Delta Slope (Last 3 Trials - First 3 Trials)", fontsize=14)
    plt.title("Delta Slope Across Conditions", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.legend(['Mean ± SEM'], loc='upper right')

    plt.tight_layout()
    fig_filename = "violin_bar_plot_delta_slope.pdf"
    plt.savefig(fig_filename,format='pdf')
    plt.close()
    print(f"Saved {fig_filename}")


def rename_and_plot(master_df, target_task_type, target_regions, smoothing_window_size=3, offset=0.0, delta_slopes_list=None):
    """
    Renames columns, processes data, calculates slopes and correlations,
    generates plots for the specified task type and regions, and collects delta slopes.

    Parameters:
    - master_df (pd.DataFrame): DataFrame containing unit data.
    - target_task_type (str): The task type to filter (e.g., "structured", "random").
    - target_regions (list): List of regions to include (e.g., ["HPC"]).
    - smoothing_window_size (int): Window size for moving average smoothing.
    - offset (float): Offset for colormap shifting.
    - delta_slopes_list (list): List to append delta slope data dictionaries.

    Returns:
    - master_df (pd.DataFrame): The renamed DataFrame.
    """
    # Rename relevant columns for clarity
    columns_to_rename = {
        "baseline_rate_array": "baseline_rate_array_per_centeredScene_per_trial",
        "scene_rate_array": "scene_rate_array_per_centeredScene_per_trial",
        "response_array": "response_array_per_centeredScene_per_trial"
    }
    master_df = master_df.rename(columns=columns_to_rename)

    # Containers for accumulating data
    baseline_accum = []
    scene_accum = []
    response_accum = []

    # Containers for slopes and correlations across units
    slopes_per_scene = {i: [] for i in range(-4, 6)}
    correlations_per_scene = {i: [] for i in range(-4, 6)}

    # Containers for slopes per trial (positive and negative scene distances)
    slopes_per_trial_pos = {}
    slopes_per_trial_neg = {}

    # Define the colormap and apply the offset
    base_colormap = plt.colormaps['twilight_shifted']
    cmap = shift_colormap(base_colormap, start=offset)

    # Define positive and negative scene indices
    scene_distances = np.arange(-4, 6)  # Scene distances from -4 to +5
    positive_scene_indices = [i for i, sd in enumerate(scene_distances) if sd > 0]
    negative_scene_indices = [i for i, sd in enumerate(scene_distances) if sd < 0]

    # Determine line color based on target_regions
    if "HPC" in target_regions:
        line_color = REGION_COLOR['HPC'][target_task_type]
    else:
        line_color = REGION_COLOR['non-HPC'][target_task_type]

    # Iterate over each unit in the DataFrame
    for idx, row in master_df.iterrows():
        baseline_data = row["baseline_rate_array_per_centeredScene_per_trial"]
        scene_data = row["scene_rate_array_per_centeredScene_per_trial"]
        response_data = row["response_array_per_centeredScene_per_trial"]

        # Apply smoothing if enabled
        if USE_SMOOTH:
            baseline_data_smooth = moving_average_smooth(baseline_data, smoothing_window_size)
            scene_data_smooth = moving_average_smooth(scene_data, smoothing_window_size)
            response_data_smooth = moving_average_smooth(response_data, smoothing_window_size)
        else:
            baseline_data_smooth = baseline_data
            scene_data_smooth = scene_data
            response_data_smooth = response_data

        # Subtract the average of the first N_TRIALS_EARLY trials from all trials for common starting point
        # This is done per scene
        baseline_avg_first = np.nanmean(baseline_data_smooth[:, :N_TRIALS_EARLY], axis=1, keepdims=True)
        scene_avg_first = np.nanmean(scene_data_smooth[:, :N_TRIALS_EARLY], axis=1, keepdims=True)
        response_avg_first = np.nanmean(response_data_smooth[:, :N_TRIALS_EARLY], axis=1, keepdims=True)

        baseline_data_normalized = baseline_data_smooth - baseline_avg_first
        scene_data_normalized = scene_data_smooth - scene_avg_first
        response_data_normalized = response_data_smooth - response_avg_first

        # if not USE_RAW_RATE:
        trial_avg_plot_data = response_data_normalized
        # TEST OCT 12 24

        # Set all values in the scene index 4 (0 distance from center) to NaN
        response_data_normalized[4, :] = np.nan
        trial_avg_plot_data[4, :] = np.nan
        baseline_data_normalized[4, :] = np.nan

        # Initialize slopes_per_trial dictionaries if empty
        if not slopes_per_trial_pos:
            num_trials = trial_avg_plot_data.shape[1]
            slopes_per_trial_pos = {trial: [] for trial in range(num_trials)}
            slopes_per_trial_neg = {trial: [] for trial in range(num_trials)}

        # Filter based on task type and region
        if row['task_type'] == target_task_type and row['region_name'] in target_regions:
            baseline_accum.append(baseline_data_normalized)
            # scene_accum.append(trial_avg_plot_data)
            scene_accum.append(scene_data_normalized)
            response_accum.append(response_data_normalized)

            # **Initialize a separate dictionary for the current unit's slopes**
            current_unit_slopes = {i: [] for i in range(-4, 6)}

            # Calculate and store slopes and correlations per scene distance
            for scene_idx in range(response_data_normalized.shape[0]):
                scene_distance = scene_idx - 4  # Adjust to center at 0
                # Exclude NaNs in trial_avg_plot_data
                valid_trials = ~np.isnan(trial_avg_plot_data[scene_idx, :])
                if np.sum(valid_trials) > 1:
                    slope, r_value = calculate_slope(trial_avg_plot_data[scene_idx, valid_trials])
                    # slope = r_value # TEST
                else:
                    slope, r_value = np.nan, np.nan
                slopes_per_scene[scene_distance].append(slope)
                correlations_per_scene[scene_distance].append(r_value)

                # **Append slope to current_unit_slopes if it's not NaN**
                if not np.isnan(slope):
                    current_unit_slopes[scene_distance].append(slope)

            # Calculate and store slopes per trial for positive and negative scene distances
            for trial in range(trial_avg_plot_data.shape[1]):
                # Positive scene distances
                y_pos = trial_avg_plot_data[positive_scene_indices, trial]
                x_pos = scene_distances[positive_scene_indices]
                if len(x_pos) > 1 and not np.isnan(y_pos).all():
                    slope_pos, _ = linregress(x_pos, y_pos)[:2]
                else:
                    slope_pos = np.nan
                slopes_per_trial_pos[trial].append(slope_pos)

                # Negative scene distances
                y_neg = trial_avg_plot_data[negative_scene_indices, trial]
                x_neg = scene_distances[negative_scene_indices]
                if len(x_neg) > 1 and not np.isnan(y_neg).all():
                    slope_neg, _ = linregress(x_neg, y_neg)[:2]
                else:
                    slope_neg = np.nan
                slopes_per_trial_neg[trial].append(slope_neg)

            # Calculate Delta Slope: (Avg last 3 trials) - (Avg first 3 trials)
            # Compute combined slopes per trial (slope_pos - slope_neg)
            combined_slopes = []
            for trial in range(trial_avg_plot_data.shape[1]):
                # Average over the last 3 trials
                if len(slopes_per_trial_pos[trial]) >= 3:
                    avg_last3_pos = np.nanmean(slopes_per_trial_pos[trial][-3:])
                else:
                    avg_last3_pos = np.nanmean(slopes_per_trial_pos[trial])

                if len(slopes_per_trial_neg[trial]) >= 3:
                    avg_last3_neg = np.nanmean(slopes_per_trial_neg[trial][-3:])
                else:
                    avg_last3_neg = np.nanmean(slopes_per_trial_neg[trial])

                if not np.isnan(avg_last3_pos) and not np.isnan(avg_last3_neg):
                    combined_slopes.append(avg_last3_pos - avg_last3_neg)
                else:
                    combined_slopes.append(np.nan)

            # Convert to numpy array for easy slicing
            combined_slopes = np.array(combined_slopes)

            # Calculate average of last 3 trials minus average of first 3 trials
            if len(combined_slopes) >= 6:
                #avg_last3 = np.nanmean(combined_slopes[-3:])
                #avg_first3 = np.nanmean(combined_slopes[:3])
                avg_last3 = np.nanmean(combined_slopes[-N_TRIALS_PER_PHASE:])
                avg_first3 = np.nanmean(combined_slopes[:(N_TRIALS_PER_PHASE+1)])
                delta_slope = avg_last3 - avg_first3
                # Append to delta_slopes_list
                if delta_slopes_list is not None:
                    condition = 'HPC' if "HPC" in target_regions else 'non-HPC'
                    delta_slopes_list.append({
                        'Region': condition,
                        'Task': target_task_type,
                        'Delta_Slope': delta_slope
                    })
            else:
                print(f"Warning: Not enough trials to calculate delta slope for unit {idx}.")

    # Only proceed if there is accumulated data
    if baseline_accum and scene_accum and response_accum:
        avg_baseline = np.mean(baseline_accum, axis=0)

        avg_scene = np.mean(scene_accum, axis=0)

        avg_response = np.mean(response_accum, axis=0)

        # Calculate average slopes and SEMs for each scene distance
        avg_slopes = []
        sem_slopes = []
        for i in range(-4, 6):
            if slopes_per_scene[i]:
                try:
                    avg_slopes.append(np.nanmean(slopes_per_scene[i]))
                except ValueError:
                    avg_slopes.append(np.nan)
                    print(f"Warning: Cannot compute mean for scene distance {i} due to all NaN values.")
                try:
                    sem_slopes.append(sem(slopes_per_scene[i], nan_policy='omit'))
                except ValueError:
                    sem_slopes.append(np.nan)
                    print(f"Warning: Cannot compute SEM for scene distance {i} due to insufficient data.")
            else:
                avg_slopes.append(np.nan)
                sem_slopes.append(np.nan)
                print(f"Warning: No slope data available for scene distance {i}.")

        region_type = 'HPC' if 'HPC' in target_regions else 'non-HPC'

        # Generate summary plots
        # ---------------------------------------------
        # Create and save Individual Unit Slopes Plot
        # ---------------------------------------------
        fig_slopes, ax_slopes = plt.subplots(figsize=(5, 5))  # Width same as original, height half
        plot_individual_unit_slopes(ax_slopes, slopes_per_scene, cmap=BASELINE_SCENE_RATE_CMAP,
                                    color=REGION_COLOR[region_type][target_task_type])

        # Set title for the individual unit slopes plot
        fig_slopes.suptitle(f"{region_type}, {target_task_type} {N_TRIALS_PER_PHASE} trials per phase (+1 early baseline)", fontsize=11)

        # Adjust layout and save the individual unit slopes figure
        #sns.despine(left=True, bottom=True)
        fig_slopes_filename = f"averaged_plots_regions_{region_type}_{target_task_type}_N{N_TRIALS_PER_PHASE}_slopes.pdf"
        plt.savefig(fig_slopes_filename,format='pdf')
        plt.close(fig_slopes)
        print(f"Saved {fig_slopes_filename}")

        # ---------------------------------------------
        # Create and save Heatmap Plot
        # ---------------------------------------------
        fig_heatmap, ax_heatmap = plt.subplots(figsize=(5, 5))  # Width same as original, height half

        # Plot transposed average scene heatmap (Bottom subplot)
        # To represent circularity, duplicate the first few columns at the end and add the extra -5 column
        duplication_count = 2  # Number of columns to duplicate

        # **Duplicate the first few columns at the end and add an extra -5 column by duplicating +5**
        # Assuming scene_distances range from -4 to +5, and +5 corresponds to index 9
        # So, the extra -5 column is a duplicate of +5
        duplicated_avg_scene = np.hstack((
            avg_scene[-1:, :].T,          # Duplicate +5 as -5
            avg_scene.T,                   # Original scenes
            avg_scene[:duplication_count, :].T  # Duplicate first few scenes for circularity
        ))
        # Now, duplicated_avg_scene has:
        # Column 0: -5 (duplicate of +5)
        # Columns 1-10: -4 to +5
        # Columns 11-12: -4, -3

        # **Set the column corresponding to scene distance 0 to NaN to whiten it out**
        duplicated_avg_scene[:, 5] = np.nan  # Scene distance 0 is at column index 5

        sns.heatmap(duplicated_avg_scene, ax=ax_heatmap, cmap=BASELINE_SCENE_RATE_CMAP,
                    cbar=False, vmin=BASELINE_SCENE_VMIN, vmax=BASELINE_SCENE_VMAX)
        ax_heatmap.set_title("Average Raw Scene Rate Per Trial Per Scene")

        # **Adjust X-Tick Labels for Heatmap (Shifted to Bin Centers)**
        total_scenes = duplicated_avg_scene.shape[1]
        ax_heatmap.set_xticks(np.arange(total_scenes) + 0.5)  # Shift by half a bin

        # Create labels with circular mapping, including the duplicated ones
        # First label corresponds to -5 (duplicated +5)
        circular_labels = ['-5'] + list(scene_distances) + list(scene_distances[:duplication_count])
        ax_heatmap.set_xticklabels(circular_labels, rotation=45, ha='right')  # Rotate labels for readability

        # **Adjust Y-Tick Labels for Heatmap (Trial Numbers starting at 1 and Shifted to Bin Centers)**
        num_trials = avg_scene.T.shape[0]
        ax_heatmap.set_yticks(np.arange(num_trials) + 0.5)  # Shift by half a bin
        ax_heatmap.set_yticklabels(np.arange(1, num_trials + 1))  # Set labels starting at 1
        #ax_heatmap.set_xlim([0, total_scenes])  # Adjust x-limits to include the extra column
        ax_heatmap.set_xlim([0,12])  # Adjust x-limits to include the extra column

        # **Ensure Alignment with Top Subplot**
        # The top subplot's x-axis ranges from -5.25 to +6.25, covering all scene distances including the extra column

        # Adjust layout and save the heatmap figure
        sns.despine(left=True, bottom=True)
        fig_slopes.suptitle(f"{region_type}, {target_task_type}", fontsize=16)
        fig_heatmap_filename = f"averaged_plots_regions_{region_type}_{target_task_type}_heatmap.pdf"
        plt.savefig(fig_heatmap_filename,format='pdf')
        plt.close(fig_heatmap)
        print(f"Saved {fig_heatmap_filename}")

    return master_df


def main():
    """
    Main function to execute the data processing and plotting.
    """
    # Load the DataFrame
    try:
        master_df = pd.read_pickle('unit_condition_pair_info_df.pkl')
        #master_df = pd.read_pickle('unit_condition_pair_info_df_ANY_SELECTIVITY.pkl')
    except FileNotFoundError:
        print("Error: 'unit_condition_pair_info_df.pkl' not found.")
        return
    except Exception as e:
        print(f"An error occurred while loading the DataFrame: {e}")
        return

   
    # Initialize a list to collect delta slopes
    delta_slopes_data = []

    # Define target regions and task types
    region_names_list = [
        {"regions": ["HPC"], "tasks": ["structured", "random"]},
        {"regions": ['Amygdala', 'Cingulate', 'Insula'], "tasks": ["structured", "random"]}
    ]

    # Process and plot for each set of regions and their respective tasks
    for region_task in region_names_list:
        regions = region_task["regions"]
        tasks = region_task["tasks"]
        for task in tasks:
            rename_and_plot(master_df, target_task_type=task, target_regions=regions,
                           smoothing_window_size=TRIAL_SMOOTH_WIND, offset=0.0, delta_slopes_list=delta_slopes_data)

    # Convert delta_slopes_data to DataFrame
    if delta_slopes_data:
        delta_slopes_df = pd.DataFrame(delta_slopes_data)
        #plot_violin_bar(delta_slopes_df)
    else:
        print("No delta slope data collected. Violin bar plot will not be generated.")

if __name__ == "__main__":
    main()

