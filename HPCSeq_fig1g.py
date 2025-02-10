from utils import taskInfoFunctions as tif
import pdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.stats import sem
from matplotlib.colors import Normalize
import matplotlib.cm as cm  # Import the cm module
import sys
from utils import pdfTextSet
from utils import local_paths

PLOT_INDIVIDUAL_UNITS=False
processed_data_dir=local_paths.get_processed_data_dir()

# Constants for plot limits
MIN_X = -10
MAX_X = 12.5

# Retrieve command-line arguments
condition_name = sys.argv[1]  # random or structured
phase_name = sys.argv[2]      # early, late or change

if phase_name == 'change':
    MIN_Y = -3
    MAX_Y = 3
else:
    MIN_Y=None
    MAX_Y=None
use_z_score = False
# use_z_score = True
plotColor = 'black'
if condition_name == 'structured':
    # plotColor = 'red'
    plotColor = "#8B0000"


plotColor = "#8B0000"

def causal_filter(data, window_size):
    """
    Apply a causal filter by averaging over the past values only,
    dynamically adjusting the window size near the edges.
    """
    filtered_data = np.zeros_like(data)
    for i in range(len(data)):
        current_window_size = min(i + 1, window_size)
        filtered_data[i] = np.mean(data[max(0, i - current_window_size + 1):i + 1])
    return filtered_data

def plot_unstacked_line_plots(df, smoothing_window=12, light_smoothing_window=10):
    # Filter the DataFrame for the desired region and task type
    filtered_df = df[(df['region_name'] == 'HPC') & (df['task_type'] == f'{condition_name}')]

    # Initialize list to accumulate baseline-subtracted early averages for all units
    all_units_early_avg = []

    # ======= Define Viridis Colormap =======
    cmap = cm.get_cmap('viridis')  # Full Viridis colormap
    # ======================================

    # Define the preferred scene index (0-based)
    preferred_scene_idx = 4
    num_scenes = 10  # Total number of scenes

    # ======= Manual Color Mapping for Scenes =======
    # Assign colors based on distance from the preferred scene
    colorsV = []
    max_distance = 5  # Maximum distance for normalization

    for i in range(num_scenes):
        distance = abs(i - preferred_scene_idx)
        if distance > max_distance:
            distance = max_distance  # Cap the distance to maintain symmetry
        normalized_distance = distance / max_distance  # Normalize to [0, 1]
        #color = cmap(1.0 - normalized_distance)  # Scene 0 and 9: cmap(1.0), Scene 4: cmap(0.75)
        color = cmap(normalized_distance)  # Scene 0 and 9: cmap(1.0), Scene 4: cmap(0.75)
        colorsV.append(color)
    # Now, colorsV[0] and colorsV[9] have the same color, colorsV[1] and colorsV[8], etc.
    # ==============================================

    for index, row in filtered_df.iterrows():
        # Extract the firing rate data
        sessID = row['sessIDStr']
        initial_peak_scene_num = row['initial_peak_scene_num']

        firing_rate_data = np.array(row['firing_rate_per_centeredScene_per_trial_per_timeBin'])

        # Compute the early average data by averaging across trials (axis 1)
        if phase_name == 'change':
            early_avg = np.mean(firing_rate_data[:, -3:, :], axis=1) - np.mean(firing_rate_data[:, :3, :], axis=1)
        elif phase_name == 'early':
            early_avg = np.mean(firing_rate_data[:, :3, :], axis=1)
        elif phase_name == 'late':
            early_avg = np.mean(firing_rate_data[:, -3:, :], axis=1)
        else:
            raise ValueError("Invalid phase_name. Choose from 'early', 'late', or 'change'.")

        # Calculate the adjusted time bins
        prior_gap_duration = 0.01
        baseline_plus_scene_duration = 2.0  # duration of each scene in seconds
        post_gap_duration = 0.23  # see video analysis of task 
        total_duration_per_scene = prior_gap_duration + baseline_plus_scene_duration + post_gap_duration
        adjusted_time_bins = []

        time_bounds_per_scene_num = tif.get_scene_time_bounds(sessID, condition_name,cacheDir=processed_data_dir)

        continuous_early_avg = []
        unit_early_avg = []

        for scene_idx in range(early_avg.shape[0]):
            scene_start_time = scene_idx * total_duration_per_scene - 10
            scene_end_time = scene_start_time + baseline_plus_scene_duration
            scene_time_bins = np.linspace(scene_start_time, scene_end_time, early_avg.shape[1])
            adjusted_time_bins.extend(scene_time_bins)
            
            # Light smoothing here
            curr_scene_baseline = early_avg[scene_idx][0:20].mean()
            baseline_subtracted = early_avg[scene_idx] - curr_scene_baseline
            baseline_subtracted_smoothed = uniform_filter1d(baseline_subtracted, size=light_smoothing_window)
       
            unit_early_avg.append(baseline_subtracted_smoothed)

            continuous_early_avg.extend(baseline_subtracted_smoothed)

        unit_early_avg = np.array(unit_early_avg)

        mean_value = np.nanmean(unit_early_avg)
        std_value = np.nanstd(unit_early_avg)
        zScoredCurrUnit = (unit_early_avg - mean_value) / std_value
        if use_z_score:
            all_units_early_avg.append(zScoredCurrUnit)
        else:
            all_units_early_avg.append(unit_early_avg)
        
        if PLOT_INDIVIDUAL_UNITS: 
            # Apply causal filter after concatenation
            continuous_early_avg = causal_filter(np.array(continuous_early_avg), window_size=smoothing_window)

            # Plot the continuous curve for this unit
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(adjusted_time_bins, continuous_early_avg, 
                    label=f'{row["sessIDStr"]} {row["chNameStr"]} {row["unitNumStr"]} {phase_name}', 
                    alpha=0.5)

            # ======= Apply Manual Color Mapping to Shading =======
            for scene_idx in range(num_scenes):
                scene_start_time = scene_idx * total_duration_per_scene - 10
                scene_shade_start = scene_start_time + 1.01  # Start shading at 1 second within the scene
                scene_shade_end = scene_shade_start + 1.23  # Shade for 1 second duration
                alpha_val = 0.3 if scene_idx != preferred_scene_idx else 0.5  # Emphasize the preferred scene with higher opacity
                ax.axvspan(scene_shade_start, scene_shade_end, color=colorsV[scene_idx], alpha=alpha_val)
            # ======================================================

            ax.axvline(x=0, color='red', linestyle='--')  # Mark the 0 time point for reference
            #ax.set_title(f"Unstacked Line Plot: Session: {row['sessIDStr']}, Region: {row['region_name']}, Unit: {row['unitNumStr']}")
            ax.set_xlabel('Time from preferred scene onset (s)')
            ax.set_ylabel('Firing Rate')

            # Save the figure
            filename = f"unstacked_line_plot_{row['sessIDStr']}_{condition_name}_{phase_name}_{row['region_name']}_{row['chNameStr']}_{row['unitNumStr']}_{index}.pdf"
            plt.savefig(filename)
            plt.close(fig)
            print(f'Processed row {index + 1}')

    # Convert the list to a 3D numpy array with shape (10, 40, num_units)
    all_units_early_avg = np.array(all_units_early_avg).transpose(1, 2, 0)
    num_units = all_units_early_avg.shape[2]  # Number of units

    # Compute the overall average and SEM across units
    overall_early_avg = np.mean(all_units_early_avg, axis=2)
    overall_early_sem = sem(all_units_early_avg, axis=2)

    # Apply causal filter to the overall data
    continuous_overall_early_avg = np.concatenate(overall_early_avg)
    continuous_overall_early_avg = causal_filter(continuous_overall_early_avg, window_size=smoothing_window)
    continuous_overall_early_sem = np.concatenate(overall_early_sem)

    # ======= Manual Color Mapping for Overall Plot =======
    # Assign colors based on distance from the preferred scene
    colorsV_overall = []
    for i in range(num_scenes):
        distance = abs(i - preferred_scene_idx)
        if distance > max_distance:
            distance = max_distance  # Cap the distance to maintain symmetry
        normalized_distance = distance / max_distance  # Normalize to [0, 1]
        #color = cmap(1.0 - normalized_distance)  # Scene 0 and 9: cmap(1.0), Scene 4: cmap(0.75)
        color = cmap(normalized_distance)  # Scene 0 and 9: cmap(1.0), Scene 4: cmap(0.75)
        colorsV_overall.append(color)
    # =====================================================

    # Define adjusted_time_bins for overall plot
    #adjusted_time_bins_overall = np.linspace(-10, 10, len(continuous_overall_early_avg))
    adjusted_time_bins_overall = np.linspace(0-9.97,22.4-9.97, len(continuous_overall_early_avg))

    # Plot the overall average line plot with SEM shading
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(adjusted_time_bins_overall, continuous_overall_early_avg, color=plotColor, label='Overall Average')
    ax.fill_between(adjusted_time_bins_overall, continuous_overall_early_avg - continuous_overall_early_sem,
                    continuous_overall_early_avg + continuous_overall_early_sem, alpha=0.3, color=plotColor, label='SEM')

    # ======= Apply Manual Color Mapping to Overall Shading =======
    for scene_idx in range(num_scenes):
        scene_start_time = scene_idx * total_duration_per_scene - 10
        scene_shade_start = scene_start_time + 1.01  # Start shading at 1 second within the scene
        scene_shade_end = scene_shade_start + 1.23  # Shade for 1 second duration
        alpha_val = 0.3 if scene_idx != preferred_scene_idx else 0.5  # Emphasize the preferred scene
        ax.axvspan(scene_shade_start, scene_shade_end, color=colorsV_overall[scene_idx], alpha=alpha_val)
    # ===============================================================

    ax.axhline(0, color='black', linestyle='--')  # Mark the 0 firing rate for reference
    ax.axvline(x=0, color='black', linestyle='--')  # Mark the 0 time point for reference
    #ax.set_title(f"Firing rate over time for selective HPC units early in {condition_name} condition (n={num_units} units)")
    ax.set_xlabel('Time from preferred scene onset (s)')
    if use_z_score:
        ax.set_ylabel('Baseline-subtracted firing rate (Z score)')
    else:
        ax.set_ylabel('Baseline-subtracted firing rate (Hz)')

    # ======= Add Colorbar =======
    # Create a ScalarMappable for the colorbar based on manual color mapping
    norm_overall = Normalize(vmin=0, vmax=max_distance)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm_overall)
    sm.set_array([])  # Only needed for older versions of Matplotlib

    # Add the colorbar to the plot
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
    cbar.set_label('Absolute Scene Distance from Preferred Scene', rotation=270, labelpad=20)
    # =========================

    plt.xlim([MIN_X,MAX_X])
    plt.ylim([MIN_Y,MAX_Y])
    # Save the overall average figure
    plt.savefig(f"overall_unstacked_continuous_line_plot_HPC_{condition_name}_{phase_name}_Z_{use_z_score}.pdf")
    plt.close(fig)

# Load the DataFrame
unit_condition_pair_info_df = pd.read_pickle('unit_condition_pair_info_df.pkl')

# Plot the unstacked line plots
plot_unstacked_line_plots(unit_condition_pair_info_df)

