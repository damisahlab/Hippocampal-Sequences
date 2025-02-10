#!/usr/bin/env python3
from utils import local_paths
# -*- coding: utf-8 -*-
from utils import pdfTextSet
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from utils import taskInfoFunctions as tif
import pdb
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.stats import sem
import sys

processed_data_dir=local_paths.get_processed_data_dir()
PLOT_INDIVIDUAL_UNITS=True

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

def plot_unstacked_line_plots(df, condition_name, phase_name, smoothing_window=12, light_smoothing_window=10):
    # Filter the DataFrame for the desired task type
    filtered_df = df[df['task_type'] == f'{condition_name}']
    
    # Initialize list to accumulate baseline-subtracted early averages for all units
    all_units_early_avg = []
    
    for index, row in filtered_df.iterrows():
        # Extract the firing rate data
        sessID = row['sessIDStr']
        initial_peak_scene_num = row['initial_peak_scene_num']
        
        firing_rate_data = np.array(row['firing_rate_per_centeredScene_per_trial_per_timeBin'])
        
        # Compute the early average data based on phase
        if phase_name == 'change':
            early_avg = np.mean(firing_rate_data[:, -3:, :], axis=1) - np.mean(firing_rate_data[:, :3, :], axis=1)
        elif phase_name == 'early':
            early_avg = np.mean(firing_rate_data[:, :3, :], axis=1)
        elif phase_name == 'late':
            early_avg = np.mean(firing_rate_data[:, -3:, :], axis=1)
        elif phase_name == 'all':
            early_avg = np.mean(firing_rate_data[:, :, :], axis=1)
        else:
            raise ValueError("Invalid phase_name. Choose from 'early', 'late', 'change', or 'all'.")
        
        # Calculate the adjusted time bins
        prior_gap_duration = 0.01
        baseline_plus_scene_duration = 2.0  # duration of each scene in seconds
        post_gap_duration = 0.23  # see video analysis of task
        total_duration_per_scene = prior_gap_duration + baseline_plus_scene_duration + post_gap_duration
        adjusted_time_bins = []
        
        time_bounds_per_scene_num = tif.get_scene_time_bounds(sessID, condition_name,cacheDir=processed_data_dir)
        
        continuous_early_avg = []
        unit_early_avg = []
        
        aligned_spike_times = []
        
        # Determine the maximum number of trials based on phase
        if phase_name == 'early':
            max_trial_idx = 7
        elif phase_name == 'all':
            max_trial_idx = 15
        else:
            max_trial_idx = 5  # Default value; adjust as needed
        
        for trial_idx in range(max_trial_idx):
            preferred_scene_time_start = time_bounds_per_scene_num[initial_peak_scene_num][trial_idx][0]
            
            trial_spike_times_relative = []
            for scene_idx in range(early_avg.shape[0]):
                scene_start_time = scene_idx * total_duration_per_scene - 10
                scene_end_time = scene_start_time + baseline_plus_scene_duration
                scene_time_bins = np.linspace(scene_start_time, scene_end_time, early_avg.shape[1])
                adjusted_time_bins.extend(scene_time_bins)
                
                # Light smoothing here
                baseline_subtracted = early_avg[scene_idx]
                baseline_subtracted_smoothed = uniform_filter1d(baseline_subtracted, size=light_smoothing_window)
                unit_early_avg.append(baseline_subtracted_smoothed)
                
                continuous_early_avg.extend(baseline_subtracted_smoothed)
                
                # Find the spike times that fall within the interval for the current trial
                for spike_time in row['all_spike_times_in_condition']:
                    if (time_bounds_per_scene_num[scene_idx + 1][trial_idx][0] - 1.01 <= spike_time <= 
                        time_bounds_per_scene_num[scene_idx + 1][trial_idx][1] + 0.23):
                        # Align spike times to the preferred scene for the current trial
                        spike_times_relative = spike_time - preferred_scene_time_start
                        trial_spike_times_relative.append(spike_times_relative)
            
            aligned_spike_times.append(trial_spike_times_relative)
            
            # Alternative spike time alignment (commented out as per original code)
            '''
            # Align spike times to the preferred scene for each trial
            spike_times_relative = row['all_spike_times_in_condition'][trial_idx] - preferred_scene_time_start
            trial_spike_times_relative.append(spike_times_relative)
            pdb.set_trace()
            aligned_spike_times.append(trial_spike_times_relative)
            '''
        
        unit_early_avg = np.array(unit_early_avg)
        
        mean_value = np.nanmean(unit_early_avg)
        std_value = np.nanstd(unit_early_avg)
        zScoredCurrUnit = (unit_early_avg - mean_value) / std_value
        if use_z_score:
            all_units_early_avg.append(zScoredCurrUnit)
        else:
            all_units_early_avg.append(unit_early_avg)
       
        filename = f"unstacked_line_and_raster_plot_{row['sessIDStr']}_{condition_name}_{phase_name}_{row['region_name']}_{row['chNameStr']}_{row['unitNumStr']}_{index}.pdf"
        if filename!='unstacked_line_and_raster_plot_pt7_sess1_random_early_HPC_ch241_unit1_86.pdf':
            continue
        if PLOT_INDIVIDUAL_UNITS:
            # Apply causal filter after concatenation
            continuous_early_avg = causal_filter(np.array(continuous_early_avg), window_size=smoothing_window)
            min_time = np.min(adjusted_time_bins)
            max_time = np.max(adjusted_time_bins)
            
            # Define scene distances relative to a preferred scene
            preferred_scene_idx = 4  # Example: 5th scene is preferred (0-based indexing)
            scene_distances = np.arange(early_avg.shape[0]) - preferred_scene_idx
            
            # Normalize scene distances
            norm = colors.Normalize(vmin=scene_distances.min(), vmax=scene_distances.max())
            
            # Apply Viridis colormap
            cmap = cm.get_cmap('viridis')
            colorsV = cmap(norm(scene_distances))
            
            # Plotting
            fig, (ax2, ax1) = plt.subplots(2, 1, figsize=(18, 12), gridspec_kw={'height_ratios': [0.5, 1]})
            ax1.plot(adjusted_time_bins, continuous_early_avg, label=f'{row["sessIDStr"]} {row["chNameStr"]} {row["unitNumStr"]} {phase_name}', 
                     alpha=0.5, color=plotColor, linewidth=5)
            
            for scene_idx in range(early_avg.shape[0]):
                scene_start_time = scene_idx * total_duration_per_scene - 10
                scene_shade_start = scene_start_time + 1.01  # Start shading at 1 second within the scene
                scene_shade_end = scene_shade_start + 1.23  # Shade for 1 second duration
                alpha_val = 0.3 if scene_idx != preferred_scene_idx else 0.5  # Emphasize the preferred scene with higher opacity
                ax1.axvspan(scene_shade_start, scene_shade_end, color=colorsV[scene_idx], alpha=alpha_val)
            
            ax1.axvline(x=0, color='red', linestyle='--')  # Mark the 0 time point for reference
            ax1.set_title(f"Unstacked Line Plot: Session: {row['sessIDStr']}, Region: {row['region_name']}, Unit: {row['unitNumStr']}")
            ax1.set_xlabel('Time from preferred scene onset (s)')
            ax1.set_ylabel('Firing Rate')
            
            # Spike time raster plot
            #ax2.eventplot(aligned_spike_times, color='black', linelengths=0.8, linewidths=1)
            ax2.eventplot(aligned_spike_times, color='black', linelengths=0.8, linewidths=0.25)
            ax2.set_xlabel('Time from preferred scene onset (s)')
            ax2.set_ylabel('Trials')
            
            for scene_idx in range(early_avg.shape[0]):
                scene_start_time = scene_idx * total_duration_per_scene - 10
                scene_shade_start = scene_start_time + 1.01  # Start shading at 1 second within the scene
                scene_shade_end = scene_shade_start + 1.23  # Shade for 1 second duration
                alpha_val = 0.3 if scene_idx != preferred_scene_idx else 0.5  # Emphasize the preferred scene with higher opacity
                ax2.axvspan(scene_shade_start, scene_shade_end, color=colorsV[scene_idx], alpha=alpha_val)
            
            ax1.axhline(0, color='black', linestyle='--')  # Mark the 0 firing rate
            ax1.axvline(x=0, color='black', linestyle='--')  # Mark the 0 time point for reference
            ax1.set_xlabel('Time from preferred scene onset (s)')
            
            ax1.set_xlim([-10, 10])
            ax2.set_xlim([-10, 10])
            
            # Save the figure

            plt.savefig(filename)
            plt.close(fig)
            print(f'Processed row {index + 1}')
    
    # Convert the list to a 3D numpy array with shape (num_scenes, num_time_bins, num_units)
    all_units_early_avg = np.array(all_units_early_avg).transpose(1, 2, 0)
    num_units = all_units_early_avg.shape[2]  # Number of units
    
    # Compute the overall average and SEM across units
    overall_early_avg = np.mean(all_units_early_avg, axis=2)
    overall_early_sem = sem(all_units_early_avg, axis=2)
    
    # Apply causal filter to the overall data
    continuous_overall_early_avg = np.concatenate(overall_early_avg)
    continuous_overall_early_avg = causal_filter(continuous_overall_early_avg, window_size=smoothing_window)
    continuous_overall_early_sem = np.concatenate(overall_early_sem)
    
    # Define scene distances for overall plot
    preferred_scene_idx = 4  # Example: 5th scene is preferred (0-based indexing)
    scene_distances_overall = np.arange(overall_early_avg.shape[0]) - preferred_scene_idx
    
    # Normalize scene distances for overall plot
    norm_overall = colors.Normalize(vmin=scene_distances_overall.min(), vmax=scene_distances_overall.max())
    
    # Apply Viridis colormap for overall plot
    cmap_overall = cm.get_cmap('viridis')
    colorsV_overall = cmap_overall(norm_overall(scene_distances_overall))
    
    # Define adjusted_time_bins for overall plot
    # Assuming the time_bins are consistent across units
    # You may need to adjust this based on your actual data structure
    adjusted_time_bins_overall = np.linspace(-10, 10, len(continuous_overall_early_avg))
    
    # Plot the overall average line plot with SEM shading
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(adjusted_time_bins_overall, continuous_overall_early_avg, color=plotColor)
    ax.fill_between(adjusted_time_bins_overall, continuous_overall_early_avg - continuous_overall_early_sem,
                    continuous_overall_early_avg + continuous_overall_early_sem, alpha=0.3, color=plotColor)
    
    for scene_idx in range(overall_early_avg.shape[0]):
        scene_start_time = scene_idx * total_duration_per_scene - 10
        scene_shade_start = scene_start_time + 1.01  # Start shading at 1 second within the scene
        scene_shade_end = scene_shade_start + 1.23  # Shade for 1 second duration
        alpha_val = 0.3 if scene_idx != preferred_scene_idx else 0.5  # Emphasize the preferred scene
        ax.axvspan(scene_shade_start, scene_shade_end, color=colorsV_overall[scene_idx], alpha=alpha_val)
    
    ax.axhline(0, color='black', linestyle='--')  # Mark the 0 firing rate
    ax.axvline(x=0, color='black', linestyle='--')  # Mark the 0 time point for reference
    ax.set_title(f"Firing rate over time for selective HPC units early in {condition_name} condition (n={num_units} units)")
    ax.set_xlabel('Time from preferred scene onset (s)')
    
    ax.set_xlim([-10, 10])
    if use_z_score:
        ax.set_ylabel('Baseline-subtracted firing rate (Z score)')
    else:
        ax.set_ylabel('Baseline-subtracted firing rate (Hz)')
    
    # Add colorbar to represent scene colors using Viridis
    sm = cm.ScalarMappable(cmap='viridis', norm=norm_overall)
    sm.set_array([])  # Required for some matplotlib versions
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
    cbar.set_label('Scene Distance from Preferred Scene', rotation=270, labelpad=20)
    
    if use_z_score:
        if phase_name == 'change':
            plt.ylim([-0.75, 1])
        if phase_name == 'early':
            plt.ylim([-1, 1.6])
    else:
        plt.ylim([-2.1, 2.1])
    
    #plt.savefig(f"ANY_SELECTIVITY_EXAMPLE_overall_unstacked_continuous_line_plot_{condition_name}_{phase_name}_Z_{use_z_score}.pdf")
    plt.close(fig)

if __name__ == "__main__":
    # Ensure correct number of arguments
    
    condition_name = sys.argv[1]  # random or structured
    phase_name = sys.argv[2]  # early, late, or change
    
    use_z_score = False
    plotColor = 'black'
    if condition_name == 'structured':
        plotColor = "#8B0000"

    # Load the DataFrame
    try:
        unit_condition_pair_info_df = pd.read_pickle('unit_condition_pair_info_df_ANY_SELECTIVITY.pkl')
    except FileNotFoundError:
        print("Error: The file 'unit_condition_pair_info_df_ANY_SELECTIVITY.pkl' was not found.")
        sys.exit(1)
    
    # Plot the unstacked line plots
    plot_unstacked_line_plots(unit_condition_pair_info_df, condition_name, phase_name)

