import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from utils import lfp_analysis as lfpA
from utils import dataPathDict as dpd
from utils import taskInfoFunctions as tif
from utils import lfp_loader as ll
from utils import pdfTextSet
from utils import region_info_loader as ril
from scipy.stats import zscore
import pdb
from utils import local_paths

VMIN = -3
VMAX = 3

# Cache file paths
cache_dir = 'spectrogramExamples'
os.makedirs(cache_dir, exist_ok=True)
data_file_path = os.path.join(cache_dir, 'processed_data.pkl')
processed_data_dir=local_paths.get_processed_data_dir()

# Save data function
def save_data(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

# Load data function
def load_data(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

# Optional Z-scoring function
def zscore_spectrogram(spectrogram):
    """
    Z-scores the spectrogram by subtracting the mean and dividing by the standard deviation
    of the entire matrix.
    
    Parameters:
    - spectrogram (np.ndarray): The input spectrogram matrix to be Z-scored.

    Returns:
    - np.ndarray: The Z-scored spectrogram matrix.
    """
    mean_val = np.nanmean(spectrogram)
    std_val = np.nanstd(spectrogram)
    
    # Avoid division by zero in case the standard deviation is zero
    if std_val == 0:
        return spectrogram - mean_val
    else:
        return (spectrogram - mean_val) / std_val


import matplotlib.pyplot as plt
import numpy as np
import os

# Function to plot early and late trial spectrograms
def plot_spectrograms(early_spectrogram_codes, late_spectrogram_codes, condition_name, sessID, apply_zscore=False):
    fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(20, 8), constrained_layout=True)  # 2 rows, 10 columns

    # Helper function to average and stack channels in groups of 8
    def average_and_stack_channels(spectrogram_code):
        num_channels = spectrogram_code.shape[0]  # Number of channels
        group_size = 8
        stacked_spectrograms = []

        # Loop over the channels in groups of 8 (or fewer for remaining channels)
        for i in range(0, num_channels, group_size):
            channels_group = spectrogram_code[i:i + group_size]

            if not all(channels_group[0].shape == x.shape for x in channels_group):
                raise ValueError(f"Channel shapes are inconsistent in group starting at index {i}")

            # Average across the selected channels
            avg_spectrogram = np.mean(channels_group, axis=0)

            # Optionally Z-score the spectrogram
            if apply_zscore:
                avg_spectrogram = zscore_spectrogram(avg_spectrogram)  # Z-score across the whole matrix

            stacked_spectrograms.append(avg_spectrogram)

        # Stack the averaged spectrograms vertically
        stacked_spectrograms = np.vstack(stacked_spectrograms)

        return stacked_spectrograms

    # Define time and frequency axis labels
    time_bins = np.linspace(0, 0.6, early_spectrogram_codes[0].shape[1])  # Assuming time length of 0.6s
    freq_bins = np.linspace(0.5, 70, early_spectrogram_codes[0].shape[2])  # Frequency range from 0.5Hz to 70Hz

    # Plot Early and Late spectrograms in 2-row, 10-column format
    im = None  # To store the last imshow object for the colorbar
    for i in range(10):  # 10 scenes for early and late conditions
        # Early spectrograms: first row
        stacked_early = average_and_stack_channels(early_spectrogram_codes[i])
        im = axes[0, i].imshow(stacked_early, aspect='auto', origin='lower', cmap='plasma', vmin=VMIN, vmax=VMAX,
                               extent=[time_bins[0], time_bins[-1], freq_bins[0], freq_bins[-1]],interpolation='nearest')
        axes[0, i].set_yticks([])  # Remove y-tick labels for each subplot

        # Late spectrograms: second row
        stacked_late = average_and_stack_channels(late_spectrogram_codes[i])
        im = axes[1, i].imshow(stacked_late, aspect='auto', origin='lower', cmap='plasma', vmin=VMIN, vmax=VMAX,
                               extent=[time_bins[0], time_bins[-1], freq_bins[0], freq_bins[-1]],interpolation='nearest')
        axes[1, i].set_yticks([])  # Remove y-tick labels for each subplot

        # Set column titles only on the top row
        axes[0, i].set_title(f'Scene {i + 1}', fontsize=14)

    # Add a single colorbar for the entire figure
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Power (Z)', fontsize=14)

    # Set shared x-axis and y-axis labels
    #fig.text(0.5, 0.04, 'Time from scene onset (sec)', ha='center', fontsize=12)
    #fig.text(0.04, 0.5, 'Frequency (0.5-70 Hz)', va='center', rotation='vertical', fontsize=12)

    # Set the overall title and save the figure
    fig.suptitle(f'Session {sessID} - Condition: {condition_name}', fontsize=9)
    fig_file_path = os.path.join(cache_dir, f'avg_spectrogram_{sessID}_{condition_name}_early_late.pdf')
    plt.savefig(fig_file_path)
    plt.close()
    print(f"Figure saved to {fig_file_path}")



# Function for processing and plotting (preserved)
def process_and_plot():
    # Check if cached data exists
    if False and os.path.exists(data_file_path):
        print("Loading cached data...")
        early_spectrogram_codes, late_spectrogram_codes = load_data(data_file_path)
    else:
        print("Processing data...")

        #sessNames = dpd.getSessNames()
        sessNames = dpd.getSessNameFig3Eg()
        max_channel_number = 24
        min_time = 0
        max_time = 0.6
        min_freq = 0.5
        max_freq = 70
        num_trials_in_timing = 3  # First 3 and last 3 trials

        # Placeholder for early and late spectrogram_codes (adjust for real data processing)
        early_spectrogram_codes = [None] * 10
        late_spectrogram_codes = [None] * 10

        # Loop through sessions and processm
        for sessID in sessNames:
            # Channel selection based on region_type (modify as needed)
            region_type = 'hpc'  # Example, replace with dynamic input if needed
            channel_range = ril.get_hpc_lfp_channels(sessID, 'both sides')

            # Process channels
            trial_scene_spectrograms = {
                trial_group: {trial: {scene: [] for scene in range(1, 11)} for trial in range(15)} 
                for trial_group in range(2)  # Assuming 2 trial groups (random/structured)
            }

            for channel in channel_range:
                #downsampled_lfp, all_event_times, originalFs, dsFact = ll.load_lfp_and_event_times(sessID, int(channel))
                downsampled_lfp, all_event_times, originalFs, dsFact = ll.load_downsampled_lfp_and_events_with_cache(sessID, int(channel),dataDir=processed_data_dir)
                if downsampled_lfp is None:
                    continue

                for scene_num in range(1, 11):
                    classified_scene_event_times = tif.classify_events_for_scene_with_trial_num(sessID, scene_num, all_event_times, early_threshold=num_trials_in_timing, late_threshold=num_trials_in_timing)

                    for idx, (event_time, condition, timing, trial_within_group) in enumerate(classified_scene_event_times):
                        f, t, f_edges, t_edges, Sxx = lfpA.compute_event_triggered_spectrogram(
                            downsampled_lfp, event_time, originalFs / dsFact, 
                            min_freq=min_freq, max_freq=max_freq, nperseg=256)
                        
                        Sxx_filtered = Sxx[:, (t >= min_time) & (t <= max_time)]
                        if Sxx_filtered is not None and Sxx_filtered.size > 0:
                            trial_group_idx_local = idx // 15
                            trial_scene_spectrograms[trial_group_idx_local][trial_within_group-1][scene_num].append(Sxx_filtered)

            # Flatten and average for first 3 ("Early") and last 3 ("Late") trials
            for trial_group in range(2):
                trials = list(trial_scene_spectrograms[trial_group].keys())
                early_trials_indices = trials[:3]  # First 3 trials
                late_trials_indices = trials[-3:]  # Last 3 trials

                for scene in range(1, 11):
                    # Early trials
                    early_trials = np.mean(
                        [trial_scene_spectrograms[trial_group][i][scene] for i in early_trials_indices], 
                        axis=0
                    )
                    # Late trials
                    late_trials = np.mean(
                        [trial_scene_spectrograms[trial_group][i][scene] for i in late_trials_indices], 
                        axis=0
                    )
                    
                    early_spectrogram_codes[scene - 1] = early_trials if early_trials is not None else np.zeros((10, 10))
                    late_spectrogram_codes[scene - 1] = late_trials if late_trials is not None else np.zeros((10, 10))

                # Get the condition name for the current trial group
                condition_name = tif.get_condition_name_for_sessID(sessID, trial_group)

                # Plot the spectrograms for this condition
                plot_spectrograms(early_spectrogram_codes, late_spectrogram_codes, condition_name, sessID, apply_zscore=True)

        # Cache the processed data
        save_data((early_spectrogram_codes, late_spectrogram_codes), data_file_path)

# Execute the function
process_and_plot()


