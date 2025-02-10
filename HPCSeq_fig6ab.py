import os  # Import os for directory operations
from utils import single_unit_checkpoint as suc
from scipy.signal import welch, butter, filtfilt, spectrogram
from utils import region_info_loader as ril
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import patchworklib as pw
import sys
from utils import lfp_loader as ll
from utils import dataPathDict as dpd
import pdb
from utils import taskInfoFunctions as ti
from scipy.signal import convolve2d
from matplotlib.cm import get_cmap
import pandas as pd  # For handling DataFrames and pickling
from scipy.stats import mannwhitneyu  # For statistical testing
from scipy.stats import wilcoxon  # Import Wilcoxon test
from utils import local_paths

processed_data_dir=local_paths.get_processed_data_dir()
# Set random seed for reproducibility
np.random.seed(0)

# Configuration Variables
normalize_each_spec = True
brainSideStr = 'both sides'

GROUP_SIZE = 150
EARLY_THRESHOLD = 75
LATE_THRESHOLD = 75

FREQ_BAND = (3, 6)  # Frequency band of interest (Hz)

# Define global variables for intervals
before_interval = (-1, 0)
after_interval = (0, 1)

# ================== Function Definitions ================== #
def plot_aggregated_bar_plot(df_power_change, save_dir):
    """
    Plot an aggregated bar plot showing the mean ± SEM of the power change across all sessions,
    with individual session points as slightly transparent black dots and lines connecting
    points from the same session. Connecting lines are colored red if power increases and blue if power decreases.
    Additionally, perform paired Wilcoxon tests between structured and random conditions within each region
    and annotate the plot with W and p-values.

    Parameters:
    - df_power_change (DataFrame): DataFrame containing power change information for each session.
    - save_dir (str): Directory where the aggregated plot PDF will be saved.

    Returns:
    - Saves the aggregated bar plot PDF with statistical annotations.
    """
    if not df_power_change.empty:
        # Replace specific sessID as per user changes
        df_power_change['sessID'] = df_power_change['sessID'].replace('pt1_sess2', 'pt1_sess1')  # random and structured done in consecutive sessions

        # Define color mapping based on region and condition
        color_mapping = {
            ('HPC', 'random'): 'lightcoral',
            ('HPC', 'structured'): 'red',
            ('AIC', 'random'): '#C4A484',
            ('AIC', 'structured'): 'brown'
        }

        # Calculate mean and SEM for each condition-region combination
        agg_data = df_power_change.groupby(['Region', 'Condition']).agg(
            Mean_Power_Change=('Mean_Power_Change', 'mean'),
            SEM_Power_Change=('Mean_Power_Change', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))
        ).reset_index()

        # Set up the bar plot
        plt.figure(figsize=(8, 8))

        # Define regions and conditions for consistent ordering
        regions_order = ['HPC', 'AIC']
        conditions_order = ['structured', 'random']
        x = np.arange(len(regions_order))  # [0, 1]
        width = 0.5  # Total width for both conditions within a region

        # Calculate offsets for each condition within a region
        num_conditions = len(conditions_order)
        offsets = np.linspace(-width/2, width/2, num_conditions)

        # Plot bars for each condition within each region
        for i, condition in enumerate(conditions_order):
            means = []
            sems = []
            colors = []
            for region in regions_order:
                row = agg_data[(agg_data['Region'] == region) & (agg_data['Condition'] == condition.lower())]
                if not row.empty:
                    means.append(row['Mean_Power_Change'].values[0])
                    sems.append(row['SEM_Power_Change'].values[0])
                    colors.append(color_mapping.get((region, condition), 'grey'))
                else:
                    means.append(0)
                    sems.append(0)
                    colors.append('grey')
            plt.bar(x + offsets[i], means, width/num_conditions, yerr=sems, label=condition,
                    color=colors, capsize=5, edgecolor='black')

        # Overlay individual session points and connect them with lines
        for sessID in df_power_change['sessID'].unique():
            sess_data = df_power_change[df_power_change['sessID'] == sessID]
            #print(sess_data)
            for region in regions_order:
                region_data = sess_data[sess_data['Region'] == region]
                if len(region_data) == 2:
                    # Extract power changes for structured and random
                    power_structured = region_data[region_data['Condition'] == 'structured']['Mean_Power_Change'].values[0]
                    power_random = region_data[region_data['Condition'] == 'random']['Mean_Power_Change'].values[0]

                    # Calculate x positions for structured and random bars
                    structured_idx = conditions_order.index('structured')
                    random_idx = conditions_order.index('random')
                    x_structured = x[regions_order.index(region)] + offsets[structured_idx]
                    x_random = x[regions_order.index(region)] + offsets[random_idx]

                    # Determine the color based on the direction of change
                    lineColor = 'blue' if power_structured > power_random else 'red'
                    plt.plot([x_structured, x_random], [power_structured, power_random],
                             color=lineColor, alpha=0.5, linewidth=2)

        # Overlay individual session points as slightly transparent black dots
        for idx, row in df_power_change.iterrows():
            region = row['Region']
            condition = row['Condition']#.capitalize()
            mean = row['Mean_Power_Change']
            sessID = row['sessID']
            # Determine the x position based on condition and region
            if condition not in conditions_order:
                continue  # Skip if condition is not recognized
            condition_idx = conditions_order.index(condition)
            region_idx = regions_order.index(region)
            x_pos = x[region_idx] + offsets[condition_idx]
            plt.scatter(x_pos, mean, color='black', alpha=0.6, s=30)
       
        plt.axhline(0,linestyle='--',color='k')
        plt.xticks(x, regions_order)
        plt.ylabel('Change in Power (dB change from baseline)')
        plt.title('Mean ± SEM of Band Power Change by Condition and Region')

        # ================== Paired Wilcoxon Tests and Annotations ================== #

        # Define a function to perform Wilcoxon test and return W and p-value
        def perform_wilcoxon(region_df):
            # Pivot the DataFrame to have conditions as columns
            pivot_df = region_df.pivot(index='sessID', columns='Condition', values='Mean_Power_Change').dropna()
            if pivot_df.shape[0] < 1:
                return None, None  # Not enough data for the test
            # Perform paired Wilcoxon signed-rank test
            stat, p = wilcoxon(pivot_df['structured'], pivot_df['random'])
            return stat, p

        # Initialize a list to store test results
        test_results = []

        # Iterate over each region to perform the test
        for i, region in enumerate(regions_order):
            region_df = df_power_change[df_power_change['Region'] == region]
            W, p = perform_wilcoxon(region_df)
            if W is not None and p is not None:
                test_results.append({'Region': region, 'W': W, 'p': p})
            else:
                test_results.append({'Region': region, 'W': np.nan, 'p': np.nan})

        # Determine the maximum y-value for each region to place the annotations
        y_max = agg_data.groupby('Region')['Mean_Power_Change'].max() + \
                agg_data.groupby('Region')['SEM_Power_Change'].max() + 0.5  # Add some offset

        # Add annotations for each region
        for result in test_results:
            region = result['Region']
            W = result['W']
            p = result['p']
            if pd.isna(W) or pd.isna(p):
                annotation_text = 'n/a'
            else:
                # Format p-value
                if p < 0.001:
                    p_text = 'p < 0.001'
                else:
                    p_text = f'p = {p:.5f}'
                annotation_text = f'W = {W:.2f}, {p_text}'

            # Get x positions for the two conditions
            structured_x = x[regions_order.index(region)] + offsets[conditions_order.index('structured')]
            random_x = x[regions_order.index(region)] + offsets[conditions_order.index('random')]

            # Get the maximum y-value in this region for annotation placement
            #current_y_max = y_max[region]
            current_y_max = 0.001 #y_max[region]

            # Define the y position for the annotation
            y, h, col = current_y_max, 0.01, 'k'
            #plt.plot([structured_x, structured_x, random_x, random_x], [y, y+h, y+h, y], lw=1.5, c=col)
            plt.text((structured_x + random_x) * .5,0.0048 , annotation_text,
                     ha='center', va='bottom', color='black', fontsize=12)

        # ================== End of Paired Wilcoxon Tests and Annotations ================== #

        plt.tight_layout()
        aggregated_filename = 'Aggregated_BandPowerChange_BarPlot.pdf'
        aggregated_save_path = os.path.join(save_dir, aggregated_filename)
        plt.savefig(aggregated_save_path)
        plt.close()

        print(f"Aggregated bar plot saved as '{aggregated_save_path}'")
    else:
        print("No power change data collected. Aggregated bar plot was not created.")


# Butterworth Filter Functions
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Gaussian Kernel for Moving Average Filter
def gaussian_kernel(size, sigma=1.0):
    """Creates a 2D Gaussian kernel with given size and standard deviation."""
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

def apply_moving_average_filter(spectrogram):
    """
    Apply a 2D Gaussian moving average filter to the spectrogram.

    Parameters:
    - spectrogram (2D array): The spectrogram data.

    Returns:
    - filtered_spectrogram (2D array): The filtered spectrogram.
    """
    kernel = gaussian_kernel(3, 1.0)  # 3x3 Gaussian kernel
    return convolve2d(spectrogram, kernel, mode='same', boundary='fill', fillvalue=0)

def compute_event_triggered_spectrogram(lfp_data, event_time, sampling_rate, pre_event_time=1, post_event_time=1, buffer_time=1, baseline_time_b4Zero_start=1, baseline_time_b4Zero_end=0.5, nperseg=800, normalize_each_spec=True):
    """
    Compute the spectrogram of LFP data around a specific event time, normalized by the baseline period.

    Parameters:
    - lfp_data (array): The LFP data.
    - event_time (float): Time of the event in seconds.
    - sampling_rate (float): Sampling rate of the LFP data.
    - pre_event_time (float): Time before the event to include in spectrogram.
    - post_event_time (float): Time after the event to include in spectrogram.
    - buffer_time (float): Additional time before and after for buffer.
    - baseline_time_b4Zero_start (float): Start time of baseline period before event.
    - baseline_time_b4Zero_end (float): End time of baseline period before event.
    - nperseg (int): Length of each segment for computing the spectrogram.
    - normalize_each_spec (bool): Whether to normalize each spectrogram.

    Returns:
    - f_filtered (array): Filtered frequency bins.
    - t_filtered (array): Filtered time bins.
    - f_edges (array): Frequency edges for plotting.
    - t_edges (array): Time edges for plotting.
    - Sxx_db_filtered (array): Spectrogram in decibels relative to baseline.
    """
    nperseg = int(nperseg)

    event_idx = int(event_time * sampling_rate)
    pre_event_samples = int(pre_event_time * sampling_rate)
    post_event_samples = int(post_event_time * sampling_rate)
    buffer_samples = int(buffer_time * sampling_rate)

    start_idx = max(event_idx - pre_event_samples - buffer_samples, 0)
    end_idx = min(event_idx + post_event_samples + buffer_samples, len(lfp_data))

    lfp_segment_with_buffer = lfp_data[start_idx:end_idx]

    # Compute the spectrogram with buffer
    f, t_buffered, Sxx_buffered = spectrogram(lfp_segment_with_buffer, fs=sampling_rate, nperseg=nperseg)

    Sxx_buffered = apply_moving_average_filter(Sxx_buffered)

    # Correct the calculation for the start and end time bins to exclude the buffer
    total_duration_with_buffer = 2 * buffer_time + pre_event_time + post_event_time
    time_step = total_duration_with_buffer / len(t_buffered)
    start_time_bin = int(buffer_time / time_step)
    end_time_bin = int((buffer_time + pre_event_time + post_event_time) / time_step)

    # Compute the baseline spectrum
    baseline_start_time = pre_event_time + buffer_time - baseline_time_b4Zero_start  # Start of the baseline period
    baseline_end_time = pre_event_time + buffer_time - baseline_time_b4Zero_end      # End of the baseline period

    baseline_start_idx = int(baseline_start_time / time_step)
    baseline_end_idx = int(baseline_end_time / time_step)

    # Handle cases where baseline_start_idx or baseline_end_idx might be out of bounds
    baseline_start_idx = max(baseline_start_idx, 0)
    baseline_end_idx = min(baseline_end_idx, Sxx_buffered.shape[1])

    if baseline_end_idx <= baseline_start_idx:
        print(f"Invalid baseline indices for event_time: {event_time}")
        return None, None, None, None, None

    baseline_mean = np.mean(Sxx_buffered[:, baseline_start_idx:baseline_end_idx], axis=1, keepdims=True)

    # Normalize the entire spectrogram by the baseline
    Sxx_normalized = Sxx_buffered / baseline_mean

    # Convert to decibels
    Sxx_db = 10 * np.log10(Sxx_normalized)

    # Exclude the buffer from the spectrogram
    t = t_buffered[start_time_bin:end_time_bin+2] - (pre_event_time + buffer_time)
    Sxx_db = Sxx_db[:, start_time_bin:end_time_bin+2]

    # Filter the frequencies between 0.5 and 40 Hz
    freq_mask = (f >= 0.5) & (f <= 40)
    t_mask = (t >= -1) & (t <= 1)
    f_filtered = f[freq_mask]
    t_filtered = t[t_mask]

    Sxx_db_filtered = Sxx_db[freq_mask, :]
    Sxx_db_filtered = Sxx_db_filtered[:, t_mask]

    if normalize_each_spec:
        # Normalize spectrogram by sum of absolute values
        Sxx_db_filtered = Sxx_db_filtered / np.sum(np.abs(Sxx_db_filtered))

    # Define frequency and time edges for plotting
    t_edges = np.concatenate([t_filtered - np.diff(t_filtered)[0]/2, [t_filtered[-1] + np.diff(t_filtered)[0]/2]])
    f_edges = np.concatenate([f_filtered - np.diff(f_filtered)[0]/2, [f_filtered[-1] + np.diff(f_filtered)[0]/2]])

    return f_filtered, t_filtered, f_edges, t_edges, Sxx_db_filtered

def find_global_min_max(spectrogram_data):
    """
    Find the global minimum and maximum across all spectrograms.

    Parameters:
    - spectrogram_data (dict): Dictionary of spectrogram data.

    Returns:
    - global_min (float): Minimum value across all spectrograms.
    - global_max (float): Maximum value across all spectrograms.
    """
    all_values = []
    for data in spectrogram_data.values():
        if data is not None:
            all_values.extend(data.ravel())
    if all_values:
        global_min = min(all_values)
        global_max = max(all_values)
    else:
        global_min, global_max = 0, 1  # Default values
    return global_min, global_max

def edgesToBins(edges):
    bin_width = edges[1] - edges[0]
    bins = edges[:-1] + (bin_width / 2.0)
    return bins

def plot_spectrograms_with_diff(spectrogram_data, f_edges, t_edges, sessID, REGION_TYPE, save_dir):
    """
    Plot spectrograms and their differences for a given session and region.

    Parameters:
    - spectrogram_data (dict): Dictionary of spectrogram data for each condition.
    - f_edges (array): Frequency edges for plotting.
    - t_edges (array): Time edges for plotting.
    - sessID (str): Session ID.
    - REGION_TYPE (str): Type of region ('HPC' or 'AIC').
    - save_dir (str): Directory where the plot PDF will be saved.

    Returns:
    - Saves a spectrogram plot with differences.
    """
    f = edgesToBins(f_edges)
    t = edgesToBins(t_edges)

    # Determine the global min and max across all spectrograms for symmetrical color limits around 0 dB
    print('Plotting spectrograms with differences...')
    global_min, global_max = find_global_min_max(spectrogram_data)
    vmax = global_max
    vmin = global_min

    # Optionally, set symmetrical limits if desired
    vmax = max(abs(global_min), abs(global_max))
    vmin = -vmax

    # Prepare the figure with subplots for each condition and their differences
    fig, axs = plt.subplots(3, 2, figsize=(18, 12), sharex=False, sharey=False)

    conditions = ['structured_early', 'structured_late', 'random_early', 'random_late']
    for i, condition in enumerate(conditions):
        row = i // 2
        col = i % 2
        data = spectrogram_data.get(condition, None)
        ax = axs[row, col]
        if data is not None:
            # Debugging: Print shapes
            print(f"Plotting condition: {condition}, Data shape: {data.shape}, f_edges: {f_edges.shape}, t_edges: {t_edges.shape}")
            pcm = ax.pcolormesh(t_edges, f_edges, data, cmap='coolwarm', vmin=vmin, vmax=vmax, shading='auto')  # Changed shading here
            fig.colorbar(pcm, ax=ax, pad=0.01).set_label('Power (dB change from baseline, norm)')
        else:
            ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes, ha='center', va='center')
            ax.axis('off')

        ax.set_title(condition.replace('_', ' ').capitalize())
        ax.axvline(x=0, color='black', linestyle='--', linewidth=2)
        ax.set_xlabel('Time from display (s)')
        ax.set_ylabel('Frequency (Hz)')

    # Calculate and plot the difference heatmaps
    # Difference between late and early structured
    structured_diff = spectrogram_data.get('structured_late', None) - spectrogram_data.get('structured_early', None) if spectrogram_data.get('structured_late') is not None and spectrogram_data.get('structured_early') is not None else None
    if structured_diff is not None:
        ax = axs[2, 0]
        print(f"Plotting structured_diff, Data shape: {structured_diff.shape}, f_edges: {f_edges.shape}, t_edges: {t_edges.shape}")
        pcm = ax.pcolormesh(t_edges, f_edges, structured_diff, cmap='coolwarm', vmin=vmin, vmax=vmax, shading='auto')  # Changed shading here
        ax.set_title('structured Late - structured Early')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=2)
        ax.set_xlabel('Time from display (s)')
        ax.set_ylabel('Frequency (Hz)')
        fig.colorbar(pcm, ax=ax, pad=0.01).set_label('Power Difference (dB change)')

    # Difference between late and early random
    random_diff = spectrogram_data.get('random_late', None) - spectrogram_data.get('random_early', None) if spectrogram_data.get('random_late') is not None and spectrogram_data.get('random_early') is not None else None
    if random_diff is not None:
        ax = axs[2, 1]
        print(f"Plotting random_diff, Data shape: {random_diff.shape}, f_edges: {f_edges.shape}, t_edges: {t_edges.shape}")
        pcm = ax.pcolormesh(t_edges, f_edges, random_diff, cmap='coolwarm', vmin=vmin, vmax=vmax, shading='auto')  # Changed shading here
        ax.set_title('random Late - random Early')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=2)
        ax.set_xlabel('Time from display (s)')
        ax.set_ylabel('Frequency (Hz)')
        fig.colorbar(pcm, ax=ax, pad=0.01).set_label('Power Difference (dB change)')

    plt.suptitle(f'Session: {sessID}, Region: {REGION_TYPE} - Scene-triggered Spectrograms', fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    structured_diffThetaPower = get_power_change_from_spectograms(structured_diff, f, t)
    random_diffThetaPower = get_power_change_from_spectograms(random_diff, f, t)
    # Save the figure with a filename based on the session ID and region
    filename = f"SpectrogramChangePerCondition_{sessID}_{REGION_TYPE}.pdf"
    save_path = os.path.join(save_dir, filename)  # Define the full save path
    plt.savefig((save_path), format="pdf", bbox_inches='tight')
    plt.close()
    print(f'Plotted spectrograms with differences, saved at {save_path}')
    return save_path, structured_diffThetaPower, random_diffThetaPower

def get_power_change_from_spectograms(Sxx, f, t): 
    freq_indices = np.logical_and(f >= FREQ_BAND[0], f <= FREQ_BAND[1])

    if Sxx is None:
        return None
    power_in_band = Sxx[freq_indices, :][:, np.logical_and(t >= after_interval[0], t <= after_interval[1])].mean()
    return power_in_band

def classify_events(event_times, sessID, early_threshold=EARLY_THRESHOLD, late_threshold=LATE_THRESHOLD):
    """
    Classify events into conditions and timings based on thresholds.

    Parameters:
    - event_times (array): Array of event times in seconds.
    - sessID (str): Session ID.
    - early_threshold (int): Number of early events to classify.
    - late_threshold (int): Number of late events to classify.

    Returns:
    - classified_events (list of tuples): Each tuple contains (event_time, condition, timing).
    """
    conditions = ti.get_condition_names_for_sessID(sessID)

    classified_events = []
    for i, event_time in enumerate(event_times):
        # Determine the condition
        if i < GROUP_SIZE:
            condition = conditions[0]
        else:
            condition = conditions[1] if len(conditions) > 1 else conditions[0]

        # Determine if the event is early or late within its group
        index_in_group = i % GROUP_SIZE  # Index within the group
        if index_in_group < early_threshold:
            timing = "early"
        elif index_in_group >= GROUP_SIZE - late_threshold:
            timing = "late"
        else:
            timing = "middle"
            continue  # Skip middle events

        classified_events.append((event_time, condition, timing))

    return classified_events

def average_spectrograms_across_channels(channel_range, all_spectrograms):
    """
    Average spectrograms across multiple channels.

    Parameters:
    - channel_range (list): List of channel numbers.
    - all_spectrograms (dict): Dictionary of spectrogram data for each condition.

    Returns:
    - avg_spectrograms (dict): Averaged spectrograms for each condition.
    """
    # Initialize dictionary to store the averaged spectrograms
    print('Averaging spectrograms across channels...')
    avg_spectrograms = {cat: [] for cat in all_spectrograms.keys()}

    # Loop over each channel and accumulate spectrogram data
    for channel in channel_range:
        for cat, data_list in all_spectrograms.items():
            if data_list:  # Check if there's any spectrogram data for this category
                # Stack and average the data across the current channel
                stacked_data = np.stack(data_list)
                avg_spectrograms[cat].append(np.mean(stacked_data, axis=0))
            else:
                print(f"No data for category {cat} in channel {channel}")

    # Average across the accumulated data for each category
    for cat in avg_spectrograms:
        if avg_spectrograms[cat]:  # Check if there's any accumulated data
            avg_spectrograms[cat] = np.mean(np.stack(avg_spectrograms[cat]), axis=0)
        else:
            avg_spectrograms[cat] = None
            print(f"No data accumulated for category {cat} across channels")

    print('Averaged spectrograms across channels.')
    return avg_spectrograms

# ================== Main Code ================== #

def main():
    # Define the base directory for saving plots
    base_save_dir = 'fig6plots'

    # Define the subdirectory for aggregated plots
    aggregated_save_dir = os.path.join(base_save_dir, 'aggregated')
    os.makedirs(aggregated_save_dir, exist_ok=True)  # Create the subdirectory if it doesn't exist

    # Define the path for the processed data pickle file
    processed_data_path = os.path.join(aggregated_save_dir, 'power_change_data.pkl')

    if os.path.exists(processed_data_path):
        print("Processed data found. Loading from pickle file...")
        df_power_change = pd.read_pickle(processed_data_path)
    else:
        print("No processed data found. Processing raw data...")

        # Retrieve session names
        sessNames = dpd.getSessNames()

        # Initialize dictionaries to accumulate spectrograms per region and condition
        accumulated_spectrograms = {
            'HPC': {"structured_early": [], "structured_late": [], "random_early": [], "random_late": []},
            'AIC': {"structured_early": [], "structured_late": [], "random_early": [], "random_late": []}
        }

        # Initialize a list to collect power change data for the aggregated bar plot
        power_change_data = []

        # Define the regions to loop through
        regions = ['HPC', 'AIC']

        for REGION_TYPE in regions:
            print(f"\nProcessing region: {REGION_TYPE}")

            # Define the subdirectory for the current region
            region_save_dir = os.path.join(base_save_dir, REGION_TYPE)
            os.makedirs(region_save_dir, exist_ok=True)  # Create the subdirectory if it doesn't exist

            for sessID in sessNames:
                # Get channel range based on region
                if REGION_TYPE.lower() == 'hpc':
                    channel_range = ril.get_hpc_lfp_channels(sessID, brainSideStr)
                else:
                    channel_range = ril.get_aic_LFPchannels(sessID)

                if len(channel_range) == 0:
                    print(f"No channels found for session: {sessID}, region: {REGION_TYPE}")
                    continue

                # Initialize current session spectrograms
                all_spectrograms_curr_sess = {"structured_early": [], "structured_late": [], "random_early": [], "random_late": []}

                # Dictionary to store LFP data per channel (optional, based on your requirements)
                lfp_data_by_channel = {}

                # Loop over each channel and accumulate spectrogram data
                for channel in channel_range:
                    # Load downsampled LFP and event times
                    downsampled_lfp, event_times, originalFs, dsFact = ll.load_downsampled_lfp_and_events_with_cache(sessID, int(channel),dataDir=processed_data_dir)
                    lfp_data_by_channel[channel] = downsampled_lfp

                    # Handle specific session IDs if needed
                    if 'pt1' in sessID:
                        event_times = event_times[0:150]  # Adjust based on your data structure

                    # Classify events
                    classified_events = classify_events(event_times, sessID, early_threshold=EARLY_THRESHOLD, late_threshold=LATE_THRESHOLD)

                    # Compute spectrograms for each classified event
                    for event_time, condition, timing in classified_events:
                        f, t, f_edges, t_edges, Sxx = compute_event_triggered_spectrogram(
                            downsampled_lfp,
                            event_time,
                            originalFs / dsFact,
                            pre_event_time=1,
                            post_event_time=1,
                            buffer_time=1,
                            baseline_time_b4Zero_start=1,
                            baseline_time_b4Zero_end=0.5,
                            nperseg=800,
                            normalize_each_spec=normalize_each_spec
                        )

                        if Sxx is not None and Sxx.size > 0:
                            category = f"{condition}_{timing}"
                            all_spectrograms_curr_sess[category].append(Sxx)
                            accumulated_spectrograms[REGION_TYPE][category].append(Sxx)
                        else:
                            print(f"No data for event_time: {event_time}, condition: {condition}, timing: {timing}")

                # After processing all channels, average spectrograms across channels for the current session
                avg_spectrograms_across_channels = average_spectrograms_across_channels(channel_range, all_spectrograms_curr_sess)

                # Assuming t and f have been computed from the last channel's spectrogram data
                # Define freq_indices here for use in the main loop
                freq_indices = np.logical_and(f >= FREQ_BAND[0], f <= FREQ_BAND[1])

                # Plotting individual session spectrograms with differences
                _, structured_diffThetaPower, random_diffThetaPower = plot_spectrograms_with_diff(
                    avg_spectrograms_across_channels, f_edges, t_edges, sessID, REGION_TYPE, region_save_dir
                )

                # Corrected condition labels
                if structured_diffThetaPower is not None:
                    power_change_data.append({
                        'Region': REGION_TYPE,
                        'Condition': 'structured',  # Corrected from 'random'
                        'Mean_Power_Change': structured_diffThetaPower,
                        'sessID': sessID
                    })

                if random_diffThetaPower is not None:
                    power_change_data.append({
                        'Region': REGION_TYPE,
                        'Condition': 'random',  # Corrected from 'structured'
                        'Mean_Power_Change': random_diffThetaPower,
                        'sessID': sessID
                    })

        # After processing all regions and sessions, create the DataFrame
        df_power_change = pd.DataFrame(power_change_data)

        # Save the processed data to a pickle file
        df_power_change.to_pickle(processed_data_path)
        print(f"Processed data saved to {processed_data_path}")

    # Plot the aggregated bar plot with statistical annotations
    plot_aggregated_bar_plot(df_power_change, base_save_dir)
    # ================== Aggregated Bar Plot ================== #


if __name__ == "__main__":
    main()

