from utils import pdfTextSet
import pdb
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch #, gaussian
from scipy.ndimage import gaussian_filter1d  # For additional smoothing
from scipy import stats  # For SEM calculations
from utils import taskInfoFunctions as tif  # Ensure this module is available and correctly implemented
from utils import local_paths

# ---------------------
# Parameters
# ---------------------
processed_data_dir=local_paths.get_processed_data_dir()
#processed_data_dir = "E:\\Rest\\OneDrive\\OneDrive - Yale University\\HippocampalSequences_epsFigures\\manuscript_analysis\\downsampledLFPwithEventTimes"


# Flags and Parameters
RESTRICT_AROUND_SCENE_FLAG = True
JUST_FORWARDS_FLAG = False

TIME_AROUND_SCENE = 10  # Time around the initial preferred scene number to consider (in seconds)
PREF_SCENE_TIME_OFFSET = 0
SPECTRUM_TIME_WIND_SEC = 0.9

MIN_MEAN_FIRING_RATE = 0
MIN_PERC = 1
MAX_PERC = 99
MIN_PERC = 0.5
MAX_PERC = 99.5
STANDARD_YLIM = True
YMIN = -2
YMAX = 2.2
YMAX = 2.6
XMIN = 0
XMAX=70

PHASE_DURATION = 100  # Duration for early and late phases in seconds
MAX_ISI = 0.2  # Maximum ISI to consider for the plots in seconds
NUM_BINS = 1000  # Number of bins for KDE
FS = 1000  # Sampling frequency for the spike train power spectrum (in Hz)
MAX_FREQ = 70  # Maximum frequency to consider for the power spectrum (in Hz)
MIN_FREQ = 1  # Minimum frequency to consider for the power spectrum (in Hz)
BASELINE_SUBTRACTION = False  # Flag to control baseline subtraction
SMOOTHING_WINDOW_INITIAL = 3  # Initial smoothing window in Hz
SMOOTHING_WINDOW_INITIAL = 2  # Initial smoothing window in Hz
SMOOTHING_WINDOW_INITIAL = 1  # Initial smoothing window in Hz
SMOOTHING_WINDOW_INITIAL = 0.1  # Initial smoothing window in Hz
#SMOOTHING_WINDOW_MEAN = 0.75  # Additional smoothing window for mean and percentiles in Hz
SMOOTHING_WINDOW_MEAN = 1  # Additional smoothing window for mean and percentiles in Hz
SMOOTHING_WINDOW_MEAN = 0.1  # Additional smoothing window for mean and percentiles in Hz
JITTER_WINDOW = 0.5  # Jitter window in seconds
NUM_SHUFFLES = 500  # Number of shuffles for null power spectrum calculation
NUM_POOLED_SHUFFLES = 50000  # Number of shuffles for pooled permutation testing

y_limits=[YMIN,YMAX]
x_limits=[XMIN,XMAX]

# Define colors based on region
REGION_COLORS = {
    'HPC': 'red',
    'non-HPC': 'brown'
}

# ---------------------
# Helper Function Definitions
# ---------------------

def convolve_spike_train(spike_train, kernel_width_seconds=0.05, fs=1000):
    """
    Convolve spike train with a Gaussian kernel.

    Parameters:
        spike_train (np.array): Binary spike train.
        kernel_width_seconds (float): Width of the Gaussian kernel in seconds.
        fs (int): Sampling frequency in Hz.

    Returns:
        np.array: Convolved spike train.
    """
    kernel_width_samples = int(kernel_width_seconds * fs)
    if kernel_width_samples < 1:
        kernel_width_samples = 1
    kernel = gaussian(kernel_width_samples, std=kernel_width_samples / 6)  # 6 sigma to cover 99.7% of the area
    kernel /= np.sum(kernel)  # Normalize kernel to preserve spike train total sum
    return np.convolve(spike_train, kernel, mode='same')


def calculate_power_spectrum(spike_times, fs=FS, min_freq=MIN_FREQ, max_freq=MAX_FREQ, kernel_width_seconds=0.05):
    """
    Calculate the power spectrum of a spike train using Welch's method.

    Parameters:
        spike_times (np.array): Spike times in seconds.
        fs (int): Sampling frequency in Hz.
        min_freq (float): Minimum frequency to consider.
        max_freq (float): Maximum frequency to consider.
        kernel_width_seconds (float): Width of the Gaussian kernel for convolution.

    Returns:
        tuple: Frequencies and power spectrum.
    """
    spike_train = np.zeros(int(np.ceil(spike_times[-1] * fs)) + 1)
    spike_indices = (spike_times * fs).astype(int)
    spike_indices = np.clip(spike_indices, 0, len(spike_train) - 1)  # Ensure indices are within bounds
    spike_train[spike_indices] = 1
    # Uncomment the next line to apply convolution if desired
    # spike_train = convolve_spike_train(spike_train, kernel_width_seconds, fs)
    nperseg = int(SPECTRUM_TIME_WIND_SEC * fs)
    freqs, power = welch(spike_train, fs=fs, nperseg=nperseg)
    valid_indices = (freqs >= min_freq) & (freqs <= max_freq)
    freqs = freqs[valid_indices]
    power = power[valid_indices]
    power /= np.sum(power)  # Normalize to get spectral density
    return freqs, power


def calculate_null_power_spectrum_with_jitter(phase_name, spike_times, unit_id, num_shuffles=NUM_SHUFFLES, fs=FS, min_freq=MIN_FREQ, max_freq=MAX_FREQ, jitter_window=JITTER_WINDOW):
    """
    Calculate null power spectra by introducing random jitter to spike times.

    Parameters:
        phase_name (str): 'early' or 'late'.
        spike_times (np.array): Spike times in seconds.
        unit_id (str): Unique identifier for the unit.
        num_shuffles (int): Number of shuffles.
        fs (int): Sampling frequency in Hz.
        min_freq (float): Minimum frequency.
        max_freq (float): Maximum frequency.
        jitter_window (float): Jitter window in seconds.

    Returns:
        np.array: Null power spectra.
    """
    # Generate the filename based on parameters
    filename = f"null_powers_{phase_name}_{unit_id}_{len(spike_times)}spikes_{num_shuffles}shuffles_{jitter_window}s_jitter_{fs}Hz_fs_FreqDomain{min_freq}_{max_freq}Hz_{SPECTRUM_TIME_WIND_SEC}_sec_spectrum_time_window.npy"
    filepath = os.path.join(processed_data_dir,'null_power_cache', filename)

    print("Saving to:", os.path.abspath(filepath))

    #filepath = 'E:/Rest/Lab/Members/Tibin/manuscript_analysis/downsampledLFPwithEventTimes/null_power_cache/null_powers_early_HPC_structured_pt5_sess1_ch248_unit4Only10secAroundPreferredScene_Offset0sec_234spikes_500shuffles_0.5s_jitter_1000Hz_fs_FreqDomain1_70Hz_0.9_sec_spectrum_time_window.npy'
    # Check if the file already exists
    if os.path.exists(filepath):
        print(f"Loading null power spectra from {filepath}")
        null_powers = np.load(filepath)
    else:
        print(f"Calculating null power spectra and saving to {filepath}")
        null_powers = []
        for shuffle_idx in range(num_shuffles):
            jittered_spike_times = spike_times + np.random.uniform(-jitter_window, jitter_window, size=len(spike_times))
            jittered_spike_times = np.clip(jittered_spike_times, 0, spike_times[-1])  # Ensure times stay within valid range
            jittered_spike_times.sort()  # Sort the jittered spike times
            _, null_power = calculate_power_spectrum(np.array(jittered_spike_times), fs=fs, min_freq=min_freq, max_freq=max_freq)
            null_powers.append(null_power)
        null_powers = np.array(null_powers)

        # Save the null powers to a file
        os.makedirs('null_power_cache', exist_ok=True)
        np.save(filepath, null_powers)

    return null_powers


def smooth_power_spectrum(power, window_size=SMOOTHING_WINDOW_INITIAL, fs=FS, freqs=None):
    """
    Smooth the power spectrum using a moving average or Gaussian filter.

    Parameters:
        power (np.array): Power spectrum.
        window_size (float): Window size in Hz for smoothing.
        fs (int): Sampling frequency in Hz.
        freqs (np.array): Frequencies corresponding to the power spectrum.

    Returns:
        np.array: Smoothed power spectrum.
    """
    if freqs is None or len(freqs) < 2:
        return power
    freq_res = freqs[1] - freqs[0]
    window_size_samples = int(window_size / freq_res)
    if window_size_samples < 1:
        window_size_samples = 1
    smoothed_power = gaussian_filter1d(power, sigma=window_size_samples / 3)  # Approx. window_size_samples / 3 for Gaussian
    return smoothed_power


def get_scene_bounds_around_peak(scene_time_bounds, initial_peak_scene_num, time_around_scene, just_forwards=False, offset=0):
    """
    Get scene time bounds around the initial peak scene number.

    Parameters:
        scene_time_bounds (dict): Dictionary mapping scene numbers to time intervals.
        initial_peak_scene_num (int): Initial peak scene number.
        time_around_scene (float): Time around the scene to consider (in seconds).
        just_forwards (bool): If True, consider only forward times.
        offset (float): Time offset in seconds.

    Returns:
        list: List of (start, end) tuples representing scene bounds.
    """
    scene_intervals = scene_time_bounds.get(initial_peak_scene_num, [])
    new_intervals = []
    for start, end in scene_intervals:
        if just_forwards:
            new_start = max(0, start) + offset
            new_end = end + time_around_scene + offset
        else:
            new_start = max(0, start - time_around_scene) + offset
            new_end = end + time_around_scene + offset
        new_intervals.append((new_start, new_end))
    return new_intervals


# ---------------------
# Plotting Function Definitions
# ---------------------

def plot_pooled_surrogate_power_spectrum(
    observed_diff,
    surrogate_diffs_pooled,
    freqs,
    region,
    output_dir,
    min_freq,
    max_freq,
    y_limits,
    smoothing_window_mean
):
    """
    Plot the observed double difference against the surrogate null distribution.

    Parameters:
        observed_diff (np.array): Observed double difference (structured_change - random_change).
        surrogate_diffs_pooled (np.array): Surrogate differences from permutation testing.
        freqs (np.array): Frequencies corresponding to the power spectrum.
        region (str): Region name ('HPC' or 'non-HPC').
        output_dir (str): Directory to save the plot.
        min_freq (float): Minimum frequency to display.
        max_freq (float): Maximum frequency to display.
        y_limits (tuple): Tuple of (ymin, ymax) for y-axis limits.
        smoothing_window_mean (float): Smoothing window size in Hz for mean and percentiles.
    """
    plt.figure(figsize=(6, 6))

    # Compute percentiles from surrogate differences
    percentile_5 = np.percentile(surrogate_diffs_pooled, MIN_PERC, axis=0)
    percentile_95 = np.percentile(surrogate_diffs_pooled, MAX_PERC, axis=0)

    # Smooth the observed difference and percentiles
    observed_diff_smoothed = gaussian_filter1d(observed_diff, sigma=smoothing_window_mean / (freqs[1] - freqs[0]))
    percentile_5_smoothed = gaussian_filter1d(percentile_5, sigma=smoothing_window_mean / (freqs[1] - freqs[0]))
    percentile_95_smoothed = gaussian_filter1d(percentile_95, sigma=smoothing_window_mean / (freqs[1] - freqs[0]))

    # Plot surrogate percentile thresholds
    plt.fill_between(freqs, percentile_5_smoothed, percentile_95_smoothed, color=REGION_COLORS[region], alpha=0.3, label='0.5th-99.5th Percentile Surrogate')

    # Plot observed double difference
    plt.plot(freqs, observed_diff_smoothed, label='Observed Structured - Random', color=REGION_COLORS[region])

    # Highlight significant regions
    significant = (observed_diff_smoothed < percentile_5_smoothed) | (observed_diff_smoothed > percentile_95_smoothed)
    #significant = (observed_diff_smoothed > percentile_95_smoothed)


    consecutive_sig = np.zeros_like(significant)
    for i in range(1,len(significant)-1):
        if (significant[i-1] and significant[i] and significant[i+1]):
            consecutive_sig[i-1]=True
            consecutive_sig[i]=True
            consecutive_sig[i+1]=True

    significant=consecutive_sig
    plt.scatter(freqs[significant], observed_diff_smoothed[significant], color=REGION_COLORS[region], s=50, label='Consecutively significant frequencies')
    # Calculate p-values for significant frequencies
    p_values = np.ones_like(freqs)
    peakFreqs = np.zeros_like(freqs)
    for i, freq in enumerate(freqs):
        if significant[i]:
            if observed_diff_smoothed[i] > percentile_95_smoothed[i]:
                p = np.sum(surrogate_diffs_pooled[:, i] >= observed_diff_smoothed[i]) / NUM_POOLED_SHUFFLES
                peakFreqs[i]=freq
            else:
                p = np.sum(surrogate_diffs_pooled[:, i] <= observed_diff_smoothed[i]) / NUM_POOLED_SHUFFLES
                peakFreqs[i]=freq
            p_values[i] = p

    peak_pVal=np.min(p_values)
    peakFreq=peakFreqs[np.argmin(p_values)]
    if peak_pVal==1:
        peak_pVal=np.nan
        pValStr=f'no significant change in spike train spectral power'
    else:
        pValStr=f'Peak difference at {peakFreq:.1f} Hz (p={peak_pVal:.5f})'

    #if region.lower() != 'hpc':
    #    pdb.set_trace()
    # Configure plot aesthetics
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Difference in Spectral Density (Spike jitter Z-score)')
    plt.title(f'Double Difference Spike Train Spectrum - {region}\n(Structured Change) - (Random Change)\n {pValStr}')
    plt.xlim([XMIN,XMAX])
    if STANDARD_YLIM:
        plt.ylim(y_limits)
    plt.axhline(0, color='black', linestyle='--')
    plt.legend(fontsize='small', loc='upper right', framealpha=0.5)
    plt.tight_layout()

    # Save the plot
    plot_filename = os.path.join(output_dir, f"double_difference_power_spectrum_pooled_surrogate_{region}.pdf")
    plt.savefig(plot_filename)
    plt.close()


def plot_peak_frequency_distributions(
    peak_changes_structured,
    peak_changes_random,
    region,
    output_dir,
    num_pooled_shuffles
):
    """
    Plot distributions of peak frequency changes using violin plots with mean ± SEM and surrogate thresholds.

    Parameters:
        peak_changes_structured (list): List of peak frequency changes (late - early) for structured condition.
        peak_changes_random (list): List of peak frequency changes (late - early) for random condition.
        region (str): Region name ('HPC' or 'non-HPC').
        output_dir (str): Directory to save the plot.
        num_pooled_shuffles (int): Number of shuffles for surrogate differences.
    """
    plt.figure(figsize=(10, 6))
    data = [
        peak_changes_structured,
        peak_changes_random
    ]
    labels = [
        'Structured (Late - Early)',
        'Random (Late - Early)'
    ]

    # Create violin plot
    parts = plt.violinplot(data, showmeans=False, showmedians=True, showextrema=True)

    # Define color based on region
    color = REGION_COLORS.get(region, 'blue')  # Default to blue if region not found

    # Customize violin plots
    for pc in parts['bodies']:
        pc.set_facecolor(color)
        pc.set_edgecolor(color)
        pc.set_alpha(0.5)

    # Calculate means and SEMs
    means = [np.nanmean(d) for d in data]
    sems = [stats.sem(d, nan_policy='omit') for d in data]
    positions = np.arange(1, len(data) + 1)

    # Calculate surrogate differences by shuffling pooled changes
    combined_changes = peak_changes_structured + peak_changes_random
    n_structured = len(peak_changes_structured)
    n_random = len(peak_changes_random)

    surrogate_diffs = []
    for _ in range(num_pooled_shuffles):
        shuffled = np.random.permutation(combined_changes)
        shuffled_structured = shuffled[:n_structured]
        shuffled_random = shuffled[n_structured:]
        surrogate_diff = np.mean(shuffled_structured) - np.mean(shuffled_random)
        surrogate_diffs.append(surrogate_diff)
    surrogate_diffs = np.array(surrogate_diffs)

    # Compute 5th and 95th percentiles
    percentile_5 = np.percentile(surrogate_diffs, MIN_PERC)
    percentile_95 = np.percentile(surrogate_diffs, MAX_PERC)

    # Plot mean ± SEM
    plt.errorbar(positions, means, yerr=sems, fmt='o', color='black', label='Mean ± SEM')

    # Plot surrogate thresholds
    plt.axhline(y=percentile_5, color='grey', linestyle='--', label='1st Percentile Surrogate')
    plt.axhline(y=percentile_95, color='grey', linestyle='-.', label='99th Percentile Surrogate')

    plt.xticks(positions, labels, rotation=45)
    plt.ylabel('Peak Frequency Change (Hz)')
    plt.title(f'Distribution of Peak Frequency Changes - {region}')
    plt.legend()
    plt.tight_layout()

    # Save the plot
    plot_filename = os.path.join(output_dir, f"peak_frequency_change_distribution_{region}.pdf")
    plt.savefig(plot_filename)
    plt.close()


def plot_double_difference(
    observed_diff,
    surrogate_diffs_pooled,
    freqs,
    region,
    output_dir,
    min_freq,
    max_freq,
    y_limits,
    smoothing_window_mean
):
    """
    Plot the observed double difference against the surrogate null distribution.

    Parameters:
        observed_diff (np.array): Observed double difference (structured_change - random_change).
        surrogate_diffs_pooled (np.array): Surrogate differences from permutation testing.
        freqs (np.array): Frequencies corresponding to the power spectrum.
        region (str): Region name ('HPC' or 'non-HPC').
        output_dir (str): Directory to save the plot.
        min_freq (float): Minimum frequency to display.
        max_freq (float): Maximum frequency to display.
        y_limits (tuple): Tuple of (ymin, ymax) for y-axis limits.
        smoothing_window_mean (float): Smoothing window size in Hz for mean and percentiles.
    """
    # This function is now integrated into 'plot_pooled_surrogate_power_spectrum'
    pass  # Placeholder as it's handled within the plotting functions


def plot_power_spectrum_early_late(
    df,
    region='HPC',
    phase_duration=PHASE_DURATION,
    fs=FS,
    min_freq=MIN_FREQ,
    max_freq=MAX_FREQ,
    baseline_subtraction=BASELINE_SUBTRACTION,
    smoothing_window=SMOOTHING_WINDOW_INITIAL,
    output_dir='spike_spectra_plots',
    restrict_to_scene=RESTRICT_AROUND_SCENE_FLAG,
    time_around_scene=TIME_AROUND_SCENE,
    just_forwards=JUST_FORWARDS_FLAG,
    jitter_window=JITTER_WINDOW,
    num_shuffles=NUM_SHUFFLES,
    num_pooled_shuffles=NUM_POOLED_SHUFFLES
):
    """
    Process the DataFrame and generate power spectrum plots for early and late phases using pooled surrogate sets.

    Parameters:
        df (pd.DataFrame): DataFrame containing unit condition pair information.
        region (str): Region to filter ('HPC' or 'non-HPC').
        phase_duration (float): Duration for early and late phases in seconds.
        fs (int): Sampling frequency in Hz.
        min_freq (float): Minimum frequency to consider.
        max_freq (float): Maximum frequency to consider.
        baseline_subtraction (bool): Flag to control baseline subtraction.
        smoothing_window (float): Smoothing window size in Hz.
        output_dir (str): Directory to save the plots.
        restrict_to_scene (bool): Flag to restrict analysis around scene times.
        time_around_scene (float): Time around the scene to consider (in seconds).
        just_forwards (bool): Flag to consider only forward times.
        jitter_window (float): Jitter window in seconds.
        num_shuffles (int): Number of shuffles for null power spectrum calculation.
        num_pooled_shuffles (int): Number of shuffles for pooled permutation testing.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Initialize lists for aggregating power spectra and percentiles
    early_random_list = []
    late_random_list = []
    early_structured_list = []
    late_structured_list = []

    # Lists to collect peak frequency changes
    peak_changes_structured = []  # (late - early) for structured
    peak_changes_random = []      # (late - early) for random

    # Iterate over each unit in the DataFrame
    for idx, row in df.iterrows():
        sessIDStr = row['sessIDStr']
        region_name = row['region_name']
        task_type = row['task_type']

        # Filter based on specified region
        if region != 'non-HPC':
            if region != region_name:
                continue
        else:
            if region_name == 'HPC':
                continue

        unit_id = f"{region}_{task_type}_{row['sessIDStr']}_{row['chNameStr']}_{row['unitNumStr']}"

        # Filter based on mean firing rate
        if row['mean_firing_rate'] < MIN_MEAN_FIRING_RATE:
            continue

        # Modify unit_id based on flags
        if restrict_to_scene:
            unit_id += f'Only{time_around_scene}secAroundPreferredScene_Offset{PREF_SCENE_TIME_OFFSET}sec'

        if just_forwards:
            unit_id += "JustForwardOfScene"

        condition_time_bounds = row['condition_time_bounds']
        start_time, end_time = condition_time_bounds

        # Define early and late phases
        early_phase_end = start_time + phase_duration
        late_phase_start = end_time - phase_duration

        # Restrict to scene time bounds if the flag is set
        if restrict_to_scene:
            initial_peak_scene_num = row['initial_peak_scene_num']
            scene_time_bounds = row['time_bounds_per_scene_num']
            scene_bounds = get_scene_bounds_around_peak(scene_time_bounds, initial_peak_scene_num, time_around_scene, just_forwards, offset=PREF_SCENE_TIME_OFFSET)
            early_spike_times = []
            late_spike_times = []
            for start, end in scene_bounds:
                early_spike_times.extend([t for t in row['all_spike_times_in_condition'] if start <= t <= min(end, early_phase_end)])
                late_spike_times.extend([t for t in row['all_spike_times_in_condition'] if max(start, late_phase_start) <= t <= end])
        else:
            early_spike_times = [t for t in row['all_spike_times_in_condition'] if start_time <= t <= early_phase_end]
            late_spike_times = [t for t in row['all_spike_times_in_condition'] if late_phase_start <= t <= end_time]

        # Calculate power spectrum for early phase
        if len(early_spike_times) > 1:
            freqs_early, power_early = calculate_power_spectrum(np.array(early_spike_times), fs=fs, min_freq=min_freq, max_freq=max_freq)
            null_powers_early = calculate_null_power_spectrum_with_jitter('early', np.array(early_spike_times), unit_id, num_shuffles=num_shuffles, fs=fs, min_freq=min_freq, max_freq=max_freq, jitter_window=jitter_window)
            power_early = smooth_power_spectrum(power_early, window_size=smoothing_window, fs=fs, freqs=freqs_early)
            if power_early.size > 0:
                peak_freq_early = freqs_early[np.argmax(power_early)]
            else:
                peak_freq_early = np.nan
        else:
            freqs_early = np.array([])
            power_early = np.array([])
            null_powers_early = np.array([])
            peak_freq_early = np.nan

        # Calculate power spectrum for late phase
        if len(late_spike_times) > 1:
            freqs_late, power_late = calculate_power_spectrum(np.array(late_spike_times), fs=fs, min_freq=min_freq, max_freq=max_freq)
            null_powers_late = calculate_null_power_spectrum_with_jitter('late', np.array(late_spike_times), unit_id, num_shuffles=num_shuffles, fs=fs, min_freq=min_freq, max_freq=max_freq, jitter_window=jitter_window)
            power_late = smooth_power_spectrum(power_late, window_size=smoothing_window, fs=fs, freqs=freqs_late)
            if power_late.size > 0:
                peak_freq_late = freqs_late[np.argmax(power_late)]
            else:
                peak_freq_late = np.nan
        else:
            freqs_late = np.array([])
            power_late = np.array([])
            null_powers_late = np.array([])
            peak_freq_late = np.nan

        # Process early phase
        if power_early.size > 0:
            # Calculate Z-scores for early power spectrum
            if null_powers_early.size > 0:
                null_mean_early = np.mean(null_powers_early, axis=0)
                null_std_early = np.std(null_powers_early, axis=0)
                z_power_early = (power_early - null_mean_early) / null_std_early
            else:
                z_power_early = np.array([])

            # Aggregate Z-scored power spectrum for master plot
            if task_type == 'random':
                early_random_list.append(z_power_early)
            elif task_type == 'structured':
                early_structured_list.append(z_power_early)

        # Process late phase
        if power_late.size > 0:
            # Calculate Z-scores for late power spectrum
            if null_powers_late.size > 0:
                null_mean_late = np.mean(null_powers_late, axis=0)
                null_std_late = np.std(null_powers_late, axis=0)
                z_power_late = (power_late - null_mean_late) / null_std_late
            else:
                z_power_late = np.array([])

            # Aggregate Z-scored power spectrum for master plot
            if task_type == 'random':
                late_random_list.append(z_power_late)
            elif task_type == 'structured':
                late_structured_list.append(z_power_late)

        # Calculate peak frequency changes (late - early) for each condition
        if not np.isnan(peak_freq_late) and not np.isnan(peak_freq_early):
            peak_change = peak_freq_late - peak_freq_early
            if task_type == 'random':
                peak_changes_random.append(peak_change)
            elif task_type == 'structured':
                peak_changes_structured.append(peak_change)

    # Calculate mean power spectra for early and late phases
    mean_early_random = np.mean(early_random_list, axis=0) if early_random_list else np.array([])
    mean_late_random = np.mean(late_random_list, axis=0) if late_random_list else np.array([])
    mean_early_structured = np.mean(early_structured_list, axis=0) if early_structured_list else np.array([])
    mean_late_structured = np.mean(late_structured_list, axis=0) if late_structured_list else np.array([])

    # Calculate changes (late - early) for each condition
    if len(late_structured_list) > 0 and len(early_structured_list) > 0:
        structured_changes = np.array(late_structured_list) - np.array(early_structured_list)  # Shape: (n_structured, num_freqs)
        mean_structured_change = np.mean(structured_changes, axis=0)
    else:
        structured_changes = np.array([])
        mean_structured_change = np.array([])

    if len(late_random_list) > 0 and len(early_random_list) > 0:
        random_changes = np.array(late_random_list) - np.array(early_random_list)  # Shape: (n_random, num_freqs)
        mean_random_change = np.mean(random_changes, axis=0)
    else:
        random_changes = np.array([])
        mean_random_change = np.array([])

    # Calculate observed double difference
    if mean_structured_change.size > 0 and mean_random_change.size > 0:
        observed_diff = mean_structured_change - mean_random_change
    else:
        observed_diff = np.array([])

    # ---------------------
    # Pooled Surrogate Set Approach
    # ---------------------

    if observed_diff.size > 0:
        # Combine all changes
        pooled_changes = np.concatenate((structured_changes, random_changes), axis=0)  # Shape: (total_units, num_freqs)

        # Number of structured and random units
        num_structured = structured_changes.shape[0]
        num_random = random_changes.shape[0]

        # Total number of units
        total_units = num_structured + num_random

        # Initialize array to store surrogate differences
        surrogate_diffs_pooled = np.zeros((NUM_POOLED_SHUFFLES, pooled_changes.shape[1]))

        print("Starting pooled surrogate permutation testing...")
        for shuffle_idx in range(NUM_POOLED_SHUFFLES):
            # Shuffle the condition labels
            shuffled_indices = np.random.permutation(total_units)
            shuffled_structured = pooled_changes[shuffled_indices[:num_structured], :]
            shuffled_random = pooled_changes[shuffled_indices[num_structured:], :]

            # Compute mean changes for shuffled groups
            mean_shuffled_structured = np.mean(shuffled_structured, axis=0)
            mean_shuffled_random = np.mean(shuffled_random, axis=0)

            # Compute surrogate difference
            surrogate_diff = mean_shuffled_structured - mean_shuffled_random
            surrogate_diffs_pooled[shuffle_idx, :] = surrogate_diff

            if (shuffle_idx + 1) % 100 == 0 or (shuffle_idx + 1) == NUM_POOLED_SHUFFLES:
                print(f"Completed {shuffle_idx + 1}/{NUM_POOLED_SHUFFLES} shuffles.")

        print("Permutation testing completed.")

        # Plot the observed difference against surrogate percentiles
        plot_pooled_surrogate_power_spectrum(
            observed_diff=observed_diff,
            surrogate_diffs_pooled=surrogate_diffs_pooled,
            freqs=freqs_late,  # Assuming freqs_late is representative
            region=region,
            output_dir=output_dir,
            min_freq=min_freq,
            max_freq=max_freq,
            y_limits=y_limits,
            smoothing_window_mean=SMOOTHING_WINDOW_MEAN
        )
    else:
        print(f"No valid power spectra found for region: {region}")

    # Plot peak frequency change distributions
    if peak_changes_structured and peak_changes_random:
        plot_peak_frequency_distributions(
            peak_changes_structured=peak_changes_structured,
            peak_changes_random=peak_changes_random,
            region=region,
            output_dir=output_dir,
            num_pooled_shuffles=NUM_POOLED_SHUFFLES
        )
    else:
        print(f"No valid peak frequency changes to plot for region: {region}")

    print("Analysis complete. Plots saved successfully.")


# ---------------------
# Execution
# ---------------------

if __name__ == "__main__":
    # Set a random seed for reproducibility
    np.random.seed(42)

    print("CWD:", os.getcwd())
    # print("Saving to:", os.path.abspath(filepath))

    # Load the DataFrame
    try:
        unit_condition_pair_info_df = pd.read_pickle('unit_condition_pair_info_df.pkl')
    except FileNotFoundError:
        print("Error: The file 'unit_condition_pair_info_df.pkl' was not found.")
        exit(1)
    except Exception as e:
        print(f"Error loading DataFrame: {e}")
        exit(1)

    # Call the function to plot for HPC
    plot_power_spectrum_early_late(
        df=unit_condition_pair_info_df,
        region='HPC',
        smoothing_window=SMOOTHING_WINDOW_INITIAL,
        baseline_subtraction=BASELINE_SUBTRACTION,
        output_dir='spike_spectra_plots',
        restrict_to_scene=RESTRICT_AROUND_SCENE_FLAG,
        time_around_scene=TIME_AROUND_SCENE,
        just_forwards=JUST_FORWARDS_FLAG,
        jitter_window=JITTER_WINDOW,
        num_shuffles=NUM_SHUFFLES,
        num_pooled_shuffles=NUM_POOLED_SHUFFLES
    )

    # Call the function to plot for non-HPC
    plot_power_spectrum_early_late(
        df=unit_condition_pair_info_df,
        region='non-HPC',
        smoothing_window=SMOOTHING_WINDOW_INITIAL,
        baseline_subtraction=BASELINE_SUBTRACTION,
        output_dir='spike_spectra_plots',
        restrict_to_scene=RESTRICT_AROUND_SCENE_FLAG,
        time_around_scene=TIME_AROUND_SCENE,
        just_forwards=JUST_FORWARDS_FLAG,
        jitter_window=JITTER_WINDOW,
        num_shuffles=NUM_SHUFFLES,
        num_pooled_shuffles=NUM_POOLED_SHUFFLES
    )

