from utils import cleanStrings as cs
import pdb
import numpy as np
import pandas as pd
import matplotlib
from utils import local_paths
matplotlib.use('Agg')  # Use 'Agg' backend for environments without a display
import matplotlib.pyplot as plt
import os
import pickle  # For loading .pkl files
from utils import coherenceTest_3 as sfc  # Import your helper functions module
from tqdm import tqdm  # For progress bars
import logging
from joblib import Parallel, delayed  # For parallel processing
from scipy.ndimage import gaussian_filter1d  # For smoothing
from scipy import stats  # For SEM calculations
from utils import pdfTextSet

# ---------------------
# Parameters
# ---------------------
STRUCTURED_COLOR_HPC='red'
RANDOM_COLOR_HPC='lightcoral'
STRUCTURED_COLOR_NON_HPC='brown'
RANDOM_COLOR_NON_HPC='#D2B48C'

processed_data_dir=local_paths.get_processed_data_dir()

import pandas as pd

import numpy as np
from sklearn.neighbors import KernelDensity

def subtract_kde_mode(arr, sigma=0.01):
    # Reshape array for KernelDensity input
    arr = arr.reshape(-1, 1)
    
    # Fit Kernel Density Estimation model
    kde = KernelDensity(kernel='gaussian', bandwidth=sigma).fit(arr)
    
    # Generate log density estimate for each point
    log_density = kde.score_samples(arr)
    
    # Find the index of the mode (point with maximum density)
    mode_index = np.argmax(log_density)
    
    # Extract the mode value
    mode_value = arr[mode_index][0]
    
    # Subtract the mode from the array
    result = arr.flatten() - mode_value
    
    return result


def add_region_category(df):
    # Add a new column 'region_category'
    df['region_category'] = df['region_name'].apply(lambda x: 'HPC' if x == 'HPC' else 'non-HPC')
    return df


# Flags and Parameters
RESTRICT_AROUND_SCENE_FLAG = True
JUST_FORWARDS_FLAG = False

MIN_NUM_SPIKES = 180

MIN_NUM_SPIKES = 0
MAX_NUM_SPIKES = 1000000
#MAX_NUM_SPIKES = 1000
#MAX_NUM_SPIKES = 500

TIME_AROUND_SCENE = 10  # Time around the initial preferred scene number to consider (in seconds)
PREF_SCENE_TIME_OFFSET = 0
SPECTRUM_TIME_WIND_SEC = 0.9

MIN_MEAN_FIRING_RATE = 0
MIN_PERC = 5
MAX_PERC = 95
STANDARD_YLIM = True
YMIN = -2
YMAX = 2.2
YMAX = 2.6
XMIN = 0
XMAX = 70

PHASE_DURATION = 100  # Duration for early and late phases in seconds
MAX_ISI = 0.2  # Maximum ISI to consider for the plots in seconds
NUM_BINS = 1000  # Number of bins for KDE
FS = 1000  # Sampling frequency for the spike train power spectrum (in Hz)
MAX_FREQ = 70  # Maximum frequency to consider for the power spectrum (in Hz)
MIN_FREQ = 1  # Minimum frequency to consider for the power spectrum (in Hz)
BASELINE_SUBTRACTION = False  # Flag to control baseline subtraction
SMOOTHING_WINDOW_INITIAL = 0.1  # Initial smoothing window in Hz
SMOOTHING_WINDOW_MEAN = 0.1  # Additional smoothing window for mean and percentiles in Hz
JITTER_WINDOW = 0.5  # Jitter window in seconds
NUM_SHUFFLES = 5000  # Number of shuffles for null power spectrum calculation
NUM_POOLED_SHUFFLES = 50000  # Number of shuffles for pooled permutation testing

y_limits = [YMIN, YMAX]
x_limits = [XMIN, XMAX]

# Define colors based on region
REGION_COLORS = {
    'HPC': 'red',
    'non-HPC': 'brown'
}

# ---------------------
# Helper Function Definitions
# ---------------------

def calculate_null_coherence_differences(structured_changes, random_changes, num_shuffles=5000):
    """
    Generate null distributions for coherence differences by shuffling condition labels within each region.
    
    Parameters:
        structured_changes (np.array): Coherence changes under structured condition. Shape: (num_structured_units, num_freqs)
        random_changes (np.array): Coherence changes under random condition. Shape: (num_random_units, num_freqs)
        num_shuffles (int): Number of shuffles to perform.
    
    Returns:
        np.array: Null distribution of coherence differences. Shape: (num_shuffles, num_freqs)
    """
    combined_changes = np.concatenate((structured_changes, random_changes), axis=0)
    num_structured = structured_changes.shape[0]
    num_random = random_changes.shape[0]
    total_units = num_structured + num_random

    surrogate_diffs = np.zeros((num_shuffles, combined_changes.shape[1]))

    for shuffle_idx in range(num_shuffles):
        shuffled_indices = np.random.permutation(total_units)
        shuffled_structured = combined_changes[shuffled_indices[:num_structured], :]
        shuffled_random = combined_changes[shuffled_indices[num_structured:], :]
        surrogate_diff = np.mean(shuffled_structured, axis=0) - np.mean(shuffled_random, axis=0)
        surrogate_diffs[shuffle_idx, :] = surrogate_diff

        if (shuffle_idx + 1) % 1000 == 0 or (shuffle_idx + 1) == num_shuffles:
            print(f"Completed {shuffle_idx + 1}/{num_shuffles} shuffles.")

    return surrogate_diffs

def plot_coherence_double_difference(observed_diff, surrogate_diffs, freqs, region, output_dir, y_limits, smoothing_window=1):
    """
    Plot the observed double difference coherence against surrogate null distributions.
    
    Parameters:
        observed_diff (np.array): Observed double difference (structured - random).
        surrogate_diffs (np.array): Surrogate coherence differences from bootstrapping.
        freqs (np.array): Frequency values.
        region (str): Brain region name ('HPC' or 'non-HPC').
        output_dir (str): Directory to save the plot.
        y_limits (list): y-axis limits for the plot.
        smoothing_window (float): Smoothing window size in Hz.
    """
    #plt.figure(figsize=(10, 6))
    plt.figure(figsize=(6, 6))

    # Compute percentiles
    percentile_1 = np.percentile(surrogate_diffs, 1, axis=0)
    percentile_99 = np.percentile(surrogate_diffs, 99, axis=0)

    # Smooth the observed difference and percentiles
    observed_diff_smoothed = gaussian_filter1d(observed_diff, sigma=smoothing_window)
    percentile_1_smoothed = gaussian_filter1d(percentile_1, sigma=smoothing_window)
    percentile_99_smoothed = gaussian_filter1d(percentile_99, sigma=smoothing_window)
    
    region_cat='HPC'
    if region != 'HPC':
        region_cat = 'non-HPC'

    # Plot surrogate percentile thresholds
    plt.fill_between(freqs, percentile_1_smoothed, percentile_99_smoothed, color=REGION_COLORS[region_cat], alpha=0.3, label='1st-99th Percentile Surrogate')

    # Plot observed double difference
    plt.plot(freqs, observed_diff_smoothed, label='Observed Structured - Random', color=REGION_COLORS[region_cat])

    # Highlight significant regions
    significant = (observed_diff_smoothed < percentile_1_smoothed) | (observed_diff_smoothed > percentile_99_smoothed)

    # Enhance significance by requiring consecutive significant points
    consecutive_sig = np.zeros_like(significant, dtype=bool)
    for i in range(1, len(significant)-1):
        if significant[i-1] and significant[i] and significant[i+1]:
            consecutive_sig[i-1] = True
            consecutive_sig[i] = True
            consecutive_sig[i+1] = True

    significant = consecutive_sig
    plt.scatter(freqs[significant], observed_diff_smoothed[significant], color=REGION_COLORS[region_cat], s=50, label='Significant Frequencies')

    # Configure plot aesthetics
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Coherence Difference (Structured - Random)')
    plt.title(f'Double Difference Spike-Field Coherence Spectrum - {region}')
    plt.xlim([freqs.min(), freqs.max()])
    if y_limits:
        plt.ylim(y_limits)
    plt.axhline(0, color='black', linestyle='--')
    plt.legend(fontsize='small', loc='upper right', framealpha=0.5)
    plt.tight_layout()

    # Save the plot
    plot_filename = os.path.join(output_dir, f"double_difference_coherence_{region}.pdf")
    plt.savefig(plot_filename)
    plt.close()
    print(f"Double difference coherence plot saved to {plot_filename}")

def plot_average_coherence_changes(observed_structured, observed_random, observed_structured_SEM, observed_random_SEM, freqs, region, output_dir, smoothing_window=1):
    """
    Plot the average coherence changes for structured and random conditions separately with SEM error bars.

    Parameters:
        observed_structured (np.array): Observed average coherence change for structured condition.
        observed_random (np.array): Observed average coherence change for random condition.
        observed_structured_SEM (np.array): SEM for structured condition.
        observed_random_SEM (np.array): SEM for random condition.
        freqs (np.array): Frequency values.
        region (str): Brain region name ('HPC' or 'non-HPC').
        output_dir (str): Directory to save the plot.
        smoothing_window (float): Smoothing window size in Hz.
    """
    #plt.figure(figsize=(10, 6))
    plt.figure(figsize=(6, 6))

    # Smooth the observed averages and SEMs
    observed_structured_smoothed = gaussian_filter1d(observed_structured, sigma=smoothing_window)
    observed_random_smoothed = gaussian_filter1d(observed_random, sigma=smoothing_window)
    observed_structured_SEM_smoothed = gaussian_filter1d(observed_structured_SEM, sigma=smoothing_window)
    observed_random_SEM_smoothed = gaussian_filter1d(observed_random_SEM, sigma=smoothing_window)

    # Plot average coherence changes with SEM shading for structured condition
    # Map non-HPC regions to "non-HPC" category
    region_cat='HPC'
    sColor=STRUCTURED_COLOR_HPC
    rColor=RANDOM_COLOR_HPC
    if region != 'HPC':
        region_cat= 'non-HPC'
        sColor=STRUCTURED_COLOR_NON_HPC
        rColor=RANDOM_COLOR_NON_HPC

    plt.plot(freqs, observed_structured_smoothed, label='Structured Condition', color=sColor)
    plt.fill_between(freqs, 
                     observed_structured_smoothed - observed_structured_SEM_smoothed, 
                     observed_structured_smoothed + observed_structured_SEM_smoothed, 
                     color=sColor, alpha=0.3)

    # Plot average coherence changes with SEM shading for random condition
    plt.plot(freqs, observed_random_smoothed, label='Random Condition', color=rColor)
    plt.fill_between(freqs, 
                     observed_random_smoothed - observed_random_SEM_smoothed, 
                     observed_random_smoothed + observed_random_SEM_smoothed, 
                     color=rColor, alpha=0.3)

    # Configure plot aesthetics
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Average Coherence Change')
    plt.title(f'Average Spike-Field Coherence Change - {region}')
    plt.xlim([freqs.min(), freqs.max()])
    #plt.ylim([-0.012,0.017])
    #plt.ylim([-0.01,0.01])
    plt.ylim([-0.0025,0.0075])
    plt.axhline(0, color='black', linestyle='--')
    plt.legend(fontsize='small', loc='upper right', framealpha=0.5)
    plt.tight_layout()

    # Save the plot
    plot_filename = os.path.join(output_dir, f"average_coherence_changes_{region}.pdf")
    plt.savefig(plot_filename)
    plt.close()
    print(f"Average coherence changes plot saved to {plot_filename}")


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
    color = REGION_COLORS.get(region) #, 'lightbrown')  # Default to lightbrown if region not found

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

    # Compute 1st and 99th percentiles
    percentile_1 = np.percentile(surrogate_diffs, MIN_PERC)
    percentile_99 = np.percentile(surrogate_diffs, MAX_PERC)

    # Plot mean ± SEM
    plt.errorbar(positions, means, yerr=sems, fmt='o', color='black', label='Mean ± SEM')

    # Plot surrogate thresholds
    plt.axhline(y=percentile_1, color='grey', linestyle='--', label='1st Percentile Surrogate')
    plt.axhline(y=percentile_99, color='grey', linestyle='-.', label='99th Percentile Surrogate')

    plt.xticks(positions, labels, rotation=45)
    plt.ylabel('Peak Frequency Change (Hz)')
    plt.title(f'Distribution of Peak Frequency Changes - {region}')
    plt.legend()
    plt.tight_layout()

    # Save the plot
    plot_filename = os.path.join(output_dir, f"peak_frequency_change_distribution_{region}.pdf")
    plt.savefig(plot_filename)
    plt.close()
    print(f"Peak frequency change distribution plot saved to {plot_filename}")

# ---------------------
# Main Processing Functions
# ---------------------

def process_unit_with_coherence_diff(idx, row, params):
    """
    Processes a single unit to compute coherence differences.

    Parameters:
        idx: int, index of the unit.
        row: pandas Series, row data.
        params: dict, parameters for processing.

    Returns:
        tuple: (condition, region, coherence_change_array, peak_frequency_change) or None if processing fails.
    """
    condition = row[params['condition_col']]
    sessIDStr = row[params['sessIDStr_col']]
    region_name = row[params['region_name_col']]
    specific_region_name = row.get(params['specific_region_name_col'], 'NA')  # Handle missing data
    subfield_raw = row.get(params['subfield_col'], 'NA')  # Handle missing data
    subfield = cs.clean_string(subfield_raw)  # Clean subfield

    chNameStr = row.get(params['chNameStr_col'], 'NA')  # Handle missing data
    unitNumStr = row.get(params['unitNumStr_col'], 'NA')  # Handle missing data
    condition_time_bounds = row[params['condition_time_bounds_col']]  # Expected to be a tuple/list (start, end)
    all_spike_times = row[params['all_spike_times_col']]
    lfp_filepath = row[params['lfp_filepath_col']]

    firing_rate_distance_dependence_response = row.get('firing_rate_distance_dependence_response', 0)

    # Adjust lfp_filepath based on sessIDStr and subfield
    if region_name =='HPC':
        if 'pt3' in sessIDStr:
            if 'dentate' in subfield.lower(): 
                lfp_filepath = 'pt3_sess1_ch73_downsampled_lfp.pkl'  # LEFT MACRO
            else:
                lfp_filepath = 'pt3_sess1_ch153_downsampled_lfp.pkl'  # RIGHT MACRO
        elif 'pt1' in sessIDStr:  # local microwire is high passed filter for this pt
            lfp_filepath = f'{sessIDStr}_ch101_downsampled_lfp.pkl'

    '''
    if region_name =='non-HPC':
        if 'pt3' in sessIDStr:
            return None #microwire has high passed filter
        elif 'pt1' in sessIDStr:  # local microwire is high passed filter for this pt
            return None
    '''
    # Log processing information
    logging.debug(f"Processing Unit {idx+1} - Condition: {condition}, Region: {region_name}, Session: {sessIDStr}")

    # Load LFP data and sampling rate from the .pkl file
    try:
        with open(lfp_filepath, 'rb') as f:
            lfp_data = pickle.load(f)
            if not isinstance(lfp_data, (list, tuple)) or len(lfp_data) != 2:
                raise ValueError("Invalid .pkl file structure. Expected a list or tuple with two elements.")
            lfp = lfp_data[0]  # First element is the LFP numpy array
            sampling_rate = lfp_data[1]  # Second element is the sampling rate as float
        logging.debug(f"LFP data loaded successfully from {lfp_filepath}. Sampling rate: {sampling_rate}")
    except Exception as e:
        logging.error(f"Error loading LFP data from {lfp_filepath} for Unit {idx+1}: {e}")
        return None  # Skip to the next unit

    # Subset LFP based on condition_time_bounds
    try:
        condition_start, condition_end = condition_time_bounds  # in seconds
        if condition_end <= condition_start:
            raise ValueError(f"Invalid condition_time_bounds: start ({condition_start}) >= end ({condition_end})")
        start_index = int(condition_start * sampling_rate)
        end_index = int(condition_end * sampling_rate)
        start_index = max(start_index, 0)
        end_index = min(end_index, len(lfp))
        lfp_condition = lfp[start_index:end_index]
        if len(lfp_condition) == 0:
            raise ValueError("Empty LFP segment after subsetting based on condition_time_bounds.")
        logging.debug(f"LFP condition subset: {len(lfp_condition)} samples from {condition_start}s to {condition_end}s")
    except Exception as e:
        logging.error(f"Error subsetting LFP for Unit {idx+1}: {e}")
        return None

    # Generate time vector for the condition-specific LFP
    duration_condition = condition_end - condition_start  # seconds
    time_condition = np.linspace(condition_start, condition_end, len(lfp_condition), endpoint=False)

    # Subset spike times within the condition
    spike_times = np.array(all_spike_times)
    spike_times_condition = spike_times[(spike_times >= condition_start) & (spike_times <= condition_end)]
    spike_times_condition -= condition_start  # Adjust spike times relative to condition start
    logging.debug(f"Total spike times in condition: {len(spike_times_condition)}")

    phase_duration = 150
    # Define early and late phase boundaries
    early_phase_end = phase_duration  # First phase_duration seconds
    late_phase_start = duration_condition - phase_duration  # Last phase_duration seconds

    # Split spike times into early and late
    early_spike_times_full = spike_times_condition[spike_times_condition <= early_phase_end]
    late_spike_times_full = spike_times_condition[spike_times_condition > late_phase_start]
    num_early_spikes = len(early_spike_times_full)
    num_late_spikes = len(late_spike_times_full)
    logging.debug(f"Early spikes: {num_early_spikes}, Late spikes: {num_late_spikes}")

    if num_early_spikes > MAX_NUM_SPIKES or num_late_spikes > MAX_NUM_SPIKES:
        return None
    # Determine the minimum number of spikes between early and late
    if num_early_spikes < MIN_NUM_SPIKES or num_late_spikes < MIN_NUM_SPIKES:
        logging.warning(f"Unit {idx+1} has insufficient spikes in {'early' if num_early_spikes < MIN_NUM_SPIKES else 'late'} phase. Skipping.")
        return None  # Skip units with insufficient spikes in either phase

    #if num_early_spikes > MAX_NUM_SPIKES or num_late_spikes > MAX_NUM_SPIKES:
    #    return None

    min_spikes = min(num_early_spikes, num_late_spikes)  # MIN_NUM_SPIKES
    min_spikes=150
    logging.debug(f"Minimum spikes for matching: {min_spikes}")

    # Initialize arrays to accumulate coherence results
    coh_early_accum = None
    coh_late_accum = None

    # Initialize frequency array
    freqs = None

    # Perform resampling to match spike counts
    for resample_idx in range(params['num_resamples']):
        # Randomly sample without replacement
        try:
            early_spike_sample = np.random.choice(early_spike_times_full, size=min_spikes, replace=False)
            late_spike_sample = np.random.choice(late_spike_times_full, size=min_spikes, replace=False)
        except ValueError as ve:
            logging.error(f"Sampling error for Unit {idx+1}, Resample {resample_idx+1}: {ve}")
            return None  # Exit processing for this unit

        # Compute coherence for the sampled early spike times
        try:
            freqs, coh_early = sfc.compute_spike_field_coherence(
                early_spike_sample, lfp_condition, sampling_rate, 
                nperseg=params['nperseg'], noverlap=params['noverlap'], 
                freq_range=params['freq_range']
            )
        except Exception as e:
            logging.error(f"Error computing early coherence for Unit {idx+1}, Resample {resample_idx+1}: {e}")
            return None  # Exit processing for this unit

        # Compute coherence for the sampled late spike times
        try:
            _, coh_late = sfc.compute_spike_field_coherence(
                late_spike_sample, lfp_condition, sampling_rate, 
                nperseg=params['nperseg'], noverlap=params['noverlap'], 
                freq_range=params['freq_range']
            )
        except Exception as e:
            logging.error(f"Error computing late coherence for Unit {idx+1}, Resample {resample_idx+1}: {e}")
            return None  # Exit processing for this unit

        # Initialize accumulators based on the shape of coh_early and coh_late
        if coh_early_accum is None and coh_late_accum is None:
            coh_early_accum = np.zeros_like(coh_early)
            coh_late_accum = np.zeros_like(coh_late)
            logging.debug(f"Initialized accumulators with shape {coh_early_accum.shape}")

        # Accumulate coherence results
        coh_early_accum += coh_early
        coh_late_accum += coh_late

    # Average coherence across resamples
    coh_early_avg = coh_early_accum / params['num_resamples']
    coh_late_avg = coh_late_accum / params['num_resamples']
    logging.debug(f"Average coherence computed for Unit {idx+1}")

    coh_late_avg = gaussian_filter1d(coh_late_avg, sigma=1)
    coh_early_avg = gaussian_filter1d(coh_early_avg, sigma=1)

    # Calculate Change in Coherence from Early to Late
    coh_change = coh_late_avg - coh_early_avg  # Positive values indicate increased coherence
    
    coh_change=subtract_kde_mode(coh_change)

    #coh_change = (coh_late_avg - coh_early_avg)/ (coh_late_avg + coh_early_avg) # Positive values indicate increased coherence
    logging.debug(f"Coherence change calculated for Unit {idx+1}")

    # Compute Peak Frequency Changes (late - early)
    if not np.isnan(coh_change).all() and not np.isnan(coh_early_avg).all():
        peak_freq_early = freqs[np.argmax(coh_early_avg)]
        peak_freq_late = freqs[np.argmax(coh_late_avg)]
        peak_change = peak_freq_late - peak_freq_early
    else:
        peak_change = np.nan

    # Define output paths for plots
    coherence_plot_path = os.path.join(
        params['output_dir'], 
        f'unit_{idx+1}_sess_{sessIDStr}_ch{chNameStr}_unit{unitNumStr}_cond_{condition}_region_{region_name}_subfield_{subfield}_coherence.pdf'
    )
    coherence_change_plot_path = os.path.join(
        params['output_dir'],
        f'unit_{idx+1}_sess_{sessIDStr}_ch{chNameStr}_unit{unitNumStr}_cond_{condition}_region_{region_name}_subfield_{subfield}_coh_change.pdf'
    )
    lfp_spike_plot_path = os.path.join(
        params['output_dir'], 
        f'unit_{idx+1}_sess_{sessIDStr}_ch{chNameStr}_unit{unitNumStr}_cond_{condition}_region_{region_name}_subfield_{subfield}_lfp_spikes.pdf'
    )

    # Construct detailed plot titles
    coherence_title = (
        f"Spike-Field Coherence for Unit {chNameStr}-{unitNumStr}\n"
        f"Session: {sessIDStr}, Condition: {condition}, Region: {specific_region_name}, Subfield: {subfield}\n"
        f"Early Spikes: {num_early_spikes}, Late Spikes: {num_late_spikes}, Resamples: {params['num_resamples']}"
    )

    coh_change_title = (
        f"Change in Spike-Field Coherence for Unit {chNameStr}-{unitNumStr}\n"
        f"Session: {sessIDStr}, Condition: {condition}, Region: {specific_region_name}, Subfield: {subfield}\n"
        f"Early Spikes: {num_early_spikes}, Late Spikes: {num_late_spikes}, Resamples: {params['num_resamples']}\n"
        f"Peak Frequency Change: {peak_change:.2f} Hz"
    )

    lfp_spike_title = (
        f"LFP and Spike Times for Unit {chNameStr}-{unitNumStr}\n"
        f"Session: {sessIDStr}, Condition: {condition}, Region: {specific_region_name}, Subfield: {subfield}\n"
        f"Early Spikes: {num_early_spikes}, Late Spikes: {num_late_spikes}, Resamples: {params['num_resamples']}"
    )

    # Plot and Save Spike-Field Coherence for Early and Late
    try:
        #plt.figure(figsize=(6, 6))
        plt.figure(figsize=(4, 4))
        plt.plot(freqs, coh_early_avg, label='Early Spike-Field Coherence', color='k', marker='o',fillstyle='none', linestyle='-',markeredgewidth=0.5,linewidth=0.5,markersize=10)
        plt.plot(freqs, coh_late_avg, label='Late Spike-Field Coherence', color='k', marker='o', linestyle='-',markeredgewidth=0.5,linewidth=0.5,markersize=10)
        plt.title(coherence_title)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Coherence')
        plt.xlim(freqs.min(), freqs.max())
        plt.ylim([-0.002, 0.05])  # Uncomment and adjust if needed
        plt.legend()
        plt.tight_layout()
        plt.savefig(coherence_plot_path)
        plt.close()
        logging.debug(f"Coherence plot saved for Unit {idx+1}")
    except Exception as e:
        logging.error(f"Error plotting coherence for Unit {idx+1}: {e}")
        return None

    # Plot and Save Change in Coherence with Z-scores
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(freqs, coh_change, label='Coherence Change (Late - Early)', color='purple', marker='o', linestyle='-')
        plt.axhline(0, color='k', linestyle='--')
        plt.title(coh_change_title)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Coherence Change')
        plt.xlim(freqs.min(), freqs.max())
        #plt.ylim([-5, 5])  # Adjust based on your data
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(coherence_change_plot_path)
        plt.close()
        logging.debug(f"Coherence change plot saved for Unit {idx+1}")
    except Exception as e:
        logging.error(f"Error plotting coherence change for Unit {idx+1}: {e}")
        return None

    # Optionally, plot LFP and Spike Times (if desired)
    # Uncomment and implement if necessary

    # Append results to summary
    summary_dict = {
        'unit_id': idx+1,
        'sessIDStr': sessIDStr,
        'chNameStr': chNameStr,
        'unitNumStr': unitNumStr,
        'condition': condition,
        'region_name': region_name,
        'specific_region_name': specific_region_name,
        'subfield': subfield,
        'num_early_spikes': num_early_spikes,
        'num_late_spikes': num_late_spikes,
        'min_spikes_matched': min_spikes,
        'num_resamples': params['num_resamples'],
        'avg_coherence_early': np.mean(coh_early_avg),
        'avg_coherence_late': np.mean(coh_late_avg),
        'change_in_coherence': coh_change.tolist(),  # Store as list for per-frequency analysis
        'peak_frequency_change': peak_change,
        'coherence_plot': coherence_plot_path,
        'coherence_change_plot': coherence_change_plot_path,
        'lfp_spike_plot': lfp_spike_plot_path
    }

    logging.debug(f"Summary results appended for Unit {idx+1}")
    logging.info(f"Completed processing for Unit {idx+1}")

    return (condition, region_name, coh_change, peak_change)

def process_dataframe_with_bootstrapping(df, 
                                         condition_col='task_type',  # Set to 'task_type' as per your change
                                         condition_time_bounds_col='condition_time_bounds',
                                         sessIDStr_col='sessIDStr',
                                         region_name_col='region_name',
                                         specific_region_name_col='specific_region_name',
                                         subfield_col='subfield',
                                         chNameStr_col='chNameStr',
                                         unitNumStr_col='unitNumStr',
                                         all_spike_times_col='all_spike_times_in_condition',
                                         lfp_filepath_col='lfp_filepath',
                                         output_dir='coherence_resultsWithZ',
                                         summary_csv='coherence_summary.csv',
                                         summary_pickle='coherence_summary.pkl',
                                         population_summary_dir='population_summary',
                                         nperseg=1024,
                                         noverlap=512, 
                                         freq_range=(1, 70),
                                         num_resamples=30,  # Number of resampling iterations for spike matching
                                         num_nulls=20,      # Number of null resamples for Z-score normalization
                                         jitter_window=0.05, # Jitter window in seconds
                                         num_shuffles=5000,  # Number of shuffles for bootstrapping
                                         smoothing_window=1,  # Smoothing window for plotting
                                         y_limits=None, #[-0.1, 0.1],  # Y-axis limits for average coherence plot
                                         n_jobs=-1):  # Number of parallel jobs (-1 uses all available cores)
    """
    Processes the entire DataFrame in parallel to compute Spike-Field Coherence for early and late phases,
    analyze changes, normalize using bootstrapping, and generate plots for structured and random conditions separately,
    keeping different brain regions separate.

    Parameters:
        df (pd.DataFrame): DataFrame containing unit condition pair information.
        condition_col (str): Name of the column specifying experimental condition.
        condition_time_bounds_col (str): Name of the column with condition time bounds.
        sessIDStr_col (str): Name of the session ID string column.
        region_name_col (str): Name of the brain region name column.
        specific_region_name_col (str): Name of the specific region name column.
        subfield_col (str): Name of the subfield name column.
        chNameStr_col (str): Name of the channel name column.
        unitNumStr_col (str): Name of the unit number string column.
        all_spike_times_col (str): Name of the column with all spike times in condition.
        lfp_filepath_col (str): Name of the column with LFP filepath.
        output_dir (str): Directory to save the results.
        summary_csv (str): Filename for the summary CSV.
        summary_pickle (str): Filename for the summary pickle file.
        population_summary_dir (str): Directory to save population-level summaries.
        nperseg (int): Length of each segment for coherence computation.
        noverlap (int): Number of points to overlap between segments.
        freq_range (tuple): Frequency range for coherence computation.
        num_resamples (int): Number of resampling iterations for spike matching.
        num_nulls (int): Number of null resamples for Z-score normalization.
        jitter_window (float): Jitter window in seconds for null model.
        num_shuffles (int): Number of shuffles for bootstrapping.
        smoothing_window (float): Smoothing window size in Hz for plotting.
        y_limits (list): Y-axis limits for the average coherence plot.
        n_jobs (int): Number of parallel jobs (-1 uses all available cores).
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    population_summary_path = os.path.join(output_dir, population_summary_dir)
    os.makedirs(population_summary_path, exist_ok=True)

    # Configure logging
    logging.basicConfig(filename=os.path.join(output_dir, 'coherence_processing.log'), level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s:%(message)s')

    # Prepare parameters dictionary
    params = {
        'condition_col': condition_col,
        'condition_time_bounds_col': condition_time_bounds_col,
        'sessIDStr_col': sessIDStr_col,
        'region_name_col': region_name_col,
        'specific_region_name_col': specific_region_name_col,
        'subfield_col': subfield_col,
        'chNameStr_col': chNameStr_col,
        'unitNumStr_col': unitNumStr_col,
        'all_spike_times_col': all_spike_times_col,
        'lfp_filepath_col': lfp_filepath_col,
        'output_dir': output_dir,
        'nperseg': nperseg,
        'noverlap': noverlap,
        'freq_range': freq_range,
        'num_resamples': num_resamples,
        'num_nulls': num_nulls,
        'jitter_window': jitter_window
    }

    '''
    # Process units in parallel
    print("Starting parallel processing of units...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_unit_with_coherence_diff)(idx, row, params) for idx, row in tqdm(df.iterrows(), total=df.shape[0])
    )

    # Separate structured and random coherence changes per region
    summary_results = []
    for res in results:
        if res is None:
            continue
        condition, region, coh_change, peak_change = res
        summary_results.append(res)

    # Create summary DataFrame with region information
    summary_df = pd.DataFrame({
        'condition': [res[0] for res in summary_results],
        'region_name': [res[1] for res in summary_results],
        'coherence_change': [res[2] for res in summary_results],
        'peak_frequency_change': [res[3] for res in summary_results]
    })

    # Save summary to CSV and pickle
    summary_csv_path = os.path.join(output_dir, summary_csv)
    summary_pickle_path = os.path.join(output_dir, summary_pickle)
    summary_df.to_csv(summary_csv_path, index=False)
    with open(summary_pickle_path, 'wb') as f:
        pickle.dump(summary_df, f)
    print(f"Summary saved to {summary_csv_path} and {summary_pickle_path}")
    '''
    summary_pickle_path = os.path.join(processed_data_dir, summary_pickle)
    # Check if the pickle file exists
    if os.path.exists(summary_pickle_path):
        # Load the summary DataFrame from the pickle
        with open(summary_pickle_path, 'rb') as f:
            summary_df = pickle.load(f)
        print(f"Loaded summary from {summary_pickle_path}")
    else:
        # Process units in parallel
        print("Starting parallel processing of units...")
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_unit_with_coherence_diff)(idx, row, params) for idx, row in tqdm(df.iterrows(), total=df.shape[0])
        )

        # Separate structured and random coherence changes per region
        summary_results = []
        for res in results:
            if res is None:
                continue
            condition, region, coh_change, peak_change = res
            summary_results.append(res)

        # Create summary DataFrame with region information
        summary_df = pd.DataFrame({
            'condition': [res[0] for res in summary_results],
            'region_name': [res[1] for res in summary_results],
            'coherence_change': [res[2] for res in summary_results],
            'peak_frequency_change': [res[3] for res in summary_results]
        })

        # Save summary to CSV and pickle
        summary_csv_path = os.path.join(output_dir, summary_csv)
        summary_df.to_csv(summary_csv_path, index=False)
        with open(summary_pickle_path, 'wb') as f:
            pickle.dump(summary_df, f)
        print(f"Summary saved to {summary_csv_path} and {summary_pickle_path}")

    # Identify unique regions
    unique_regions = summary_df['region_name'].unique()
    print(f"Unique regions found: {unique_regions}")

    # Process each region separately
    for region in unique_regions:
        print(f"\nProcessing region: {region}")

        # Subset data for the current region
        region_df = summary_df[summary_df['region_name'] == region]

        # Separate structured and random coherence changes
        structured_changes = np.array([coh_change for cond, coh_change in zip(region_df['condition'], region_df['coherence_change']) if cond == 'structured'])
        random_changes = np.array([coh_change for cond, coh_change in zip(region_df['condition'], region_df['coherence_change']) if cond == 'random'])
        peak_changes_structured = [peak_change for cond, peak_change in zip(region_df['condition'], region_df['peak_frequency_change']) if cond == 'structured']
        peak_changes_random = [peak_change for cond, peak_change in zip(region_df['condition'], region_df['peak_frequency_change']) if cond == 'random']

        # Create separate arrays for structured and random coherence changes
        if structured_changes.size > 0:
            structured_changes_mean = np.mean(structured_changes, axis=0)
            structured_changes_sem = np.std(structured_changes, axis=0) / np.sqrt(structured_changes.shape[0])
            print("Computed average coherence change for structured condition.")
        else:
            structured_changes_mean = None
            print("No structured coherence changes available for this region.")

        if random_changes.size > 0:
            random_changes_mean = np.mean(random_changes, axis=0)
            random_changes_sem = np.std(random_changes, axis=0) / np.sqrt(random_changes.shape[0])

            print("Computed average coherence change for random condition.")
        else:
            random_changes_mean = None
            print("No random coherence changes available for this region.")

        # Compute observed difference between structured and random
        if structured_changes.size > 0 and random_changes.size > 0:
            observed_diff = structured_changes_mean - random_changes_mean
            print("Computed observed difference between structured and random coherence changes.")
        else:
            observed_diff = None
            print("Insufficient data to compute observed difference for this region.")

        # Generate null distribution for the difference
        if structured_changes.size > 0 and random_changes.size > 0:
            print("Generating null distribution for coherence difference...")
            surrogate_diff = calculate_null_coherence_differences(structured_changes, random_changes, num_shuffles=num_shuffles)
            print("Null distribution for coherence difference generated.")
        else:
            surrogate_diff = None
            print("Insufficient data to generate null distribution for this region.")

        # Define frequency array
        start_freq = 1.953125
        end_freq = 68.359375
        step_size = 1.953125
        #freqs = np.arange(start_freq, end_freq + step_size, step_size)
        freqs = np.arange(start_freq, end_freq + step_size, step_size/(nperseg/1024.0))

        # Plot the observed average coherence changes separately
        if structured_changes_mean is not None and random_changes_mean is not None:
            print("Plotting average coherence changes for structured and random conditions...")
            plot_average_coherence_changes(
                observed_structured=structured_changes_mean,
                observed_random=random_changes_mean,
                observed_structured_SEM=structured_changes_sem,
                observed_random_SEM=random_changes_sem,
                freqs=freqs,
                region=region,
                output_dir=output_dir,
                smoothing_window=smoothing_window
            )
        else:
            print("Cannot plot average coherence changes due to insufficient data for this region.")

        # Plot the observed difference with null distribution
        if observed_diff is not None and surrogate_diff is not None:
            print("Plotting coherence difference with significance thresholds...")
            plot_coherence_double_difference(
                observed_diff=observed_diff,
                surrogate_diffs=surrogate_diff,
                freqs=freqs,
                region=region,
                output_dir=output_dir,
                y_limits=y_limits,
                smoothing_window=smoothing_window
            )
        else:
            print("Cannot plot coherence difference due to insufficient data for this region.")

        # Plot peak frequency change distributions
        if peak_changes_structured and peak_changes_random:
            print("Plotting peak frequency change distributions...")
            plot_peak_frequency_distributions(
                peak_changes_structured=peak_changes_structured,
                peak_changes_random=peak_changes_random,
                region=region,
                output_dir=output_dir,
                num_pooled_shuffles=NUM_POOLED_SHUFFLES
            )
        else:
            print("No valid peak frequency changes to plot for this region.")

    print("\nBootstrapped coherence analysis complete.")

if __name__ == "__main__":
    # Set a random seed for reproducibility
    np.random.seed(42)

    # Load the DataFrame
    try:
        # Replace with your actual pickle file name
        unit_condition_pair_info_df = pd.read_pickle('unit_condition_pair_info_df_with_firing_rate_dependence_folded.pkl')
        print("DataFrame loaded successfully.")
    except FileNotFoundError:
        print("Error: The file 'unit_condition_pair_info_df_with_firing_rate_dependence_folded.pkl' was not found.")
        exit(1)
    except Exception as e:
        print(f"Error loading DataFrame: {e}")
        exit(1)

    # Ensure 'all_spike_times_in_condition' are numpy arrays
    if unit_condition_pair_info_df['all_spike_times_in_condition'].dtype == object:
        unit_condition_pair_info_df['all_spike_times_in_condition'] = unit_condition_pair_info_df['all_spike_times_in_condition'].apply(
            lambda x: np.array(x) if isinstance(x, (list, np.ndarray)) else np.array([])
        )


    unit_condition_pair_info_df=add_region_category(unit_condition_pair_info_df)
   
    # Call the processing function with bootstrapping
    process_dataframe_with_bootstrapping(
        df=unit_condition_pair_info_df,
        condition_col='task_type',                             # Column specifying experimental condition
        condition_time_bounds_col='condition_time_bounds',     # Column with condition time bounds (tuple/list)
        sessIDStr_col='sessIDStr',                             # Session ID column
        region_name_col='region_category',                         # Brain region name column
        specific_region_name_col='specific_region_name',       # Specific region name column
        subfield_col='subfield',                                # Subfield name column
        chNameStr_col='chNameStr',                              # Channel name column
        unitNumStr_col='unitNumStr',                            # Unit number string column
        all_spike_times_col='all_spike_times_in_condition',     # Spike times column
        lfp_filepath_col='lfp_filepath',                       # LFP filepath column
        output_dir='coherence_resultsWithZ_SeparateRegions',                    # Desired output directory
        summary_csv='coherence_summary.csv',                    # Summary CSV filename
        summary_pickle='coherence_summary.pkl',                 # Summary pickle filename
        population_summary_dir='population_summary',            # Directory for population-level summaries
        nperseg=1024,#2000,#1024,                                           # Length of each segment for coherence computation
        noverlap=512,                                           # Number of points to overlap between segments
        freq_range=(1, 70),                                     # Frequency range for coherence computation
        num_resamples=50,                                       # Number of resampling iterations for spike matching
        num_nulls=20,                                           # Number of null resamples for Z-score normalization
        jitter_window=0.05,                                     # Jitter window in seconds
        num_shuffles=5000,                                      # Number of shuffles for bootstrapping
        smoothing_window=0.01,#1,                                     # Smoothing window size in Hz for plotting
        y_limits=None, #[-0.1, 0.1],                                   # Y-axis limits for average coherence plot
        n_jobs=-1                                               # Use all available CPU cores
    )

    print("Processing complete.")


