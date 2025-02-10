from utils import pdfTextSet
import pdb
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from scipy.stats import circmean
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from matplotlib.colors import Normalize, ListedColormap
import random

from utils import kempter  # Ensure this module is available in your environment

# Define the colormap
plasma_cmap = plt.cm.plasma
cutoff = 0.85  # 85% cutoff
CMAP = ListedColormap(plasma_cmap(np.linspace(0, cutoff, 256)))
SIGMA=2
SIGMA=1.5
SIGMA=1
#SIGMA=0
MIN_SLOPE=-5
MAX_SLOPE=5
BOUND_LINE_WIDTH=1
# Parse command-line arguments
lowcut = float(sys.argv[1])
highcut = float(sys.argv[2])
N_TRIALS_PER_PHASE = int(sys.argv[3])

filterSettingStr = f'{lowcut}HzTo{highcut}Hz'

# Load the units_df DataFrame from the pickle file
#with open(f'coherence_resultsWithZ_SeparateRegions/coherence_summary_{filterSettingStr}.pkl', 'rb') as f:
with open(f'coherence_summary_{filterSettingStr}.pkl', 'rb') as f:
    units_df = pickle.load(f)

# Ensure the output directory exists
output_dir = 'spike_phase_time_heatmaps'
os.makedirs(output_dir, exist_ok=True)

# Initialize dictionaries to hold cumulative histograms for each (condition, region_category)
# Separate dictionaries for first and last N_TRIALS_PER_PHASE trials
cumulative_heatmaps_first = {}  # For first N_TRIALS_PER_PHASE trials
cumulative_relative_times_first = {}
cumulative_phases_first = {}

cumulative_heatmaps_last = {}   # For last N_TRIALS_PER_PHASE trials
cumulative_relative_times_last = {}
cumulative_phases_last = {}

# Initialize a dictionary to keep track of the number of units for each key
unit_counts = {}  # Key: (condition, region_category), Value: number of units

# Initialize dictionary to collect delta metrics per category
delta_metrics = {}  # Key: (condition, region_category), Value: delta_metric per category

# Initialize dictionary to collect observed metrics for last trials per category
observed_metrics_last = {}  # Key: (condition, region_category), Value: observed metric

# Initialize dictionary to store p-values
p_values = {}  # Key: (condition, region_category), Value: p-value

# Initialize a list to collect individual unit data for the last trials group
unit_pool_last = []  # List of dictionaries with 'heatmap', 'relative_times', 'phases'

# Flags
UNWRAP_MEAN = True

# Change in Phase-Time Correlation choice: 'unwrapped_linear_regression', 'circular_linear_regression_rho', 'circular_linear_regression_error', 'circular_linear_regression_slope'
metric_choice = 'circular_linear_regression_rho'  # Set your desired metric here
metric_choice = 'unwrapped_linear_regression'  # Set your desired metric here

# Phase display settings
PHASE_DISP_MIN = -3 * np.pi
PHASE_DISP_MAX = 3 * np.pi

# Time before and after scene onset
TIME_BEFORE = 2.5  # Time before scene onset in seconds
TIME_AFTER = 2.5   # Time after scene onset in seconds

# Adjust number of bins based on TIME_BEFORE
num_time_bins = int(20 * (TIME_BEFORE / 3))  # Adjust as needed
num_phase_bins = 20  # Adjust as needed
#num_time_bins = 2*int(20 * (TIME_BEFORE / 3))  # Adjust as needed
#num_phase_bins = 20  # Adjust as needed

# Time range for plotting
TIME_DISP_MIN = -2
TIME_DISP_MAX = 2
#TIME_DISP_MAX = 2.1

# Time and phase ranges for histograms
time_range = [-TIME_BEFORE, TIME_AFTER]
phase_range = [-np.pi, np.pi]  # Phase ranges from -π to π

# Prepare bin edges (edges are the same for all units)
xedges = np.linspace(time_range[0], time_range[1], num_time_bins + 1)
yedges = np.linspace(phase_range[0], phase_range[1], num_phase_bins + 1)
phase_bin_centers = (yedges[:-1] + yedges[1:]) / 2
time_bin_centers = (xedges[:-1] + xedges[1:]) / 2

# Color scale settings
VMIN = 120
VMAX = 160
VMIN_cntrl = 110
VMAX_cntrl = 145

# Define window size and step size for moving average
window_size = 0.25  # in seconds
step_size = 0.25   # in seconds
#window_size = 0.5  # in seconds
#step_size = 0.5   # in seconds
#window_size = 0.1  # in seconds
#step_size = 0.1  # in seconds

# Gaussian kernel for smoothing (3x3)
gaussian_kernel = np.array([[0.07511361, 0.1238414 , 0.07511361],
                            [0.1238414 , 0.20417996, 0.1238414 ],
                            [0.07511361, 0.1238414 , 0.07511361]])

# Flags for plotting options
normalize_columns = False  # Set to True to normalize columns
individual_unit_plotting = False  # Set to True to enable individual unit plotting
plot_linear_regression = True  # Set to True to plot linear regression on unwrapped means

def limited_unwrap(phases, max_discontinuity=np.pi, limit=2*np.pi):
#def limited_unwrap(phases, max_discontinuity=np.pi, limit=1*np.pi):
    """
    Unwraps phase angles but limits the cumulative unwrapping to ±limit.

    Parameters:
    - phases: array-like, input phase angles in radians.
    - max_discontinuity: float, maximum allowed discontinuity between samples.
    - limit: float, maximum cumulative unwrapping in either direction.

    Returns:
    - unwrapped: array-like, phase angles after limited unwrapping.
    """
    #return np.unwrap(phases)
    unwrapped = np.copy(phases)
    cumulative_shift = 0.0

    for i in range(1, len(phases)):
        delta = phases[i] - phases[i - 1]
        # Wrap delta to the range [-max_discontinuity, max_discontinuity]
        delta = (delta + max_discontinuity) % (2 * max_discontinuity) - max_discontinuity
        
        # Check if applying this delta would exceed the limit
        if cumulative_shift + delta > limit:
            delta = limit - cumulative_shift
            cumulative_shift = limit
        elif cumulative_shift + delta < -limit:
            delta = -limit - cumulative_shift
            cumulative_shift = -limit
        else:
            cumulative_shift += delta
        
        unwrapped[i] = unwrapped[i - 1] + delta

    return unwrapped


import numpy as np

import numpy as np
from scipy.stats import circmean
from scipy.ndimage import gaussian_filter

#def compute_circmean_per_window(x_data, theta_data, window_size, step_size, num_bins=20, apply_smoothing=True, sigma=(1,1)):
def compute_circmean_per_window(x_data, theta_data, window_size, step_size, num_bins=20, apply_smoothing=True, sigma=(SIGMA,SIGMA)):
    """
    Computes the peak phase for each moving window by binning phases into specified bins,
    optionally applying 2D Gaussian smoothing to the histogram counts, and selecting the
    phase corresponding to the peak bin.

    Parameters:
    - x_data (np.ndarray): Time relative to scene onset (arbitrary spacing allowed).
    - theta_data (np.ndarray): Corresponding theta phases (in radians, range [-π, π]).
    - window_size (float): Width of each window (in the same units as x_data).
    - step_size (float): Step size between consecutive windows (in the same units as x_data).
    - num_bins (int): Number of phase bins to divide the phase range into (default is 10).
    - apply_smoothing (bool): Whether to apply 2D Gaussian smoothing to the histogram counts (default is False).
    - sigma (tuple or float): Standard deviation for Gaussian kernel. If a single float is provided,
                              it is used for both dimensions. Defaults to (1,1).

    Returns:
    - peak_phases (np.ndarray): Peak phases for each window.
    - window_ranges (list): List of tuples indicating the (start, end) time for each window.
    """
    # Define the phase range
    phase_low = -np.pi
    phase_high = np.pi
    bin_edges = np.linspace(phase_low, phase_high, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Initialize lists to store histograms and window ranges
    histograms = []
    window_ranges = []

    # Initialize the start of the first window
    start = np.min(x_data)

    # Iterate until the end of the data
    while start < np.max(x_data):
        end = start + window_size

        # Create a mask for data points within the current window
        window_mask = (x_data >= start) & (x_data < end)

        if np.any(window_mask):
            # Extract phases within the window
            window_phases = theta_data[window_mask]

            # Compute histogram
            counts, _ = np.histogram(window_phases, bins=bin_edges)
            histograms.append(counts)
        else:
            # If no data in the window, append zeros to maintain histogram shape
            histograms.append(np.zeros(num_bins))

        # Save the window range
        window_ranges.append((start, end))

        # Move the window forward by the step size
        start += step_size

    # Convert histograms list to a 2D NumPy array (windows x bins)
    hist_matrix = np.array(histograms)

    # Apply 2D Gaussian smoothing if enabled
    if apply_smoothing:
        hist_matrix = gaussian_filter(hist_matrix, sigma=sigma)

    # Initialize list to store peak phases
    peak_phases = []

    # Identify peak bin for each window
    for counts in hist_matrix:
        if np.all(counts == 0):
            # If no data in the window, append NaN
            peak_phases.append(np.nan)
        else:
            # Identify the peak bin
            peak_bin_index = np.argmax(counts)
            peak_phase = bin_centers[peak_bin_index]
            peak_phases.append(peak_phase)

    return np.array(peak_phases), window_ranges


def OLDcompute_circmean_per_window(x_data, theta_data, window_size, step_size, num_bins=20):
    """
    Computes the peak phase for each moving window by binning phases into specified bins
    and selecting the phase corresponding to the peak bin.

    Parameters:
    - x_data (np.ndarray): Time relative to scene onset (arbitrary spacing allowed).
    - theta_data (np.ndarray): Corresponding theta phases (in radians, range [-π, π]).
    - window_size (float): Width of each window (in the same units as x_data).
    - step_size (float): Step size between consecutive windows (in the same units as x_data).
    - num_bins (int): Number of phase bins to divide the phase range into (default is 10).

    Returns:
    - peak_phases (np.ndarray): Peak phases for each window.
    - window_ranges (list): List of tuples indicating the (start, end) time for each window.
    """
    peak_phases = []
    window_ranges = []

    # Define the phase range
    phase_low = -np.pi
    phase_high = np.pi
    bin_edges = np.linspace(phase_low, phase_high, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Initialize the start of the first window
    start = np.min(x_data)

    # Iterate until the end of the data
    while start < np.max(x_data):
        end = start + window_size

        # Create a mask for data points within the current window
        window_mask = (x_data >= start) & (x_data < end)

        if np.any(window_mask):
            # Extract phases within the window
            window_phases = theta_data[window_mask]

            # Compute histogram
            counts, _ = np.histogram(window_phases, bins=bin_edges)

            # Identify the peak bin
            peak_bin_index = np.argmax(counts)

            # Handle circularity: if the last bin is the peak, its center might wrap around
            peak_phase = bin_centers[peak_bin_index]

            peak_phases.append(peak_phase)
        else:
            # If no data in the window, append NaN
            peak_phases.append(np.nan)

        # Save the window range
        window_ranges.append((start, end))

        # Move the window forward by the step size
        start += step_size

    return np.array(peak_phases), window_ranges


def OLDcompute_circmean_per_window(x_data, theta_data, window_size, step_size):
    """
    Computes the circular mean for each moving window.

    Parameters:
    - x_data (np.ndarray): Time relative to scene onset (arbitrary spacing allowed).
    - theta_data (np.ndarray): Corresponding theta phases.
    - window_size (float): Width of each window (in the same units as x_data).
    - step_size (float): Step size between consecutive windows (in the same units as x_data).

    Returns:
    - circmeans (np.ndarray): Circular means for each window.
    - window_ranges (list): (start, end) time for each window.
    """
    circmeans = []
    window_ranges = []

    # Initialize the start of the first window
    start = np.min(x_data)

    # Iterate until the end of the data
    while start < np.max(x_data):
        end = start + window_size

        # Create a mask for data points within the current window
        window_mask = (x_data >= start) & (x_data < end)

        if np.any(window_mask):
            # Compute the circular mean for the current window
            cm = circmean(theta_data[window_mask], high=np.pi, low=-np.pi)
            circmeans.append(cm)
        else:
            # If no data in the window, append NaN
            circmeans.append(np.nan)

        # Save the window range
        window_ranges.append((start, end))

        # Move the window forward by the step size
        start += step_size

    return np.array(circmeans), window_ranges

def constrained_linear_regression(x,y):
    
    # Remove NaNs from x and y
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean = x[mask]
    y_clean = y[mask]

    # Calculate Pearson correlation coefficient
    correlation_matrix = np.corrcoef(x_clean, y_clean)
    r_value = correlation_matrix[0, 1]

    return np.nan,np.nan, r_value

def OLDconstrained_linear_regression(x, y):
    """
    Performs linear regression with the constraint that the slope is negative.
    Returns the slope, intercept, and r_value.
    """
    # Remove NaNs from x and y
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    if len(x) == 0 or len(y) == 0:
        return np.nan, np.nan, np.nan

    # Define the loss function (sum of squared residuals)
    def loss(params):
        slope, intercept = params
        residuals = y - (slope * x + intercept)
        return np.sum(residuals**2)

    # Initial guess for slope and intercept
    slope_guess, intercept_guess = np.polyfit(x, y, 1)
    if slope_guess > 0:
        slope_guess = -slope_guess  # Ensure initial slope is negative

    # Bounds: slope <= 0
    bounds = [(-10, 0), (-10, 10)]

    # Perform minimization with bounds
    res = minimize(loss, x0=[slope_guess, intercept_guess], bounds=bounds)

    # Extract results
    slope, intercept = res.x

    # Compute R-squared
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    if ss_tot == 0:
        r_squared = np.nan  # Can't compute R-squared
    else:
        r_squared = 1 - ss_res / ss_tot
    r_value = np.sqrt(r_squared)

    if slope<0:
        r_value*=-1
    return slope, intercept, r_value

def compute_average_error(x_data, theta_data, s_fit, b_fit, window_size, step_size):
    """
    Computes the average error between the observed phases and the fitted line.

    Returns:
    - avg_error: Average error in radians.
    """
    circmeans, _ = compute_circmean_per_window(x_data, theta_data, window_size, step_size)
    predicted_phases = s_fit * x_data + b_fit
    errors = np.abs(theta_data - predicted_phases)
    avg_error = np.mean(errors)
    return avg_error, circmeans, predicted_phases

# Define the circular Gaussian filter function
def circular_gaussian_filter(matrix, sigma=1.5, normalize=False):
    """
    Applies a Gaussian filter to a 2D matrix with circularity on the y-axis.
    Optionally peak normalizes each column by its maximum value.

    Parameters:
    - matrix (np.ndarray): The input 2D array to be filtered.
    - sigma (float): The standard deviation for Gaussian kernel. Default is 1.5.
    - normalize (bool): If True, peak normalizes each column after filtering. Default is False.

    Returns:
    - np.ndarray: The filtered (and optionally normalized) 2D array with circular y-axis handling.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Input matrix must be a NumPy array.")
    
    if matrix.ndim != 2:
        raise ValueError("Input matrix must be 2-dimensional.")
    
    # Vertically stack the matrix three times to handle circular y-axis
    stacked = np.vstack([matrix, matrix, matrix])
    
    # Apply Gaussian filter to the stacked matrix
    # Using mode='nearest' to handle boundaries within the stacked matrix
    filtered_stacked = gaussian_filter(stacked, sigma=sigma, mode='nearest')
    
    # Extract the central part corresponding to the original matrix
    original_rows = matrix.shape[0]
    start_row = original_rows
    end_row = 2 * original_rows
    filtered_central = filtered_stacked[start_row:end_row, :]
    
    if normalize:
        # Compute the mean value for each column
        column_mean = filtered_central.mean(axis=0)
        
        # To avoid division by zero, set zeros to one temporarily
        column_mean_safe = np.where(column_mean == 0, 1, column_mean)
        
        # Divide each column by its mean value
        filtered_central = filtered_central / column_mean_safe
        
        # Optionally, set columns that originally had a mean of zero back to zero
        zero_mean_columns = column_mean == 0
        if np.any(zero_mean_columns):
            filtered_central[:, zero_mean_columns] = 0
        
    return filtered_central

# Define the heatmap plotting function with regression
def plot_heatmap(
    heatmap_first, heatmap_last, key,
    suptitle=None,
    normalize_columns=False,
    xedges=None, yedges=None,
    condition='random', region_category='HPC',
    unit_counts=None,
    lowcut=4.0, highcut=12.0,
    filterSettingStr='default',
    N_TRIALS_PER_PHASE=50,
    output_dir='output',
    CMAP='viridis',
    VMIN_cntrl=0, VMAX_cntrl=100,
    VMIN=0, VMAX=200,
    TIME_DISP_MIN=-2, TIME_DISP_MAX=2,
    PHASE_DISP_MIN=-3*np.pi, PHASE_DISP_MAX=3*np.pi,
    cumulative_relative_times_first_key=None,
    cumulative_phases_first_key=None,
    cumulative_relative_times_last_key=None,
    cumulative_phases_last_key=None,
    time_bin_centers=None,
    num_time_bins=100,
    plot_linear_regression=False,
    UNWRAP_MEAN=False,
    circular_gaussian_filter=None,
    metric_choice='circular_linear_regression_rho',
    delta_metric=None,
    observed_metric=None,
    p_value=None
):
    """
    Plots and saves cumulative or individual heatmaps for the first and last trials,
    including regression analysis based on the selected metric.
    """
    if heatmap_last is None:
        print(f"Heatmap for the last trials is None for key {key}. Skipping.")
        return

    if unit_counts is None:
        unit_counts = {}

    if circular_gaussian_filter is None:
        from scipy.ndimage import gaussian_filter
        circular_gaussian_filter = gaussian_filter

    # Optionally normalize each column by its sum to get the distribution of phases given time
    if normalize_columns:
        # Normalize first trials
        column_sums_first = heatmap_first.sum(axis=0)
        column_sums_first[column_sums_first == 0] = 1
        heatmap_first_normalized = heatmap_first / column_sums_first[np.newaxis, :]

        # Normalize last trials
        column_sums_last = heatmap_last.sum(axis=0)
        column_sums_last[column_sums_last == 0] = 1
        heatmap_last_normalized = heatmap_last / column_sums_last[np.newaxis, :]
    else:
        heatmap_first_normalized = heatmap_first.copy()
        heatmap_last_normalized = heatmap_last.copy()

    # -------- Additional Smoothing -------- #
    # Apply Gaussian smoothing to the heatmaps
    #heatmap_first_normalized = circular_gaussian_filter(heatmap_first_normalized, sigma=1.5)
    #heatmap_last_normalized = circular_gaussian_filter(heatmap_last_normalized, sigma=1.5)
    heatmap_first_normalized = circular_gaussian_filter(heatmap_first_normalized, sigma=SIGMA*1.5) #extra smooth for display
    heatmap_last_normalized = circular_gaussian_filter(heatmap_last_normalized, sigma=SIGMA*1.5)
    # -------------------------------------- #

    # Duplicate the heatmaps in the phase axis to handle circularity
    extended_heatmap_first = np.concatenate([
        heatmap_first_normalized, 
        heatmap_first_normalized, 
        heatmap_first_normalized
    ], axis=0)
    extended_heatmap_last = np.concatenate([
        heatmap_last_normalized, 
        heatmap_last_normalized, 
        heatmap_last_normalized
    ], axis=0)
    extended_yedges = np.concatenate([yedges - 2 * np.pi, yedges, yedges + 2 * np.pi])

    if VMIN is not None:
        if region_category != 'HPC':
            vmin = VMIN_cntrl
            vmax = VMAX_cntrl
            vmaxFIRST = vmax

            if condition == 'random':
                vmin = 40
                vmax = 70
                vmaxFIRST = vmax
        else:
            vmin = VMIN
            vmax = VMAX
            vmaxFIRST = vmax
            if condition == 'structured':
                vmin = 122
                vmax = 145
                vmaxFIRST = 155
    else:
        vmin=None
        vmax=None
        vmaxFIRST=None

    # Create a figure with two subplots
    #fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig, axs = plt.subplots(1, 2, figsize=(8, 6), sharey=True)

    # Define the extent for imshow
    extent_plot = [xedges[0], xedges[-1], extended_yedges[0], extended_yedges[-1]]

    # First subplot: First N_TRIALS_PER_PHASE trials
    im1 = axs[0].imshow(
        extended_heatmap_first, 
        aspect='auto', 
        origin='lower', 
        extent=extent_plot, 
        cmap=CMAP,
        vmin=0.02 if normalize_columns else vmin,
        vmax=0.075 if normalize_columns else vmaxFIRST
    )
    axs[0].set_title(f"First {N_TRIALS_PER_PHASE} Trials")
    axs[0].set_xlabel('Time Relative to Scene Onset (s)')
    axs[0].set_ylabel('Spike Phase (radians)')
    axs[0].set_yticks(
        [-3 * np.pi, -2.5 * np.pi, -2 * np.pi, -1.5 * np.pi, -np.pi, -0.5 * np.pi, 0,
         0.5 * np.pi, np.pi, 1.5 * np.pi, 2 * np.pi, 2.5 * np.pi, 3 * np.pi]
    )
    axs[0].set_yticklabels(
        [r'$-3\pi$', r'$-2.5\pi$', r'$-2\pi$', r'$-1.5\pi$', r'$-\pi$', r'$-0.5\pi$', '0',
         r'$0.5\pi$', r'$\pi$', r'$1.5\pi$', r'$2\pi$', r'$2.5\pi$', r'$3\pi$']
    )
    # Set axis limits
    axs[0].set_xlim(TIME_DISP_MIN, TIME_DISP_MAX)
    axs[0].set_ylim(PHASE_DISP_MIN, PHASE_DISP_MAX)
    axs[0].axvline(0, color='k', linestyle='--', linewidth=BOUND_LINE_WIDTH)
    axs[0].axhline(-np.pi, color='k', linestyle='--', linewidth=BOUND_LINE_WIDTH)
    axs[0].axhline(np.pi, color='k', linestyle='--', linewidth=BOUND_LINE_WIDTH)

    # Second subplot: Last N_TRIALS_PER_PHASE trials
    im2 = axs[1].imshow(
        extended_heatmap_last, 
        aspect='auto', 
        origin='lower', 
        extent=extent_plot, 
        cmap=CMAP,
        vmin=0.02 if normalize_columns else vmin,
        vmax=0.075 if normalize_columns else vmax
    )
    axs[1].set_title(f"Last {N_TRIALS_PER_PHASE} Trials")
    axs[1].set_xlabel('Time Relative to Scene Onset (s)')
    axs[1].set_yticks(
        [-3 * np.pi, -2.5 * np.pi, -2 * np.pi, -1.5 * np.pi, -np.pi, -0.5 * np.pi, 0,
         0.5 * np.pi, np.pi, 1.5 * np.pi, 2 * np.pi, 2.5 * np.pi, 3 * np.pi]
    )
    axs[1].set_yticklabels(
        [r'$-3\pi$', r'$-2.5\pi$', r'$-2\pi$', r'$-1.5\pi$', r'$-\pi$', r'$-0.5\pi$', '0',
         r'$0.5\pi$', r'$\pi$', r'$1.5\pi$', r'$2\pi$', r'$2.5\pi$', r'$3\pi$']
    )
    # Set axis limits
    axs[1].set_xlim(TIME_DISP_MIN, TIME_DISP_MAX)
    axs[1].set_ylim(PHASE_DISP_MIN, PHASE_DISP_MAX)
    axs[1].axvline(0, color='k', linestyle='--', linewidth=BOUND_LINE_WIDTH)
    axs[1].axhline(-np.pi, color='k', linestyle='--', linewidth=BOUND_LINE_WIDTH)
    axs[1].axhline(np.pi, color='k', linestyle='--', linewidth=BOUND_LINE_WIDTH)

    # -------- Perform Regression Based on Change in Phase-Time Correlation Choice -------- #
    if plot_linear_regression:
        # Mask spikes within the display time range
        mask_spikes_first = (cumulative_relative_times_first_key >= TIME_DISP_MIN) & (cumulative_relative_times_first_key <= TIME_DISP_MAX)
        mask_spikes_last = (cumulative_relative_times_last_key >= TIME_DISP_MIN) & (cumulative_relative_times_last_key <= TIME_DISP_MAX)

        # For first trials
        x_first = cumulative_relative_times_first_key[mask_spikes_first]
        theta_first = cumulative_phases_first_key[mask_spikes_first]
        
        # Compute circmean per window
        circmean_first_bins, window_ranges_first = compute_circmean_per_window(
            x_data=x_first,
            theta_data=theta_first,
            window_size=window_size,
            step_size=step_size
        )
        window_centers_first = np.array([(start + end) / 2 for start, end in window_ranges_first])

        # For last trials
        x_last = cumulative_relative_times_last_key[mask_spikes_last]
        theta_last = cumulative_phases_last_key[mask_spikes_last]
        
        # Compute circmean per window
        circmean_last_bins, window_ranges_last = compute_circmean_per_window(
            x_data=x_last,
            theta_data=theta_last,
            window_size=window_size,
            step_size=step_size
        )
        window_centers_last = np.array([(start + end) / 2 for start, end in window_ranges_last])

        if metric_choice == 'unwrapped_linear_regression':
            # Unwrap the circular means
            #unwrapped_circmean_first_bins = np.unwrap(circmean_first_bins)
            #unwrapped_circmean_last_bins = np.unwrap(circmean_last_bins)
            unwrapped_circmean_first_bins = limited_unwrap(circmean_first_bins)
            unwrapped_circmean_last_bins = limited_unwrap(circmean_last_bins)

            # Perform constrained linear regression
            slope_first, intercept_first, r_value_first = constrained_linear_regression(window_centers_first, unwrapped_circmean_first_bins)
            slope_last, intercept_last, r_value_last = constrained_linear_regression(window_centers_last, unwrapped_circmean_last_bins)

            # Define regression lines
            x_fit_first = np.linspace(TIME_DISP_MIN, TIME_DISP_MAX, 100)
            theta_fit_first = slope_first * x_fit_first + intercept_first

            x_fit_last = np.linspace(TIME_DISP_MIN, TIME_DISP_MAX, 100)
            theta_fit_last = slope_last * x_fit_last + intercept_last

            # Plot unwrapped circular means
            axs[0].plot(window_centers_first, unwrapped_circmean_first_bins, color='black', linestyle='--', linewidth=3, label='Unwrapped Peak Phase')
            axs[1].plot(window_centers_last, unwrapped_circmean_last_bins, color='black', linestyle='--', linewidth=3, label='Unwrapped Peak Phase')

            # Plot regression lines
            axs[0].plot(x_fit_first, theta_fit_first, color='black', linewidth=2)
            axs[1].plot(x_fit_last, theta_fit_last, color='black', linewidth=2)

            # Update titles
            axs[0].set_title(f"First {N_TRIALS_PER_PHASE} Trials\nr: {r_value_first:.3f}")
            axs[1].set_title(f"Last {N_TRIALS_PER_PHASE} Trials\nr: {r_value_last:.3f}")

        else:
            # Perform circular-linear regression using Kempter's method
            slope_bounds = [MIN_SLOPE, MAX_SLOPE]  # Constrain slope to be negative
            rho_first, p_first, s_fit_first, b_first = kempter.kempter_lincircTJ_slopeBounds(
                x=window_centers_first,
                theta=circmean_first_bins,
                slopeBounds=slope_bounds
            )
            rho_last, p_last, s_fit_last, b_last = kempter.kempter_lincircTJ_slopeBounds(
                x=window_centers_last,
                theta=circmean_last_bins,
                slopeBounds=slope_bounds
            )

            # Compute predicted phases
            theta_fit_first = s_fit_first * window_centers_first + b_first
            theta_fit_last = s_fit_last * window_centers_last + b_last

            # Plot circular means
            axs[0].plot(window_centers_first, circmean_first_bins, color='red', linestyle='--', linewidth=2, label='Circular Mean Phase')
            axs[1].plot(window_centers_last, circmean_last_bins, color='red', linestyle='--', linewidth=2, label='Circular Mean Phase')

            # Plot regression lines
            axs[0].plot(window_centers_first, theta_fit_first, color='black', linewidth=2, label=f'rho: {rho_first:.3f}')
            axs[1].plot(window_centers_last, theta_fit_last, color='black', linewidth=2, label=f'rho: {rho_last:.3f}')

            # Update titles based on metric
            if metric_choice == 'circular_linear_regression_rho':
                axs[0].set_title(f"First {N_TRIALS_PER_PHASE} Trials\nrho: {rho_first:.3f}")
                axs[1].set_title(f"Last {N_TRIALS_PER_PHASE} Trials\nrho: {rho_last:.3f}")
            elif metric_choice == 'circular_linear_regression_error':
                # Compute average errors
                avg_error_first, _, _ = compute_average_error(
                    x_data=x_first,
                    theta_data=theta_first,
                    s_fit=s_fit_first,
                    b_fit=b_first,
                    window_size=window_size,
                    step_size=step_size
                )
                avg_error_last, _, _ = compute_average_error(
                    x_data=x_last,
                    theta_data=theta_last,
                    s_fit=s_fit_last,
                    b_fit=b_last,
                    window_size=window_size,
                    step_size=step_size
                )
                axs[0].set_title(f"First {N_TRIALS_PER_PHASE} Trials\nAvg Error: {avg_error_first:.3f}")
                axs[1].set_title(f"Last {N_TRIALS_PER_PHASE} Trials\nAvg Error: {avg_error_last:.3f}")
            elif metric_choice == 'circular_linear_regression_slope':
                axs[0].set_title(f"First {N_TRIALS_PER_PHASE} Trials\nSlope: {s_fit_first:.3f}")
                axs[1].set_title(f"Last {N_TRIALS_PER_PHASE} Trials\nSlope: {s_fit_last:.3f}")

        # Annotate p-value if available
        if p_value is not None:
            axs[1].annotate(f'p-value: {p_value:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
                            fontsize=12, horizontalalignment='left', verticalalignment='top')

        # Update legends
        axs[0].legend(loc='upper right')
        axs[1].legend(loc='upper right')
    # ----------------------------------------------- #

    # Add colorbars
    cbar1 = plt.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)
    cbar1.set_label('Normalized Spike Count' if normalize_columns else 'Cumulative Spike Count')
    cbar2 = plt.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)
    cbar2.set_label('Normalized Spike Count' if normalize_columns else 'Cumulative Spike Count')

    # Add suptitle with the number of units or custom suptitle
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=12)
    else:
        num_units = unit_counts.get(key, 'N/A')
        fig.suptitle(
            f"{lowcut}-{highcut}Hz Theta Phase Coding: {condition}, {region_category}, "
            f"N = {num_units} visually selective units", 
            fontsize=12
        )

    # Adjust layout to make room for suptitle
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the cumulative plot to a file
    if suptitle is not None:
        # For individual unit plots
        cumulative_plot_filename = (
            f"{suptitle.replace(' ', '_')}_heatmap.pdf"
        )
    else:
        # For cumulative plots
        cumulative_plot_filename = (
            f"cumulative_spike_phase_time_heatmap_condition_{condition}_"
            f"region_{region_category}_{filterSettingStr}_{N_TRIALS_PER_PHASE}trialsPerPhase.pdf"
        )
    # Clean the filename
    cumulative_plot_filename = cumulative_plot_filename.replace(':', '').replace('/', '_').replace('\\', '_').replace(',', '_').replace('?', '_')
    cumulative_plot_filepath = os.path.join(output_dir, cumulative_plot_filename)
    plt.savefig(cumulative_plot_filepath)
    plt.close()
    print(f"Saved heatmap for key '{key}' to '{cumulative_plot_filepath}'")

# Function to build heatmap
def build_heatmap(relative_spike_times, corresponding_phases, heatmap):
    """
    Builds a heatmap by applying a continuous Gaussian distribution to each spike's time and phase,
    handling circular wraparound in phase by creating shifted copies of the phase data.

    Parameters:
    - relative_spike_times: Array-like, spike times relative to an event (in seconds).
    - corresponding_phases: Array-like, spike phases in radians, ranging from -pi to pi.
    - heatmap: 2D NumPy array to accumulate the heatmap data (shape: [num_phase_bins, num_time_bins]).

    Returns:
    - heatmap: Updated 2D NumPy array with accumulated Gaussian contributions.
    """
    # Calculate average frequency and corresponding cycle period
    avg_freq = (lowcut + highcut) / 2.0  # in Hz
    cycle_period = 1.0 / avg_freq       # in seconds

    # Define resolutions based on bin sizes
    bin_size_time = 0.5 * (xedges[1] - xedges[0])      # e.g., 0.3 seconds
    bin_size_phase = 1.5 * (yedges[1] - yedges[0])    # e.g., 0.2618 radians (~15 degrees)

    # Define Gaussian standard deviations
    sigma_time = bin_size_time / 2.0           # 0.15 seconds
    sigma_phase = bin_size_phase / 2.0         # ~0.1309 radians

    # Define the range for the Gaussian (±3 sigma)
    time_range_gauss = 3 * sigma_time                 # 0.45 seconds
    phase_range_gauss = 3 * sigma_phase               # ~0.3927 radians

    # Precompute normalization factor for the Gaussian
    gaussian_norm = 1.0 / (2.0 * np.pi * sigma_phase * sigma_time)

    # Iterate over each spike
    for spike_time, spike_phase in zip(relative_spike_times, corresponding_phases):
        # Normalize spike phase to be within [-pi, pi]
        spike_phase = (spike_phase + np.pi) % (2 * np.pi) - np.pi

        # Create shifted copies of the phase for wraparound
        shifted_phases = np.array([spike_phase - 2 * np.pi, spike_phase, spike_phase + 2 * np.pi])

        for shifted_phase in shifted_phases:
            # Define the bounds for the Gaussian in time and phase
            time_min = spike_time - time_range_gauss
            time_max = spike_time + time_range_gauss
            phase_min = shifted_phase - phase_range_gauss
            phase_max = shifted_phase + phase_range_gauss

            # Find the indices of the time bins that fall within the Gaussian range
            time_indices = np.where((time_bin_centers >= time_min) & (time_bin_centers <= time_max))[0]

            # Find the indices of the phase bins that fall within the Gaussian range
            phase_indices = np.where((phase_bin_centers >= phase_min) & (phase_bin_centers <= phase_max))[0]

            # If no bins are within the range, skip to the next iteration
            if len(time_indices) == 0 or len(phase_indices) == 0:
                continue

            # Extract the relevant bin centers
            relevant_time_centers = time_bin_centers[time_indices]
            relevant_phase_centers = phase_bin_centers[phase_indices]

            # Compute the differences
            delta_t = relevant_time_centers[:, np.newaxis] - spike_time      # Shape: [num_time_bins, 1]
            delta_p = relevant_phase_centers - shifted_phase                  # Shape: [num_phase_bins]

            # Compute Gaussian contributions
            exponent = -((delta_p ** 2) / (2 * sigma_phase ** 2) + (delta_t ** 2) / (2 * sigma_time ** 2))
            contributions = gaussian_norm * np.exp(exponent)                 # Shape: [num_time_bins, num_phase_bins]

            # Accumulate the contributions into the heatmap
            # Note: heatmap shape is [num_phase_bins, num_time_bins], so transpose contributions
            heatmap[np.ix_(phase_indices, time_indices)] += contributions.T

    return heatmap

# Loop through each unit in units_df
for idx, unit_data in units_df.iterrows():
    # Extract necessary data
    spike_phases = unit_data['spike_phases']  # List of spike phases
    spike_times = unit_data['spike_times']    # List of spike times (absolute times in experiment)
    time_bounds_per_scene_num = unit_data['time_bounds_per_scene_num']  # Dictionary of time bounds per scene number
    initial_peak_scene_num = unit_data.get('initial_peak_scene_num', None)  # Preferred peak scene number

    # Check if initial_peak_scene_num is available
    if initial_peak_scene_num is None or pd.isna(initial_peak_scene_num):
        print(f"Unit {unit_data['unit_id']} has no initial_peak_scene_num. Skipping.")
        continue

    # Ensure that time_bounds_per_scene_num is a dictionary
    if isinstance(time_bounds_per_scene_num, str):
        try:
            time_bounds_per_scene_num = eval(time_bounds_per_scene_num)
        except:
            print(f"Could not parse time_bounds_per_scene_num for unit {unit_data['unit_id']}. Skipping.")
            continue

    # Ensure that spike_phases and spike_times are arrays
    if isinstance(spike_phases, str):
        try:
            spike_phases = np.array(eval(spike_phases))
        except:
            print(f"Could not parse spike_phases for unit {unit_data['unit_id']}. Skipping.")
            continue
    else:
        spike_phases = np.array(spike_phases)
    if isinstance(spike_times, str):
        try:
            spike_times = np.array(eval(spike_times))
        except:
            print(f"Could not parse spike_times for unit {unit_data['unit_id']}. Skipping.")
            continue
    else:
        spike_times = np.array(spike_times)

    # Get the time bounds for the preferred scene number
    preferred_scene_trials = time_bounds_per_scene_num.get(initial_peak_scene_num, [])

    # Check if there are enough trials
    if len(preferred_scene_trials) < N_TRIALS_PER_PHASE:
        print(f"Not enough trials for unit {unit_data['unit_id']}. Skipping.")
        continue

    # Split into first and last N_TRIALS_PER_PHASE trials
    preferred_scene_trials_first = preferred_scene_trials[:N_TRIALS_PER_PHASE]
    preferred_scene_trials_last = preferred_scene_trials[-N_TRIALS_PER_PHASE:]

    # Process first N_TRIALS_PER_PHASE trials
    relative_spike_times_first = []
    corresponding_phases_first = []

    for trial_bounds in preferred_scene_trials_first:
        trial_start, trial_end = trial_bounds

        trial_spike_mask = (spike_times >= trial_start - TIME_BEFORE) & (spike_times <= trial_start + TIME_AFTER)
        trial_spike_times = spike_times[trial_spike_mask]
        trial_spike_phases = spike_phases[trial_spike_mask]

        trial_relative_times = trial_spike_times - trial_start

        relative_spike_times_first.extend(trial_relative_times)
        corresponding_phases_first.extend(trial_spike_phases)

    # Process last N_TRIALS_PER_PHASE trials
    relative_spike_times_last = []
    corresponding_phases_last = []

    for trial_bounds in preferred_scene_trials_last:
        trial_start, trial_end = trial_bounds

        trial_spike_mask = (spike_times >= trial_start - TIME_BEFORE) & (spike_times <= trial_start + TIME_AFTER)
        trial_spike_times = spike_times[trial_spike_mask]
        trial_spike_phases = spike_phases[trial_spike_mask]

        trial_relative_times = trial_spike_times - trial_start

        relative_spike_times_last.extend(trial_relative_times)
        corresponding_phases_last.extend(trial_spike_phases)

    # Convert lists to arrays
    relative_spike_times_first = np.array(relative_spike_times_first)
    corresponding_phases_first = np.array(corresponding_phases_first)

    relative_spike_times_last = np.array(relative_spike_times_last)
    corresponding_phases_last = np.array(corresponding_phases_last)

    # Check if there are any spikes to plot
    if len(relative_spike_times_first) == 0 or len(relative_spike_times_last) == 0:
        print(f"No spikes found for unit {unit_data['unit_id']} during preferred scene trials. Skipping.")
        continue

    # Wrap phases to [-pi, pi]
    corresponding_phases_first = (corresponding_phases_first + np.pi) % (2 * np.pi) - np.pi
    corresponding_phases_last = (corresponding_phases_last + np.pi) % (2 * np.pi) - np.pi

    # Initialize heatmaps for this unit
    heatmap_first = np.zeros((num_phase_bins, num_time_bins))
    heatmap_last = np.zeros((num_phase_bins, num_time_bins))

    # Build heatmaps
    heatmap_first = build_heatmap(relative_spike_times_first, corresponding_phases_first, heatmap_first)
    heatmap_last = build_heatmap(relative_spike_times_last, corresponding_phases_last, heatmap_last)

    # Accumulate the data for cumulative plots
    condition = unit_data['condition']
    region_category = unit_data.get('Region_Category', '')  # Use 'Region_Category' or 'region_category' as available
    # Ensure region_category is either 'HPC' or 'non-HPC'
    if region_category != 'HPC':
        region_category = 'non-HPC'

    key = (condition, region_category)

    # Update unit_counts
    if key not in unit_counts:
        unit_counts[key] = 1
    else:
        unit_counts[key] += 1

    if key not in cumulative_heatmaps_first:
        cumulative_heatmaps_first[key] = heatmap_first.copy()
        cumulative_relative_times_first[key] = relative_spike_times_first.copy()
        cumulative_phases_first[key] = corresponding_phases_first.copy()
    else:
        cumulative_heatmaps_first[key] += heatmap_first
        cumulative_relative_times_first[key] = np.concatenate([cumulative_relative_times_first[key], relative_spike_times_first])
        cumulative_phases_first[key] = np.concatenate([cumulative_phases_first[key], corresponding_phases_first])

    if key not in cumulative_heatmaps_last:
        cumulative_heatmaps_last[key] = heatmap_last.copy()
        cumulative_relative_times_last[key] = relative_spike_times_last.copy()
        cumulative_phases_last[key] = corresponding_phases_last.copy()
    else:
        cumulative_heatmaps_last[key] += heatmap_last
        cumulative_relative_times_last[key] = np.concatenate([cumulative_relative_times_last[key], relative_spike_times_last])
        cumulative_phases_last[key] = np.concatenate([cumulative_phases_last[key], corresponding_phases_last])

    # Collect individual unit data into the unit pool for last trials
    unit_pool_last.append({
        'heatmap': heatmap_last,
        'relative_times': relative_spike_times_last,
        'phases': corresponding_phases_last
    })

    # Collect individual unit data into the unit pool for last trials
    unit_pool_last.append({
        'heatmap': heatmap_first,
        'relative_times': relative_spike_times_first,
        'phases': corresponding_phases_first
    })

    # Optional individual unit plotting
    if individual_unit_plotting:
        # Prepare additional metadata for the suptitle
        lfp_filepath = unit_data['lfp_filepath']
        lfp_basename = os.path.basename(lfp_filepath)
        unit_id = unit_data['unit_id']
        sessIDStr = unit_data.get('sessIDStr', '')
        region_name = unit_data.get('region_name', '')
        specific_region_name = unit_data.get('specific_region_name', '')
        subfield = unit_data.get('subfield', '')
        chNameStr = unit_data.get('chNameStr', '')
        unitNumStr = unit_data.get('unitNumStr', '')

        # Construct the suptitle with relevant metadata
        individual_suptitle = (
            f"Unit {unit_id}, Scene {initial_peak_scene_num}, Condition: {condition}\n"
            f"LFP File: {lfp_basename}, Session: {sessIDStr}, "
            f"Region: {region_name}, Subfield: {subfield}"
        )

        # Call the plot_heatmap function for individual unit
        plot_heatmap(
            heatmap_first=heatmap_first,
            heatmap_last=heatmap_last,
            key=unit_id,  # Using unit_id as the key identifier for individual plots
            suptitle=individual_suptitle,
            normalize_columns=normalize_columns,
            xedges=xedges,
            yedges=yedges,
            condition=condition,
            region_category=region_category,
            unit_counts=unit_counts,
            lowcut=lowcut,
            highcut=highcut,
            filterSettingStr=filterSettingStr,
            N_TRIALS_PER_PHASE=N_TRIALS_PER_PHASE,
            output_dir=output_dir,
            CMAP=CMAP,
            VMIN_cntrl=None,
            VMAX_cntrl=None,
            VMIN=None,
            VMAX=None,
            TIME_DISP_MIN=TIME_DISP_MIN,
            TIME_DISP_MAX=TIME_DISP_MAX,
            PHASE_DISP_MIN=PHASE_DISP_MIN,
            PHASE_DISP_MAX=PHASE_DISP_MAX,
            cumulative_relative_times_first_key=relative_spike_times_first,
            cumulative_phases_first_key=corresponding_phases_first,
            cumulative_relative_times_last_key=relative_spike_times_last,
            cumulative_phases_last_key=corresponding_phases_last,
            time_bin_centers=time_bin_centers,
            num_time_bins=num_time_bins,
            plot_linear_regression=plot_linear_regression,  # Enable linear regression plotting
            UNWRAP_MEAN=UNWRAP_MEAN,
            circular_gaussian_filter=circular_gaussian_filter,
            metric_choice=metric_choice
        )

# After processing all units and accumulating cumulative data, perform regression analysis per category

for key in cumulative_heatmaps_first.keys():
    # Get cumulative data
    cumulative_relative_times_first_key = cumulative_relative_times_first[key]
    cumulative_phases_first_key = cumulative_phases_first[key]
    cumulative_relative_times_last_key = cumulative_relative_times_last[key]
    cumulative_phases_last_key = cumulative_phases_last[key]

    # Mask spikes within the display time range
    mask_spikes_first = (cumulative_relative_times_first_key >= TIME_DISP_MIN) & (cumulative_relative_times_first_key <= TIME_DISP_MAX)
    mask_spikes_last = (cumulative_relative_times_last_key >= TIME_DISP_MIN) & (cumulative_relative_times_last_key <= TIME_DISP_MAX)

    # For first trials
    x_first = cumulative_relative_times_first_key[mask_spikes_first]
    theta_first = cumulative_phases_first_key[mask_spikes_first]

    # Compute circmean per window
    circmean_first_bins, window_ranges_first = compute_circmean_per_window(
        x_data=x_first,
        theta_data=theta_first,
        window_size=window_size,
        step_size=step_size
    )
    window_centers_first = np.array([(start + end) / 2 for start, end in window_ranges_first])

    # For last trials
    x_last = cumulative_relative_times_last_key[mask_spikes_last]
    theta_last = cumulative_phases_last_key[mask_spikes_last]

    # Compute circmean per window
    circmean_last_bins, window_ranges_last = compute_circmean_per_window(
        x_data=x_last,
        theta_data=theta_last,
        window_size=window_size,
        step_size=step_size
    )
    window_centers_last = np.array([(start + end) / 2 for start, end in window_ranges_last])

    if metric_choice == 'unwrapped_linear_regression':
        # Unwrap the circular means
        #unwrapped_circmean_first_bins = np.unwrap(circmean_first_bins)
        #unwrapped_circmean_last_bins = np.unwrap(circmean_last_bins)
        unwrapped_circmean_first_bins = limited_unwrap(circmean_first_bins)
        unwrapped_circmean_last_bins = limited_unwrap(circmean_last_bins)

        # Perform constrained linear regression
        slope_first, intercept_first, r_value_first = constrained_linear_regression(window_centers_first, unwrapped_circmean_first_bins)
        slope_last, intercept_last, r_value_last = constrained_linear_regression(window_centers_last, unwrapped_circmean_last_bins)

        # Compute delta metric (e.g., delta_r_value)
        #delta_metric_value = r_value_last  # Using r_value_last as the metric
        delta_metric_value = r_value_last - r_value_first  # Using r_value_last as the metric
    else:
        # Perform circular-linear regression using Kempter's method
        slope_bounds = [MIN_SLOPE, MAX_SLOPE]  # Constrain slope to be negative
        rho_first, p_first, s_fit_first, b_first = kempter.kempter_lincircTJ_slopeBounds(
            x=window_centers_first,
            theta=circmean_first_bins,
            slopeBounds=slope_bounds
        )
        rho_last, p_last, s_fit_last, b_last = kempter.kempter_lincircTJ_slopeBounds(
            x=window_centers_last,
            theta=circmean_last_bins,
            slopeBounds=slope_bounds
        )

        if metric_choice == 'circular_linear_regression_rho':
            delta_metric_value = rho_last  # Using rho_last as the metric
        elif metric_choice == 'circular_linear_regression_error':
            # Compute average errors
            avg_error_first, _, _ = compute_average_error(
                x_data=x_first,
                theta_data=theta_first,
                s_fit=s_fit_first,
                b_fit=b_first,
                window_size=window_size,
                step_size=step_size
            )
            avg_error_last, _, _ = compute_average_error(
                x_data=x_last,
                theta_data=theta_last,
                s_fit=s_fit_last,
                b_fit=b_last,
                window_size=window_size,
                step_size=step_size
            )
            delta_metric_value = avg_error_last - avg_error_first  # Change in error
        elif metric_choice == 'circular_linear_regression_slope':
            delta_metric_value = s_fit_last  # Using slope as the metric

    # Store delta_metric_value in delta_metrics[key]
    delta_metrics[key] = delta_metric_value

    # Store observed metric
    observed_metrics_last[key] = delta_metric_value

    # Plot cumulative heatmaps with regression
    plot_heatmap(
        heatmap_first=cumulative_heatmaps_first[key],
        heatmap_last=cumulative_heatmaps_last[key],
        key=key,
        normalize_columns=normalize_columns,
        xedges=xedges,
        yedges=yedges,
        condition=key[0],
        region_category=key[1],
        unit_counts=unit_counts,
        lowcut=lowcut,
        highcut=highcut,
        filterSettingStr=filterSettingStr,
        N_TRIALS_PER_PHASE=N_TRIALS_PER_PHASE,
        output_dir=output_dir,
        CMAP=CMAP,
        VMIN_cntrl=VMIN_cntrl,
        VMAX_cntrl=VMAX_cntrl,
        VMIN=VMIN,
        VMAX=VMAX,
        TIME_DISP_MIN=TIME_DISP_MIN,
        TIME_DISP_MAX=TIME_DISP_MAX,
        PHASE_DISP_MIN=PHASE_DISP_MIN,
        PHASE_DISP_MAX=PHASE_DISP_MAX,
        cumulative_relative_times_first_key=cumulative_relative_times_first[key],
        cumulative_phases_first_key=cumulative_phases_first[key],
        cumulative_relative_times_last_key=cumulative_relative_times_last[key],
        cumulative_phases_last_key=cumulative_phases_last[key],
        time_bin_centers=time_bin_centers,
        num_time_bins=num_time_bins,
        plot_linear_regression=plot_linear_regression,  # Enable linear regression plotting
        UNWRAP_MEAN=UNWRAP_MEAN,
        circular_gaussian_filter=circular_gaussian_filter,
        metric_choice=metric_choice,
        observed_metric=delta_metric_value,
        p_value=None  # Will be updated later
    )

# After processing all categories, create bar plots for delta_metrics per category

# For consistent ordering, define categories explicitly
categories_order = [
    ('structured', 'HPC'),
    ('random', 'HPC'),
    ('structured', 'non-HPC'),
    ('random', 'non-HPC')
]

# Collect data
data = []
category_labels = []
for category in categories_order:
    if category in delta_metrics:
        data.append(delta_metrics[category])
        category_labels.append(f"{category[1]}-{category[0]}")
    else:
        data.append(0)  # or np.nan
        category_labels.append(f"{category[1]}-{category[0]}")

# Create bar plot
fig, ax = plt.subplots(figsize=(10, 6))

# Positions for the bars
positions = np.arange(len(categories_order))

# Plot bars
ax.bar(positions, data, align='center', alpha=0.7)
ax.axhline(0, color='black', linewidth=0.8)

# Customize the plot
ax.set_ylabel('Change in Phase-Time Correlation Value')
ax.set_xticks(positions)
ax.set_xticklabels(category_labels)
ax.set_title('Change in Phase-Time Correlation for Last N Trials')
ax.yaxis.grid(True)

# Save the figure
plt.tight_layout()
output_barplot_filename = f"metric_{metric_choice}_{filterSettingStr}_{N_TRIALS_PER_PHASE}trialsPerPhase.pdf"
output_barplot_filepath = os.path.join(output_dir, output_barplot_filename)
plt.savefig(output_barplot_filepath)
plt.close()
print(f"Saved metric bar plot to '{output_barplot_filepath}'")

# Now, generate surrogate distributions and compute p-values

#num_surrogates = 100  # Number of surrogate samples
#num_surrogates = 50000  # Number of surrogate samples
num_surrogates = 5000  # Number of surrogate samples

for key in cumulative_heatmaps_last.keys():
    num_units = unit_counts[key]
    observed_metric_value = observed_metrics_last[key]

    surrogate_metric_values = []
    for n in range(num_surrogates):
        # Randomly select units from the pool
        if num_units <= len(unit_pool_last):
            selected_units_first = random.sample(unit_pool_last, num_units)
            selected_units_last = random.sample(unit_pool_last, num_units)
        else:
            selected_units = random.choices(unit_pool_last, k=num_units)

        # Concatenate their relative times and phases for the first trials
        surrogate_cumulative_relative_times_first = np.concatenate([unit['relative_times'] for unit in selected_units_first])
        surrogate_cumulative_phases_first = np.concatenate([unit['phases'] for unit in selected_units_first])

        # Compute regression on the surrogate cumulative data for the first trials
        mask_spikes_first = (surrogate_cumulative_relative_times_first >= TIME_DISP_MIN) & (surrogate_cumulative_relative_times_first <= TIME_DISP_MAX)
        x_first = surrogate_cumulative_relative_times_first[mask_spikes_first]
        theta_first = surrogate_cumulative_phases_first[mask_spikes_first]
        circmean_first_bins, window_ranges_first = compute_circmean_per_window(
            x_data=x_first,
            theta_data=theta_first,
            window_size=window_size,
            step_size=step_size
        )
        window_centers_first = np.array([(start + end) / 2 for start, end in window_ranges_first])

        # Concatenate their relative times and phases for the last trials
        surrogate_cumulative_relative_times_last = np.concatenate([unit['relative_times'] for unit in selected_units_last])
        surrogate_cumulative_phases_last = np.concatenate([unit['phases'] for unit in selected_units_last])

        # Compute regression on the surrogate cumulative data for the last trials
        mask_spikes_last = (surrogate_cumulative_relative_times_last >= TIME_DISP_MIN) & (surrogate_cumulative_relative_times_last <= TIME_DISP_MAX)
        x_last = surrogate_cumulative_relative_times_last[mask_spikes_last]
        theta_last = surrogate_cumulative_phases_last[mask_spikes_last]
        circmean_last_bins, window_ranges_last = compute_circmean_per_window(
            x_data=x_last,
            theta_data=theta_last,
            window_size=window_size,
            step_size=step_size
        )
        window_centers_last = np.array([(start + end) / 2 for start, end in window_ranges_last])

        if metric_choice == 'unwrapped_linear_regression':
            # Unwrap the circular means
            #unwrapped_circmean_first_bins = np.unwrap(circmean_first_bins)
            #unwrapped_circmean_last_bins = np.unwrap(circmean_last_bins)
            unwrapped_circmean_first_bins = limited_unwrap(circmean_first_bins)
            unwrapped_circmean_last_bins = limited_unwrap(circmean_last_bins)

            # Perform constrained linear regression
            slope_first, intercept_first, r_value_first_surrogate = constrained_linear_regression(
                window_centers_first, unwrapped_circmean_first_bins)
            slope_last, intercept_last, r_value_last_surrogate = constrained_linear_regression(
                window_centers_last, unwrapped_circmean_last_bins)

            # Compute delta metric
            surrogate_metric_values.append(r_value_last_surrogate - r_value_first_surrogate)
        else:
            slope_bounds = [MIN_SLOPE, MAX_SLOPE]  # Constrain slope to be negative
            rho_last, p_last, s_fit_last, b_last = kempter.kempter_lincircTJ_slopeBounds(
                x=window_centers_last,
                theta=circmean_last_bins,
                slopeBounds=slope_bounds
            )
            if metric_choice == 'circular_linear_regression_rho':
                surrogate_metric_values.append(rho_last)
            elif metric_choice == 'circular_linear_regression_error':
                avg_error_last, _, _ = compute_average_error(
                    x_data=x_last,
                    theta_data=theta_last,
                    s_fit=s_fit_last,
                    b_fit=b_last,
                    window_size=window_size,
                    step_size=step_size
                )
                surrogate_metric_values.append(avg_error_last)
            elif metric_choice == 'circular_linear_regression_slope':
                surrogate_metric_values.append(s_fit_last)

    # Now we have the surrogate_metric_values
    surrogate_metric_values_array = np.array(surrogate_metric_values)

    # Compute p-value
    if observed_metric_value < 0:
        p_value = np.sum(surrogate_metric_values_array <= observed_metric_value) / num_surrogates
    else:
        p_value = np.sum(surrogate_metric_values_array >= observed_metric_value) / num_surrogates

    # Store p-value
    p_values[key] = p_value

    if key[1] =='HPC':
        if key[0]=='structured':
            currColor='red'
        else:
            currColor='lightcoral'
    else:
        if key[0]=='structured':
            currColor='brown'
        else:
            currColor='#D2B48C'
    # Plot the histogram
    #plt.figure(figsize=(6,6))
    plt.figure(figsize=(9,9))
    #plt.hist(surrogate_metric_values_array, bins=30, alpha=0.7, label='Surrogate Change in Phase-Time Correlation Values')
    #plt.hist(surrogate_metric_values_array, bins=np.linspace(-1,1,500), alpha=0.7, label='Surrogate Change in Phase-Time Correlation Values')
    #plt.hist(surrogate_metric_values_array, bins=np.linspace(-2,2,300), alpha=0.4, color=currColor, label='Surrogate Change in Phase-Time Correlation Values')
    plt.hist(surrogate_metric_values_array, bins=np.linspace(-2,2,50), alpha=0.4, color=currColor, label='Surrogate Change in Phase-Time Correlation Values')
    # Plot the observed metric value
    plt.axvline(observed_metric_value, color=currColor, linewidth=10, label='Observed Change in Phase-Time Correlation')
    # Annotate p-value
    plt.title(f'Null Distribution of Change in Phase-Time Correlation, {key[0]}-{key[1]} (n={num_surrogates} label shuffles)\nObserved Change in Phase-Time Correlation: {observed_metric_value:.3f}, p-value: {p_value:.5f}',fontsize=12)
    plt.xlabel('Change in Phase-Time Correlation Value')
    plt.ylabel('Frequency')

    # Save the plot
    output_filename = f"null_distribution_metric_{key[0]}_{key[1]}_{metric_choice}_{filterSettingStr}_{N_TRIALS_PER_PHASE}trialsPerPhase.pdf"
    output_filepath = os.path.join(output_dir, output_filename)
    plt.savefig(output_filepath)
    plt.close()
    print(f"Saved null distribution plot for key '{key}' to '{output_filepath}'")


