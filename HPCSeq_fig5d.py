#!/usr/bin/env python3
from utils import pdfTextSet
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial.distance import pdist
from matplotlib.colors import LinearSegmentedColormap
from utils import kempter  # Ensure this module is installed and accessible
from utils.saveCyclicColorbar2 import shift_colormap  # Ensure this module is installed and accessible
import logging  # Added for better logging

# -----------------------------
# Configuration
# -----------------------------

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Ensure the REGION_TYPE argument is provided
if len(sys.argv) < 2:
    logging.error("Usage: python3 script_name.py <REGION_TYPE>")
    sys.exit(1)

REGION_TYPE = sys.argv[1]  # Example: 'aic', 'hpc', etc.

# -----------------------------
# Custom Colormap Definition
# -----------------------------

def create_circular_colormap_two_fades_shifted():
    """
    Creates a shifted circular colormap using the 'twilight_shifted' palette.
    """
    newCmap = plt.get_cmap('twilight_shifted', 256)
    return shift_colormap(newCmap, start=0.4)

# -----------------------------
# Helper Functions
# -----------------------------

def compute_angles_general(x, y, xc, yc, a, b, theta):
    """
    Computes the angle phi for each point relative to the fitted ellipse with orientation theta.
    """
    # Translate points to the ellipse center
    x_shift = x - xc
    y_shift = y - yc

    # Rotate coordinates to align with ellipse axes
    cos_theta = np.cos(-theta)
    sin_theta = np.sin(-theta)
    x_rot = x_shift * cos_theta - y_shift * sin_theta
    y_rot = x_shift * sin_theta + y_shift * cos_theta

    # Compute angles using the parametric equations of the ellipse
    angles = np.arctan2(y_rot / b, x_rot / a) % (2 * np.pi)
    return angles

def compute_data_span(x, y):
    """
    Computes the largest distance between any two data points.
    """
    coords = np.column_stack((x, y))
    distances = pdist(coords)
    data_span = np.max(distances)
    return data_span

def ellipse_objective(params, x, y):
    """
    Objective function for ellipse fitting.
    """
    xc, yc, a, b, theta = params
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    x_shift = x - xc
    y_shift = y - yc
    x_rot = x_shift * cos_theta + y_shift * sin_theta
    y_rot = -x_shift * sin_theta + y_shift * cos_theta
    distances = (x_rot / a)**2 + (y_rot / b)**2 - 1
    return np.sum(distances**2)

def fit_ellipse(x, y):
    """
    Fits an ellipse to the given x and y data points using constrained optimization.

    Returns:
    - xc, yc: Center coordinates of the ellipse
    - a_len, b_len: Semi-major and semi-minor axes
    - theta: Orientation angle of the ellipse (radians)
    """
    # Initial guess for parameters
    xc0, yc0 = np.mean(x), np.mean(y)
    a0 = (np.max(x) - np.min(x)) / 2
    b0 = (np.max(y) - np.min(y)) / 2
    theta0 = 0.0  # Initial orientation
    initial_params = [xc0, yc0, a0, b0, theta0]

    # Compute data span
    data_span = compute_data_span(x, y)

    # Constraints: 2a ≤ data_span, 2b ≤ data_span
    def constraint_a(params):
        return data_span - 2 * params[2]  # data_span - 2a ≥ 0

    def constraint_b(params):
        return data_span - 2 * params[3]  # data_span - 2b ≥ 0

    # Semi-axes lengths must be positive
    bounds = [
        (None, None),  # xc
        (None, None),  # yc
        (1e-3, data_span / 2),  # a
        (1e-3, data_span / 2),  # b
        (None, None)   # theta
    ]

    constraints = [
        {'type': 'ineq', 'fun': constraint_a},
        {'type': 'ineq', 'fun': constraint_b},
    ]

    result = minimize(
        ellipse_objective, initial_params, args=(x, y),
        method='SLSQP', bounds=bounds, constraints=constraints
    )

    if result.success:
        xc, yc, a_opt, b_opt, theta_opt = result.x
        return xc, yc, a_opt, b_opt, theta_opt % (2 * np.pi)
    else:
        logging.error(f"Ellipse fitting failed: {result.message}")
        return None, None, None, None, None

def average_circular_error_per_scene(stimulus_order, data_angles, fitted_angles):
    """
    Computes the average circular error per scene.
    """
    unique_scenes = np.unique(stimulus_order)
    scene_errors = []

    for scene in unique_scenes:
        indices = np.where(stimulus_order == scene)[0]
        if len(indices) > 0:
            diffs = np.angle(np.exp(1j * (data_angles[indices] - fitted_angles[indices])))
            scene_error = np.mean(np.abs(diffs))
            scene_errors.append(scene_error)

    average_error = np.mean(scene_errors) if scene_errors else np.nan
    return average_error

def project_point_to_ellipse(x, y, xc, yc, a, b, theta):
    """
    Projects a single point (x, y) onto the ellipse defined by center (xc, yc),
    semi-major axis a, semi-minor axis b, and rotation theta.
    """
    def objective(phi):
        x_e = xc + a * np.cos(phi) * np.cos(theta) - b * np.sin(phi) * np.sin(theta)
        y_e = yc + a * np.cos(phi) * np.sin(theta) + b * np.sin(phi) * np.cos(theta)
        return (x - x_e)**2 + (y - y_e)**2

    # Initial guess for phi
    phi_initial = np.arctan2((y - yc), (x - xc))
    res = minimize_scalar(objective, bounds=(0, 2 * np.pi), method='bounded')

    if res.success:
        phi_opt = res.x
        x_proj = xc + a * np.cos(phi_opt) * np.cos(theta) - b * np.sin(phi_opt) * np.sin(theta)
        y_proj = yc + a * np.cos(phi_opt) * np.sin(theta) + b * np.sin(phi_opt) * np.cos(theta)
        return x_proj, y_proj
    else:
        # Fallback method (optional)
        logging.warning(f"Optimization failed for point ({x}, {y}). Using scaling fallback.")
        return x, y  # Return the original point if projection fails

def project_points_to_ellipse(x, y, xc, yc, a, b, theta):
    """
    Projects multiple points onto the ellipse.
    """
    x_proj = np.zeros_like(x)
    y_proj = np.zeros_like(y)
    for i in range(len(x)):
        x_p, y_p = project_point_to_ellipse(x[i], y[i], xc, yc, a, b, theta)
        x_proj[i] = x_p
        y_proj[i] = y_p
    return x_proj, y_proj

# -----------------------------
# Main Analysis and Plotting
# -----------------------------

def main():
    """
    Main function to load the transformed DataFrame, perform analysis
    by fitting an ellipse to the data points with constraints, and generate the required plots.
    """
    # -----------------------------
    # Configuration
    # -----------------------------

    # Path to the transformed DataFrame file
    transformed_dataframe_path = f"transformed_features_{REGION_TYPE}.pkl"

    # Define slope intervals for Kempter regression
    slope_intervals = [(-1/7.5, -1/11.5), (1/11.5, 1/7.5)]  # Adjusted to exclude slopes near zero

    # Number of permutations
    num_permutations = 50000  # Adjust as needed
    num_permutations = 5000  # Adjust as needed

    # Number of shifts up and down for plotting
    num_shifts = 3  # Number of times to shift up and down

    # -----------------------------
    # 1. Setup Output Directories
    # -----------------------------

    # Define base output directory
    output_base_dir = "Analysis_Plots"

    # Define subdirectories
    plots_dir = os.path.join(output_base_dir, "Condition_Plots")
    aggregated_dir = os.path.join(output_base_dir, "Aggregated_Results")
    permutation_dir = os.path.join(output_base_dir, "Permutation_Plots")
    change_null_dir = os.path.join(output_base_dir, "Change_Null_Plots")

    # Create directories if they don't exist
    for directory in [plots_dir, aggregated_dir, permutation_dir, change_null_dir]:
        os.makedirs(directory, exist_ok=True)

    # -----------------------------
    # 2. Data Preparation
    # -----------------------------

    # Load Data from Transformed DataFrame
    try:
        # Load the transformed DataFrame
        df = pd.read_pickle(transformed_dataframe_path)
        logging.info(f"Transformed DataFrame loaded successfully from '{transformed_dataframe_path}'.")
    except Exception as e:
        logging.error(f"Error loading transformed DataFrame: {e}")
        sys.exit(1)

    # Ensure that the required column exists
    if 'transformed_features' not in df.columns:
        logging.error("Error: 'transformed_features' column not found in DataFrame.")
        sys.exit(1)

    # Extract unique conditions based on 'region_type' and 'condition'
    condition_columns = ['region_type', 'condition']
    unique_conditions = df[condition_columns].drop_duplicates()
    num_unique_conditions = len(unique_conditions)
    logging.info(f"Number of unique (region_type, condition) pairs: {num_unique_conditions}")

    # Initialize list to hold data for all conditions
    all_conditions_data = []
    condition_labels = []  # To store labels for plotting

    # Iterate over each unique condition and extract early and late data
    for idx, condition in unique_conditions.iterrows():
        # Extract early phase data
        subset_early = df[(df['region_type'] == condition['region_type']) &
                          (df['condition'] == condition['condition']) &
                          (df['phase'] == 'early')]

        # Extract late phase data
        subset_late = df[(df['region_type'] == condition['region_type']) &
                         (df['condition'] == condition['condition']) &
                         (df['phase'] == 'late')]

        if subset_early.empty or subset_late.empty:
            logging.warning(f"Missing early or late phase data for condition: {condition.to_dict()}. Skipping.")
            continue

        # Assuming each row corresponds to a condition and 'transformed_features' contains data for that condition
        transformed_features_early = subset_early.iloc[0]['transformed_features']
        transformed_features_late = subset_late.iloc[0]['transformed_features']

        # Validate the transformed_features
        if not isinstance(transformed_features_early, np.ndarray) or not isinstance(transformed_features_late, np.ndarray):
            logging.warning(f"'transformed_features' is not a numpy array for condition: {condition.to_dict()}. Skipping.")
            continue

        if transformed_features_early.shape[1] != 2 or transformed_features_late.shape[1] != 2:
            logging.warning(f"Expected transformed features to have 2 components, but got {transformed_features_early.shape[1]} and {transformed_features_late.shape[1]} for condition: {condition.to_dict()}. Skipping.")
            continue

        all_conditions_data.append({
            'condition_label': f"{condition['region_type']}_{condition['condition']}",
            'early': transformed_features_early,
            'late': transformed_features_late
        })

        condition_labels.append(f"{condition['region_type']}_{condition['condition']}")

    num_conditions = len(all_conditions_data)
    logging.info(f"Processed {num_conditions} (region_type, condition) pairs from Transformed DataFrame.")

    if num_conditions == 0:
        logging.error("No valid conditions to analyze.")
        sys.exit(1)

    # -----------------------------
    # 3. Analysis and Plotting
    # -----------------------------

    # Initialize aggregated results
    aggregated_results = []

    for cond_idx, condition_data in enumerate(all_conditions_data):
        condition_label = condition_data['condition_label']
        logging.info(f"\nAnalyzing Condition {cond_idx + 1}/{num_conditions}: {condition_label}")

        # Initialize dictionary to store results for this condition
        condition_results = {
            'condition_label': condition_label,
            'early_error': np.nan,
            'late_error': np.nan,
            'change_observed': np.nan,
            'change_p_value': np.nan
        }

        # -----------------------------
        # Analyze Early Phase
        # -----------------------------
        transformed_data_early = condition_data['early']
        x_pca_early, y_pca_early = transformed_data_early[:, 0], transformed_data_early[:, 1]

        # Fit ellipse
        xc_e, yc_e, a_e, b_e, theta_e = fit_ellipse(x_pca_early, y_pca_early)
        if xc_e is None:
            logging.error("Ellipse fitting failed for early phase.")
            continue  # Skip to next condition

        # Compute fitted angles
        angles_fitted_e = compute_angles_general(x_pca_early, y_pca_early, xc_e, yc_e, a_e, b_e, theta_e)

        # Define stimulus order
        num_samples_e = transformed_data_early.shape[0]
        num_unique_stimuli = 10  # Adjust if different
        num_trials_e = num_samples_e // num_unique_stimuli
        stimulus_order_e = np.tile(np.arange(1, num_unique_stimuli + 1), num_trials_e)

        # Handle any remaining samples
        remaining_e = num_samples_e % num_unique_stimuli
        if remaining_e > 0:
            stimulus_order_e = np.concatenate([stimulus_order_e, np.arange(1, remaining_e + 1)])

        # Ensure lengths match
        if len(stimulus_order_e) != len(angles_fitted_e):
            logging.error("Stimulus order and angles have different lengths for early phase.")
            continue  # Skip to next condition

        # -----------------------------
        # Analyze Late Phase
        # -----------------------------
        transformed_data_late = condition_data['late']
        x_pca_late, y_pca_late = transformed_data_late[:, 0], transformed_data_late[:, 1]

        # Fit ellipse
        xc_l, yc_l, a_l, b_l, theta_l = fit_ellipse(x_pca_late, y_pca_late)
        if xc_l is None:
            logging.error("Ellipse fitting failed for late phase.")
            continue  # Skip to next condition

        # Compute fitted angles
        angles_fitted_l = compute_angles_general(x_pca_late, y_pca_late, xc_l, yc_l, a_l, b_l, theta_l)

        # Define stimulus order
        num_samples_l = transformed_data_late.shape[0]
        num_trials_l = num_samples_l // num_unique_stimuli
        stimulus_order_l = np.tile(np.arange(1, num_unique_stimuli + 1), num_trials_l)

        # Handle any remaining samples
        remaining_l = num_samples_l % num_unique_stimuli
        if remaining_l > 0:
            stimulus_order_l = np.concatenate([stimulus_order_l, np.arange(1, remaining_l + 1)])

        # Ensure lengths match
        if len(stimulus_order_l) != len(angles_fitted_l):
            logging.error("Stimulus order and angles have different lengths for late phase.")
            continue  # Skip to next condition

        # -----------------------------
        # Perform Linear-Circular Regression with Multiple Slope Intervals
        # -----------------------------

        # Initialize variables to store best fits for Early Phase
        best_avg_error_e = np.inf
        best_fit_params_e = None

        for slope_bounds in slope_intervals:
            rho_e, p_e, s_fit_e, b_fit_e = kempter.kempter_lincircTJ_slopeBounds(
                x=stimulus_order_e,
                theta=angles_fitted_e,
                slopeBounds=slope_bounds
            )

            # Compute fitted angles
            fitted_angles_e = 2 * np.pi * s_fit_e * stimulus_order_e + b_fit_e
            avg_error_e = average_circular_error_per_scene(
                stimulus_order_e, angles_fitted_e, fitted_angles_e
            )

            if avg_error_e < best_avg_error_e:
                best_avg_error_e = avg_error_e
                best_fit_params_e = {
                    'rho': rho_e,
                    'p': p_e,
                    's_fit': s_fit_e,
                    'b_fit': b_fit_e,
                    'slope_bounds': slope_bounds,
                    'fitted_angles': fitted_angles_e,
                    'avg_error': avg_error_e
                }

        if best_fit_params_e is None:
            logging.error("No valid regression fit found for early phase within the specified slope intervals.")
            continue  # Skip to next condition

        # Store early phase results
        condition_results['early_error'] = best_fit_params_e['avg_error']
        condition_results['early_p'] = best_fit_params_e['p']

        # Initialize variables to store best fits for Late Phase
        best_avg_error_l = np.inf
        best_fit_params_l = None

        for slope_bounds in slope_intervals:
            rho_l, p_l, s_fit_l, b_fit_l = kempter.kempter_lincircTJ_slopeBounds(
                x=stimulus_order_l,
                theta=angles_fitted_l,
                slopeBounds=slope_bounds
            )

            # Compute fitted angles
            fitted_angles_l = 2 * np.pi * s_fit_l * stimulus_order_l + b_fit_l
            avg_error_l = average_circular_error_per_scene(
                stimulus_order_l, angles_fitted_l, fitted_angles_l
            )

            if avg_error_l < best_avg_error_l:
                best_avg_error_l = avg_error_l
                best_fit_params_l = {
                    'rho': rho_l,
                    'p': p_l,
                    's_fit': s_fit_l,
                    'b_fit': b_fit_l,
                    'slope_bounds': slope_bounds,
                    'fitted_angles': fitted_angles_l,
                    'avg_error': avg_error_l
                }

        if best_fit_params_l is None:
            logging.error("No valid regression fit found for late phase within the specified slope intervals.")
            continue  # Skip to next condition

        # Store late phase results
        condition_results['late_error'] = best_fit_params_l['avg_error']
        condition_results['late_p'] = best_fit_params_l['p']

        # -----------------------------
        # Permutation Testing with Joint Shuffling
        # -----------------------------

        # Initialize list to store change null differences
        change_null = []

        for perm_idx in range(num_permutations):
            # Shuffle stimulus order once
            shuffled_order = np.random.permutation(stimulus_order_e)

            # Apply shuffled order to Early Phase
            # Perform regression over all slope intervals and select best fit
            best_avg_error_perm_e = np.inf
            best_fit_params_perm_e = None

            for slope_bounds in slope_intervals:
                rho_perm_e, p_perm_e, s_fit_perm_e, b_fit_perm_e = kempter.kempter_lincircTJ_slopeBounds(
                    x=shuffled_order,
                    theta=angles_fitted_e,
                    slopeBounds=slope_bounds
                )

                # Compute fitted angles
                fitted_angles_perm_e = 2 * np.pi * s_fit_perm_e * shuffled_order + b_fit_perm_e
                avg_error_perm_e = average_circular_error_per_scene(
                    shuffled_order, angles_fitted_e, fitted_angles_perm_e
                )

                if avg_error_perm_e < best_avg_error_perm_e:
                    best_avg_error_perm_e = avg_error_perm_e
                    best_fit_params_perm_e = {
                        'rho': rho_perm_e,
                        'p': p_perm_e,
                        's_fit': s_fit_perm_e,
                        'b_fit': b_fit_perm_e,
                        'slope_bounds': slope_bounds,
                        'fitted_angles': fitted_angles_perm_e,
                        'avg_error': avg_error_perm_e
                    }

            if best_fit_params_perm_e is None:
                logging.warning(f"Permutation {perm_idx + 1}: No valid regression fit found for early phase. Skipping this permutation.")
                continue

            # Apply shuffled order to Late Phase
            # Perform regression over all slope intervals and select best fit
            best_avg_error_perm_l = np.inf
            best_fit_params_perm_l = None

            for slope_bounds in slope_intervals:
                rho_perm_l, p_perm_l, s_fit_perm_l, b_fit_perm_l = kempter.kempter_lincircTJ_slopeBounds(
                    x=shuffled_order,
                    theta=angles_fitted_l,
                    slopeBounds=slope_bounds
                )

                # Compute fitted angles
                fitted_angles_perm_l = 2 * np.pi * s_fit_perm_l * shuffled_order + b_fit_perm_l
                avg_error_perm_l = average_circular_error_per_scene(
                    shuffled_order, angles_fitted_l, fitted_angles_perm_l
                )

                if avg_error_perm_l < best_avg_error_perm_l:
                    best_avg_error_perm_l = avg_error_perm_l
                    best_fit_params_perm_l = {
                        'rho': rho_perm_l,
                        'p': p_perm_l,
                        's_fit': s_fit_perm_l,
                        'b_fit': b_fit_perm_l,
                        'slope_bounds': slope_bounds,
                        'fitted_angles': fitted_angles_perm_l,
                        'avg_error': avg_error_perm_l
                    }

            if best_fit_params_perm_l is None:
                logging.warning(f"Permutation {perm_idx + 1}: No valid regression fit found for late phase. Skipping this permutation.")
                continue

            # Compute difference and store
            change_diff = best_fit_params_perm_l['avg_error'] - best_fit_params_perm_e['avg_error']
            change_null.append(change_diff)

            if (perm_idx + 1) % 1000 == 0:
                logging.info(f"  Completed {perm_idx + 1} / {num_permutations} permutations.")

        # Store permutation null distribution and observed change
        condition_results['change_null'] = change_null

        # -----------------------------
        # Compute Observed Change
        # -----------------------------
        observed_change = condition_results['late_error'] - condition_results['early_error']
        condition_results['change_observed'] = observed_change

        # -----------------------------
        # Compute P-value for Change
        # -----------------------------
        change_null_array = np.array(change_null)
        num_extreme = np.sum(change_null_array < observed_change)
        p_value_change = (num_extreme + 1) / (num_permutations + 1)  # Adding 1 for continuity correction
        condition_results['change_p_value'] = p_value_change

        # -----------------------------
        # Plotting
        # -----------------------------

        # Define subplot axes for within-phase analysis
        fig_condition, axes_condition = plt.subplots(2, 2, figsize=(20, 15))
        # axes_condition layout:
        # [0,0] - Early Phase Scatter
        # [0,1] - Early Phase Fit Error Plot
        # [1,0] - Late Phase Scatter
        # [1,1] - Late Phase Fit Error Plot

        # -----------------------------
        # Plot Early Phase Scatter
        # -----------------------------
        ax_early_scatter = axes_condition[0, 0]
        scatter_early = ax_early_scatter.scatter(
            x_pca_early,
            y_pca_early,
            c=stimulus_order_e,
            cmap=create_circular_colormap_two_fades_shifted(),
            norm=plt.Normalize(1, num_unique_stimuli),
            s=100,
            edgecolors='black',
            alpha=0.7
        )

        # Plot the fitted ellipse
        ellipse_theta_vals = np.linspace(0, 2 * np.pi, 100)
        ellipse_x_e = xc_e + a_e * np.cos(ellipse_theta_vals) * np.cos(theta_e) - b_e * np.sin(ellipse_theta_vals) * np.sin(theta_e)
        ellipse_y_e = yc_e + a_e * np.cos(ellipse_theta_vals) * np.sin(theta_e) + b_e * np.sin(ellipse_theta_vals) * np.cos(theta_e)
        ax_early_scatter.plot(ellipse_x_e, ellipse_y_e, 'k--', linewidth=2)

        # Draw Major and Minor Axes
        # Major Axis
        major_x1_e = xc_e + a_e * np.cos(theta_e)
        major_y1_e = yc_e + a_e * np.sin(theta_e)
        major_x2_e = xc_e - a_e * np.cos(theta_e)
        major_y2_e = yc_e - a_e * np.sin(theta_e)
        ax_early_scatter.plot([major_x1_e, major_x2_e], [major_y1_e, major_y2_e], color='black', linestyle='-', linewidth=1.5)

        # Minor Axis
        minor_x1_e = xc_e + b_e * np.cos(theta_e + np.pi/2)
        minor_y1_e = yc_e + b_e * np.sin(theta_e + np.pi/2)
        minor_x2_e = xc_e - b_e * np.cos(theta_e + np.pi/2)
        minor_y2_e = yc_e - b_e * np.sin(theta_e + np.pi/2)
        ax_early_scatter.plot([minor_x1_e, minor_x2_e], [minor_y1_e, minor_y2_e], color='black', linestyle='-', linewidth=1.5)

        # Set title and labels
        ax_early_scatter.set_title(f'{condition_label} - Early Phase', fontsize=14)
        ax_early_scatter.set_xlabel('PC1', fontsize=12)
        ax_early_scatter.set_ylabel('PC2', fontsize=12)
        ax_early_scatter.set_aspect('equal')
        if REGION_TYPE.lower() == 'hpc': 
            ax_early_scatter.set_xlim([-65,65])
            ax_early_scatter.set_ylim([-65,65])
        else:
            ax_early_scatter.set_xlim([-90,100])
            ax_early_scatter.set_ylim([-90,100])

        # Project points onto ellipse and plot connections
        x_proj_e, y_proj_e = project_points_to_ellipse(x_pca_early, y_pca_early, xc_e, yc_e, a_e, b_e, theta_e)
        for i in range(len(x_pca_early)):
            ax_early_scatter.plot([x_pca_early[i], x_proj_e[i]], [y_pca_early[i], y_proj_e[i]], 'k-', linewidth=0.5, alpha=0.2)

        # Plot projected points with distinct markers
        scatter_projected_e = ax_early_scatter.scatter(
            x_proj_e,
            y_proj_e,
            c=stimulus_order_e,
            cmap=create_circular_colormap_two_fades_shifted(),
            norm=plt.Normalize(1, num_unique_stimuli),
            s=300,
            edgecolors='black',
            alpha=0.7
        )

        # -----------------------------
        # Plot Early Phase Fit Error
        # -----------------------------
        ax_early_error = axes_condition[0, 1]
        ax_early_error.set_title(f'Early Phase - Avg Circular Error: {condition_results["early_error"]:.2f} rad\nPermutation p-value: {condition_results["early_p"]:.4f}', fontsize=12)
        ax_early_error.set_xlabel('Scene Order', fontsize=12)
        ax_early_error.set_ylabel('Fitted Angle (radians)', fontsize=12)
        ax_early_error.set_xlim(0, num_unique_stimuli + 1)
        ax_early_error.grid(True, linestyle='--', alpha=0.5)
        ax_early_error.set_aspect('equal')

        # Plot horizontal lines at multiples of 2π
        for k in range(-num_shifts, num_shifts + 1):
            y_value = k * 2 * np.pi
            ax_early_error.axhline(y_value, color='k', linestyle='--', linewidth=1)

        # Scatter plot for angles shifted by multiples of 2π
        for k in range(-num_shifts, num_shifts + 1):
            angles_shifted_e = best_fit_params_e['fitted_angles'] + k * 2 * np.pi
            ax_early_error.scatter(
                stimulus_order_e,
                angles_shifted_e,
                c=stimulus_order_e,
                cmap=create_circular_colormap_two_fades_shifted(),
                norm=plt.Normalize(1, num_unique_stimuli),
                s=100,
                edgecolors='black',
                alpha=0.7,
            )

        # Plot the regression fit line and its wrapped versions
        x_fit_e = np.linspace(0, num_unique_stimuli + 1, 500)
        theta_fit_e = 2 * np.pi * best_fit_params_e['s_fit'] * x_fit_e + best_fit_params_e['b_fit']
        for k in range(-num_shifts, num_shifts + 1):
            ax_early_error.plot(x_fit_e, theta_fit_e + k * 2 * np.pi, color='black', linestyle='-', linewidth=2)

        # Set y-ticks at multiples of π
        y_min_e = np.min(best_fit_params_e['fitted_angles']) - 2 * np.pi
        y_max_e = np.max(best_fit_params_e['fitted_angles']) + 2 * np.pi
        y_ticks_e = np.arange(np.floor(y_min_e / np.pi), np.ceil(y_max_e / np.pi) + 1) * np.pi
        ax_early_error.set_yticks(y_ticks_e)
        ax_early_error.set_yticklabels([f'{(ytick / np.pi):.0f}π' for ytick in y_ticks_e])

        # -----------------------------
        # Plot Late Phase Scatter
        # -----------------------------
        ax_late_scatter = axes_condition[1, 0]
        scatter_late = ax_late_scatter.scatter(
            x_pca_late,
            y_pca_late,
            c=stimulus_order_l,
            cmap=create_circular_colormap_two_fades_shifted(),
            norm=plt.Normalize(1, num_unique_stimuli),
            s=100,
            edgecolors='black',
            alpha=0.7
        )

        # Plot the fitted ellipse
        ellipse_theta_vals_l = np.linspace(0, 2 * np.pi, 100)
        ellipse_x_l = xc_l + a_l * np.cos(ellipse_theta_vals_l) * np.cos(theta_l) - b_l * np.sin(ellipse_theta_vals_l) * np.sin(theta_l)
        ellipse_y_l = yc_l + a_l * np.cos(ellipse_theta_vals_l) * np.sin(theta_l) + b_l * np.sin(ellipse_theta_vals_l) * np.cos(theta_l)
        ax_late_scatter.plot(ellipse_x_l, ellipse_y_l, 'k--', linewidth=2)

        # Draw Major and Minor Axes
        # Major Axis
        major_x1_l = xc_l + a_l * np.cos(theta_l)
        major_y1_l = yc_l + a_l * np.sin(theta_l)
        major_x2_l = xc_l - a_l * np.cos(theta_l)
        major_y2_l = yc_l - a_l * np.sin(theta_l)
        ax_late_scatter.plot([major_x1_l, major_x2_l], [major_y1_l, major_y2_l], color='black', linestyle='-', linewidth=1.5)

        # Minor Axis
        minor_x1_l = xc_l + b_l * np.cos(theta_l + np.pi/2)
        minor_y1_l = yc_l + b_l * np.sin(theta_l + np.pi/2)
        minor_x2_l = xc_l - b_l * np.cos(theta_l + np.pi/2)
        minor_y2_l = yc_l - b_l * np.sin(theta_l + np.pi/2)
        ax_late_scatter.plot([minor_x1_l, minor_x2_l], [minor_y1_l, minor_y2_l], color='black', linestyle='-', linewidth=1.5)

        # Set title and labels
        ax_late_scatter.set_title(f'{condition_label} - Late Phase', fontsize=14)
        ax_late_scatter.set_xlabel('PC1', fontsize=12)
        ax_late_scatter.set_ylabel('PC2', fontsize=12)
        ax_late_scatter.set_aspect('equal')
        if REGION_TYPE.lower() == 'hpc': 
            ax_late_scatter.set_xlim([-65,65])
            ax_late_scatter.set_ylim([-65,65])
        else:
            ax_late_scatter.set_xlim([-90,100])
            ax_late_scatter.set_ylim([-90,100])

        # Project points onto ellipse and plot connections
        x_proj_l, y_proj_l = project_points_to_ellipse(x_pca_late, y_pca_late, xc_l, yc_l, a_l, b_l, theta_l)
        for i in range(len(x_pca_late)):
            ax_late_scatter.plot([x_pca_late[i], x_proj_l[i]], [y_pca_late[i], y_proj_l[i]], 'k-', linewidth=0.5, alpha=0.2)

        # Plot projected points with distinct markers
        scatter_projected_l = ax_late_scatter.scatter(
            x_proj_l,
            y_proj_l,
            c=stimulus_order_l,
            cmap=create_circular_colormap_two_fades_shifted(),
            norm=plt.Normalize(1, num_unique_stimuli),
            s=300,
            edgecolors='black',
            alpha=0.7
        )

        # -----------------------------
        # Plot Late Phase Fit Error
        # -----------------------------
        ax_late_error = axes_condition[1, 1]
        ax_late_error.set_title(f'Late Phase - Avg Circular Error: {condition_results["late_error"]:.2f} rad\nPermutation p-value: {condition_results["late_p"]:.4f}', fontsize=12)
        ax_late_error.set_xlabel('Scene Order', fontsize=12)
        ax_late_error.set_ylabel('Fitted Angle (radians)', fontsize=12)
        ax_late_error.set_xlim(0, num_unique_stimuli + 1)
        ax_late_error.grid(True, linestyle='--', alpha=0.5)
        ax_late_error.set_aspect('equal')

        # Plot horizontal lines at multiples of 2π
        for k in range(-num_shifts, num_shifts + 1):
            y_value = k * 2 * np.pi
            ax_late_error.axhline(y_value, color='k', linestyle='--', linewidth=1)

        # Scatter plot for angles shifted by multiples of 2π
        for k in range(-num_shifts, num_shifts + 1):
            angles_shifted_l = best_fit_params_l['fitted_angles'] + k * 2 * np.pi
            ax_late_error.scatter(
                stimulus_order_l,
                angles_shifted_l,
                c=stimulus_order_l,
                cmap=create_circular_colormap_two_fades_shifted(),
                norm=plt.Normalize(1, num_unique_stimuli),
                s=100,
                edgecolors='black',
                alpha=0.7,
            )

        # Plot the regression fit line and its wrapped versions
        x_fit_l = np.linspace(0, num_unique_stimuli + 1, 500)
        theta_fit_l = 2 * np.pi * best_fit_params_l['s_fit'] * x_fit_l + best_fit_params_l['b_fit']
        for k in range(-num_shifts, num_shifts + 1):
            ax_late_error.plot(x_fit_l, theta_fit_l + k * 2 * np.pi, color='black', linestyle='-', linewidth=2)

        # Set y-ticks at multiples of π
        y_min_l = np.min(best_fit_params_l['fitted_angles']) - 2 * np.pi
        y_max_l = np.max(best_fit_params_l['fitted_angles']) + 2 * np.pi
        y_ticks_l = np.arange(np.floor(y_min_l / np.pi), np.ceil(y_max_l / np.pi) + 1) * np.pi
        ax_late_error.set_yticks(y_ticks_l)
        ax_late_error.set_yticklabels([f'{(ytick / np.pi):.0f}π' for ytick in y_ticks_l])

        # -----------------------------
        # Plot Change Null Distribution
        # -----------------------------
        fig_change_null, ax_change_null = plt.subplots(figsize=(8, 6))
        ax_change_null.hist(change_null, bins=50, color='gray', alpha=0.7)
        ax_change_null.axvline(observed_change, color='red', linestyle='--', linewidth=2, label=f'Observed Change: {observed_change:.2f} rad')
        ax_change_null.set_title(f'Change Null Distribution\n{condition_label}\nChange p-value: {condition_results["change_p_value"]:.4f}', fontsize=14)
        ax_change_null.set_xlabel('Change in Avg Circular Error (Late - Early) (rad)', fontsize=12)
        ax_change_null.set_ylabel('Frequency', fontsize=12)
        ax_change_null.legend(fontsize=12)
        plt.tight_layout()
        change_null_plot_filename = os.path.join(change_null_dir, f'{REGION_TYPE}_Change_Null_Distribution_{condition_label}.pdf')
        plt.savefig(change_null_plot_filename, dpi=300)
        plt.close(fig_change_null)

        # -----------------------------
        # Save Within-Phase Plots
        # -----------------------------
        plt.tight_layout()
        condition_plot_filename = os.path.join(plots_dir, f'{REGION_TYPE}_Condition_{condition_label}.pdf')
        fig_condition.savefig(condition_plot_filename, dpi=300)
        plt.close(fig_condition)

        # -----------------------------
        # Aggregate Results
        # -----------------------------
        aggregated_results.append(condition_results)

    # -----------------------------
    # 4. Aggregated Results Plotting
    # -----------------------------

    # Convert aggregated results to DataFrame
    agg_df = pd.DataFrame(aggregated_results)

    # Plot Average Circular Errors for Early and Late Phases
    plt.figure(figsize=(12, 8))
    bar_width = 0.35
    indices = np.arange(len(agg_df))
    plt.bar(indices, agg_df['early_error'], bar_width, label='Early Phase', color='skyblue')
    plt.bar(indices + bar_width, agg_df['late_error'], bar_width, label='Late Phase', color='salmon')
    plt.xlabel('Condition', fontsize=14)
    plt.ylabel('Avg Circ-Lin Fit Error (rad)', fontsize=14)
    plt.title(f'Avg Circular Errors Across Conditions, Region {REGION_TYPE}', fontsize=16)
    plt.xticks(indices + bar_width / 2, agg_df['condition_label'], fontsize=12, rotation=45, ha='right')
    plt.legend(fontsize=12)
    plt.tight_layout()
    agg_error_filename = os.path.join(aggregated_dir, f'{REGION_TYPE}_Aggregated_Average_Circular_Errors.pdf')
    plt.savefig(agg_error_filename, dpi=300)
    plt.close()

    # Plot Observed Change with p-values
    plt.figure(figsize=(12, 8))
    plt.bar(indices, agg_df['change_observed'], color='mediumpurple')
    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel('Condition', fontsize=14)
    plt.ylabel('Observed Change in Avg Circ-Lin Fit Error (Late - Early) (rad)', fontsize=14)
    plt.title(f'Observed Change Across Conditions, Region {REGION_TYPE}', fontsize=16)
    plt.xticks(indices, agg_df['condition_label'], fontsize=12, rotation=45, ha='right')

    # Annotate bars with p-values
    for idx, (val, p_val) in enumerate(zip(agg_df['change_observed'], agg_df['change_p_value'])):
        if not np.isnan(val):
            plt.text(idx, val + 0.02, f"{val:.2f}\np={p_val:.4f}", ha='center', fontsize=10)

    plt.tight_layout()
    change_agg_filename = os.path.join(aggregated_dir, f'{REGION_TYPE}_Aggregated_Observed_Changes.pdf')
    plt.savefig(change_agg_filename, dpi=300)
    plt.close()

    # -----------------------------
    # 5. Combined Null Distributions Plotting
    # -----------------------------
    # (Optional: Already handled per condition above)

    logging.info("\nAll plots have been saved in the 'Analysis_Plots' directory.")

if __name__ == "__main__":
    main()

