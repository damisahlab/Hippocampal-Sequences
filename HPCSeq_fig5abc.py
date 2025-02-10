#!/usr/bin/env python3
from utils import pdfTextSet
import pdb
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar
import os
import sys
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from utils import kempter  # Ensure the kempter module is available
from scipy.stats import circmean

REGION_TYPE = sys.argv[1]#'aic'

# -----------------------------
# Custom Colormap Definition
# -----------------------------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from utils.saveCyclicColorbar2 import shift_colormap

def create_circular_colormap_two_fades_shifted():
    newCmap=plt.get_cmap('twilight_shifted', 256)
    #return  shift_colormap(newCmap, start=0.25)
    return  shift_colormap(newCmap, start=0.4)
    #return  shift_colormap(newCmap, start=0.5)
    #return plt.get_cmap('twilight', 256)


# -----------------------------
# Helper Functions for DataFrame Analysis
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
    from scipy.spatial.distance import pdist
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
    try:
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
            print(f"Ellipse fitting failed: {result.message}", file=sys.stderr)
            return None, None, None, None, None

    except Exception as e:
        print(f"Ellipse fitting failed: {e}", file=sys.stderr)
        return None, None, None, None, None  # Return None values

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

    # average_error = np.mean(scene_errors) if scene_errors else np.nan
    average_error = circmean(scene_errors) if scene_errors else np.nan
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
        print(f"Optimization failed for point ({x}, {y}). Using scaling fallback.", file=sys.stderr)
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

    # Define multiple slope intervals for kempter regression
    slope_intervals = [(-1/7.5, -1/11.5), (1/11.5, 1/7.5)]  # Adjusted to exclude slopes near zero since all scenes at the same phase is not a good encoding

    # Number of permutations
    num_permutations = 5000  # Adjust as needed
    #num_permutations = 50000  # Adjust as needed

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

    # Create directories if they don't exist
    for directory in [plots_dir, aggregated_dir, permutation_dir]:
        os.makedirs(directory, exist_ok=True)

    # -----------------------------
    # 2. Data Preparation
    # -----------------------------

    # Load Data from Transformed DataFrame
    try:
        # Load the transformed DataFrame
        df = pd.read_pickle(transformed_dataframe_path)
        print(f"Transformed DataFrame loaded successfully from '{transformed_dataframe_path}'.")
    except Exception as e:
        print(f"Error loading transformed DataFrame: {e}", file=sys.stderr)
        sys.exit(1)

    # Ensure that the required column exists
    if 'transformed_features' not in df.columns:
        print("Error: 'transformed_features' column not found in DataFrame.", file=sys.stderr)
        sys.exit(1)

    # Extract unique conditions based on 'region_type', 'condition', and 'phase'
    condition_columns = ['region_type', 'condition', 'phase']
    unique_conditions = df[condition_columns].drop_duplicates()
    num_conditions = len(unique_conditions)
    print(f"Number of unique conditions: {num_conditions}")

    # Initialize list to hold data for all conditions
    all_conditions_data = []
    condition_labels = []  # To store labels for plotting

    # Iterate over each unique condition and extract data
    for idx, condition in unique_conditions.iterrows():
        # Extract the corresponding transformed_features
        subset = df[(df['region_type'] == condition['region_type']) &
                    (df['condition'] == condition['condition']) &
                    (df['phase'] == condition['phase'])]

        if subset.empty:
            print(f"No data found for condition: {condition.to_dict()}. Skipping.", file=sys.stderr)
            continue

        # Assuming each row corresponds to a condition and 'transformed_features' contains data for that condition
        # 'transformed_features' is expected to be a NumPy array of shape (num_samples, 2)
        transformed_features = subset.iloc[0]['transformed_features']

        # Validate the transformed_features
        if not isinstance(transformed_features, np.ndarray):
            print(f"Warning: 'transformed_features' is not a numpy array for condition: {condition.to_dict()}. Skipping.", file=sys.stderr)
            continue

        if transformed_features.shape[1] != 2:
            print(f"Warning: Expected transformed features to have 2 components, but got {transformed_features.shape[1]} for condition: {condition.to_dict()}. Skipping.", file=sys.stderr)
            continue

        all_conditions_data.append(transformed_features)

        # Create a label for this condition
        label = f"{condition['region_type']}_{condition['condition']}_{condition['phase']}"
        condition_labels.append(label)

    num_conditions = len(all_conditions_data)
    print(f"Processed {num_conditions} conditions from Transformed DataFrame.")

    # -----------------------------
    # 3. Analysis and Plotting
    # -----------------------------

    num_conditions = len(all_conditions_data)
    if num_conditions == 0:
        print("No valid conditions to analyze.", file=sys.stderr)
        sys.exit(1)

    # Initialize a large figure for all conditions
    fig_all, axes_all = plt.subplots(num_conditions, 2, figsize=(10, 5 * num_conditions))
    #fig_all, axes_all = plt.subplots(num_conditions, 2, figsize=(8, 5 * num_conditions))

    # Create and initialize the custom colormap
    custom_cmap = create_circular_colormap_two_fades_shifted()
    cmap_unique = custom_cmap  # Use the custom colormap

    # Normalize stimulus_order_unique to [0,1] based on stimulus order
    stimulus_order_unique = np.arange(1, 11)  # 1 to 10
    norm = plt.Normalize(stimulus_order_unique.min(), stimulus_order_unique.max())

    # Define cycle length (assuming 2π for angular continuity)
    cycle_length = 2 * np.pi

    # Store average circular errors, p-values, and null distributions
    average_circular_errors = []
    p_values = []
    null_distributions = []  # For storing permutation errors per condition
    p_values_permutation = []  # For p-values from permutation test

    for cond in range(num_conditions):
        try:
            # Get the transformed data for this condition
            transformed_data = all_conditions_data[cond]  # Shape: (num_samples, 2)
            if transformed_data is None:
                raise ValueError("No transformed data available for this condition.")

            # Extract x and y coordinates
            x_pca, y_pca = transformed_data[:, 0], transformed_data[:, 1]

            # Check for NaN or infinite values in data
            if np.any(np.isnan(x_pca)) or np.any(np.isnan(y_pca)) or np.any(np.isinf(x_pca)) or np.any(np.isinf(y_pca)):
                raise ValueError("Data contains NaN or infinite values.")

            # Fit an ellipse to the data points using constrained optimization
            xc, yc, a, b, theta = fit_ellipse(x_pca, y_pca)
            if xc is None or np.isnan(xc) or np.isnan(yc) or np.isnan(a) or np.isnan(b) or np.isnan(theta):
                raise ValueError("Ellipse fitting failed or resulted in invalid parameters.")

            # Compute fitted angles relative to the fitted ellipse
            angles_fitted = compute_angles_general(x_pca, y_pca, xc, yc, a, b, theta)

            # Check for NaN or infinite values in angles
            if np.any(np.isnan(angles_fitted)) or np.any(np.isinf(angles_fitted)):
                raise ValueError("Computed angles contain NaN or infinite values.")

            # Define stimulus order
            # Assuming stimulus order corresponds to scene numbers (1 to 10) and repeated across trials
            num_samples = transformed_data.shape[0]
            num_unique_stimuli = 10  # Adjust if different
            num_trials = num_samples // num_unique_stimuli
            stimulus_order = np.tile(stimulus_order_unique, num_trials)

            # Handle any remaining samples if num_samples is not a multiple of num_unique_stimuli
            remaining = num_samples % num_unique_stimuli
            if remaining > 0:
                stimulus_order = np.concatenate([stimulus_order, stimulus_order_unique[:remaining]])

            # Ensure that stimulus_order and angles_fitted have the same length
            if len(stimulus_order) != len(angles_fitted):
                raise ValueError("Stimulus order and angles have different lengths.")

            # -----------------------------
            # Perform Linear-Circular Regression with Multiple Slope Intervals
            # -----------------------------

            best_avg_error = np.inf
            best_fit_params = None

            for slope_bounds in slope_intervals:
                # Perform linear-circular regression using kempter module
                rho, p, s_fit, b_fit = kempter.kempter_lincircTJ_slopeBounds(
                    x=stimulus_order,
                    theta=angles_fitted,
                    slopeBounds=slope_bounds
                )

                # Compute average circular error per scene
                fitted_angles = 2 * np.pi * s_fit * stimulus_order + b_fit
                avg_error = average_circular_error_per_scene(
                    stimulus_order, angles_fitted, fitted_angles
                )

                # Update the best fit if the current average error is lower
                if avg_error < best_avg_error:
                    best_avg_error = avg_error
                    best_fit_params = {
                        'rho': rho,
                        'p': p,
                        's_fit': s_fit,
                        'b_fit': b_fit,
                        'slope_bounds': slope_bounds,
                        'fitted_angles': fitted_angles
                    }

            # Use the best fit parameters
            if best_fit_params is None:
                raise ValueError("No valid regression fit found within the specified slope intervals.")

            rho = best_fit_params['rho']
            p = best_fit_params['p']
            s_fit = best_fit_params['s_fit']
            b_fit = best_fit_params['b_fit']
            slope_bounds_used = best_fit_params['slope_bounds']
            fitted_angles = best_fit_params['fitted_angles']
            avg_error = best_avg_error

            average_circular_errors.append(avg_error)
            p_values.append(p)

            # -----------------------------
            # Permutation Testing
            # -----------------------------

            # Initialize list to store permutation errors
            perm_errors = []

            for _ in range(num_permutations):
                # Shuffle the stimulus_order
                shuffled_order = np.random.permutation(stimulus_order)

                # Perform linear-circular regression on shuffled data with the same slope bounds
                rho_perm, p_perm, s_perm, b_perm = kempter.kempter_lincircTJ_slopeBounds(
                    x=shuffled_order,
                    theta=angles_fitted,
                    slopeBounds=slope_bounds_used
                )

                # Compute average circular error per scene for shuffled data
                fitted_angles_perm = 2 * np.pi * s_perm * shuffled_order + b_perm
                avg_error_perm = average_circular_error_per_scene(
                    shuffled_order, angles_fitted, fitted_angles_perm
                )

                perm_errors.append(avg_error_perm)

            # Store the permutation errors
            null_distributions.append(perm_errors)

            # -----------------------------
            # P-value Calculation from Permutations
            # -----------------------------

            # Calculate p-value based on permutation test
            num_extreme = np.sum(np.array(perm_errors) <= avg_error)
            p_value_perm = (num_extreme + 1) / (num_permutations + 1)
            p_values_permutation.append(p_value_perm)

            # -----------------------------
            # Plotting
            # -----------------------------

            # Define subplot axes
            if num_conditions > 1:
                ax1, ax2 = axes_all[cond, 0], axes_all[cond, 1]
            else:
                ax1, ax2 = axes_all[0], axes_all[1]

            # Assign colors based on stimulus order using the custom colormap
            colors = cmap_unique(norm(stimulus_order))

            # ---- Subplot 1: PCA Component Scatter Plot ----
            scatter1 = ax1.scatter(
                x_pca,
                y_pca,
                c=stimulus_order,
                cmap=cmap_unique,
                norm=norm,
                s=100,
                edgecolors='black',
                alpha=0.7
            )

            # Plot the fitted ellipse
            ellipse_theta_vals = np.linspace(0, 2 * np.pi, 100)
            ellipse_x = xc + a * np.cos(ellipse_theta_vals) * np.cos(theta) - b * np.sin(ellipse_theta_vals) * np.sin(theta)
            ellipse_y = yc + a * np.cos(ellipse_theta_vals) * np.sin(theta) + b * np.sin(ellipse_theta_vals) * np.cos(theta)
            ax1.plot(ellipse_x, ellipse_y, 'k--', linewidth=2)

            # Draw Major and Minor Axes
            # Major Axis
            major_x1 = xc + a * np.cos(theta)
            major_y1 = yc + a * np.sin(theta)
            major_x2 = xc - a * np.cos(theta)
            major_y2 = yc - a * np.sin(theta)
            ax1.plot([major_x1, major_x2], [major_y1, major_y2], color='black', linestyle='-', linewidth=1.5)

            # Minor Axis
            minor_x1 = xc + b * np.cos(theta + np.pi/2)
            minor_y1 = yc + b * np.sin(theta + np.pi/2)
            minor_x2 = xc - b * np.cos(theta + np.pi/2)
            minor_y2 = yc - b * np.sin(theta + np.pi/2)
            ax1.plot([minor_x1, minor_x2], [minor_y1, minor_y2], color='black', linestyle='-', linewidth=1.5)

            # Set title with average circular error and p-value
            ax1.set_title(f'{condition_labels[cond]}', fontsize=14)
            ax1.set_xlabel('PC1', fontsize=12)
            ax1.set_ylabel('PC2', fontsize=12)
            ax1.set_aspect('equal')
            if REGION_TYPE == 'hpc': 
                ax1.set_xlim([-65,65])
                ax1.set_ylim([-65,65])
            else:
                ax1.set_xlim([-90,100])
                ax1.set_ylim([-90,100])
                
            # Create a colorbar to represent stimulus progression
            '''
            cbar = fig_all.colorbar(scatter1, ax=ax1, orientation='vertical', fraction=0.046, pad=0.04)
            cbar.set_label('Scene Order', fontsize=12)
            cbar.set_ticks(stimulus_order_unique)
            cbar.set_ticklabels([str(i) for i in stimulus_order_unique])
            '''
            # Create a cyclic colormap and scatter plot (assuming scatter1 already defined)
            #cbar = fig_all.colorbar(scatter1, ax=ax1, orientation='vertical', fraction=0.046, pad=0.04)
            #cbar.set_label('Scene Order', fontsize=12)

            # Ensure the ticks go from 0 to 1 in a cyclic way and add an extra tick for the wrap-around
            #cbar.set_ticks(np.linspace(0, 1, len(stimulus_order_unique) + 1))  # +1 to close the cycle
            #cbar.set_ticklabels([str(i) for i in stimulus_order_unique] + [str(stimulus_order_unique[0])])  # Add the first label again to close the cycle

            # -----------------------------
            # Projection of Points onto Ellipse Circumference
            # -----------------------------

            # Project all points onto the ellipse
            x_proj, y_proj = project_points_to_ellipse(x_pca, y_pca, xc, yc, a, b, theta)

            # Plot lines connecting original points to their projections
            for i in range(len(x_pca)):
                ax1.plot([x_pca[i], x_proj[i]], [y_pca[i], y_proj[i]], 'k-', linewidth=0.5, alpha=0.2)

            # Plot projected points with distinct markers
            scatter_projected = ax1.scatter(
                x_proj,
                y_proj,
                c=stimulus_order,
                cmap=cmap_unique,
                norm=norm,
                s=1000,#2000,#3000,#300,
                edgecolors='black',
                alpha=0.7
            )

            # ---- Subplot 2: Avg Circ-Lin Fit Error Plot ----

            # Define subplot title with regression parameters and p-value
            ax2.set_title(f'Avg Circular Error: {avg_error:.2f} rad\nSlope: {s_fit:.3f} cycles/unit, Offset: {b_fit:.2f} rad\nPermutation p-value: {p_value_perm:.4f}\nSlope Bounds: ({round(1.0/slope_bounds_used[1])})-({round(1.0/slope_bounds_used[0])}) scenes/cycle', fontsize=12)
            ax2.set_xlabel('Scene Order', fontsize=12)
            ax2.set_ylabel('Fitted Angle (radians)', fontsize=12)
            ax2.set_xlim(0, stimulus_order_unique.max() + 1)
            ax2.grid(True, linestyle='--', alpha=0.5)

            ax2.set_aspect('equal')
            # Plot horizontal lines at multiples of 2π
            y_ticks = []
            for k in range(-num_shifts, num_shifts + 1):
                y_value = k * 2 * np.pi
                ax2.axhline(y_value, color='k', linestyle='--', linewidth=1)
                y_ticks.append(y_value)

            # Scatter plot for angles shifted by multiples of 2π
            for k in range(-num_shifts, num_shifts + 1):
                angles_shifted = angles_fitted + k * 2 * np.pi
                ax2.scatter(
                    stimulus_order,
                    angles_shifted,
                    c=stimulus_order,
                    cmap=cmap_unique,
                    norm=norm,
                    s=300,#100,
                    edgecolors='black',
                    alpha=0.7,
                )

            # Plot the regression fit line and its wrapped versions
            x_fit = np.linspace(0, stimulus_order_unique.max() + 1, 500)
            theta_fit = 2 * np.pi * s_fit * x_fit + b_fit
            for k in range(-num_shifts, num_shifts + 1):
                ax2.plot(x_fit, theta_fit + k * 2 * np.pi, color='black', linestyle='-', linewidth=2)

            # Set y-limits based on the shifted data and regression lines
            # Collect all shifted angles and regression lines to determine min and max
            angles_all_shifts = [angles_fitted + k * 2 * np.pi for k in range(-num_shifts, num_shifts + 1)]
            angles_all_shifts = np.concatenate(angles_all_shifts)
            theta_fit_shifts = [theta_fit + k * 2 * np.pi for k in range(-num_shifts, num_shifts + 1)]
            theta_fit_all = np.concatenate(theta_fit_shifts)

            # Determine overall min and max
            y_min = min(np.min(angles_all_shifts), np.min(theta_fit_all))
            y_max = max(np.max(angles_all_shifts), np.max(theta_fit_all))

            # Add some margin
            margin = 0.1 * (y_max - y_min)
            #ax2.set_ylim(y_min - margin, y_max + margin)

            # Set y-ticks at multiples of π
            y_ticks = np.arange(np.floor((y_min - margin) / (np.pi)), np.ceil((y_max + margin) / (np.pi)) + 1) * np.pi
            ax2.set_yticks(y_ticks)
            ax2.set_yticklabels([f'{(ytick / np.pi):.0f}π' for ytick in y_ticks])

            ax2.set_ylim(-3,11)
            # -----------------------------
            # Plot Null Distribution and Observed Error
            # -----------------------------
            '''
            plt.figure(figsize=(8, 6))
            plt.hist(perm_errors, bins=50, color='gray', alpha=0.7)
            plt.axvline(avg_error, color='red', linestyle='--', linewidth=5, label='Observed Avg Error')
            plt.xlabel('Average Circular Error (rad)', fontsize=14)
            plt.ylabel('Frequency', fontsize=14)
            plt.title(f'Null Distribution of Avg Circular Errors\n{condition_labels[cond]}\nPermutation p-value: {p_value_perm:.4f}', fontsize=16)
            plt.legend(fontsize=12)
            plt.tight_layout()
            perm_plot_filename = os.path.join(permutation_dir, f'{REGION_TYPE}_Permutation_Error_Distribution_{condition_labels[cond]}.pdf')
            plt.savefig(perm_plot_filename, dpi=300)
            plt.close()
            '''
        except Exception as e:
            print(f"An error occurred in Condition {cond + 1}: {e}", file=sys.stderr)
            average_circular_errors.append(np.nan)
            p_values.append(np.nan)
            p_values_permutation.append(np.nan)
            null_distributions.append([])
            if num_conditions > 1:
                ax1, ax2 = axes_all[cond, 0], axes_all[cond, 1]
            else:
                ax1, ax2 = axes_all[0], axes_all[1]
            ax1.text(0.5, 0.5, 'Error Occurred', horizontalalignment='center',
                     verticalalignment='center', transform=ax1.transAxes, color='red', fontsize=12)
            ax2.text(0.5, 0.5, 'Error Occurred', horizontalalignment='center',
                     verticalalignment='center', transform=ax2.transAxes, color='red', fontsize=12)
            ax1.set_title(f'{condition_labels[cond]}\nError', fontsize=14)
            ax2.set_title(f'Condition {cond + 1}\nError', fontsize=14)

    # Adjust layout and save the combined multi-panel figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    combined_filename = os.path.join(plots_dir, f'{REGION_TYPE}_Combined_Conditions_MultiPanel.pdf')
    fig_all.savefig(combined_filename, dpi=300)
    plt.close(fig_all)

    # -----------------------------
    # 6. Aggregated Results
    # -----------------------------

    # Aggregated Avg Circ-Lin Fit Error Results
    plt.figure(figsize=(8, 8))
    plt.bar(range(1, num_conditions + 1), average_circular_errors, color='skyblue')
    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel('Condition', fontsize=14)
    plt.ylabel('Avg Circ-Lin Fit Error (rad)', fontsize=14)
    plt.title(f'Avg Circ-Lin Fit Error Across Conditions, Region {REGION_TYPE}', fontsize=16)
    plt.ylim(0, max([val for val in average_circular_errors if not np.isnan(val)]) * 1.1 if average_circular_errors else 1)
    plt.xticks(range(1, num_conditions + 1), condition_labels, fontsize=12, rotation=45, ha='right')
    plt.yticks(fontsize=12)
    # Annotate bars with error values and p-values
    for idx, val in enumerate(average_circular_errors):
        if not np.isnan(val):
            plt.text(idx + 1, val + 0.02, f"{val:.2f}\np={p_values_permutation[idx]:.4f}", ha='center', fontsize=10)
    # Save the plot
    agg_error_filename = os.path.join(aggregated_dir, f'{REGION_TYPE}_Aggregated_Average_Circular_Errors.pdf')
    plt.savefig(agg_error_filename, dpi=300, bbox_inches='tight')
    plt.close()

    # -----------------------------
    # 7. Combined Null Distributions Plot
    # -----------------------------

    # Plot observed errors on their respective null distributions
    for idx in range(num_conditions):
        perm_errors = null_distributions[idx]
        avg_error = average_circular_errors[idx]
        p_value_perm = p_values_permutation[idx]

        if perm_errors:
            plt.figure(figsize=(8,6))
            plt.hist(perm_errors, bins=50, color='gray', alpha=0.7)
            plt.axvline(avg_error, color='red', linestyle='--', linewidth=5, label=f'Observed Avg Error\np={p_value_perm:.4f}')
            plt.title(f'Null Distribution of Avg Circular Errors\n{condition_labels[idx]}\nPermutation p-value: {p_value_perm:.4f}', fontsize=16)
            plt.xlabel('Average Circular Error (rad)', fontsize=14)
            plt.ylabel('Frequency', fontsize=14)
            #plt.legend(fontsize=12)
            plt.tight_layout()
            perm_plot_filename = os.path.join(permutation_dir, f'{REGION_TYPE}_Permutation_Error_Distribution_{condition_labels[idx]}.pdf')
            plt.savefig(perm_plot_filename, dpi=300)
            plt.close()

    print("\nAll plots have been saved in the 'Analysis_Plots' directory.")

if __name__ == "__main__":
    main()

