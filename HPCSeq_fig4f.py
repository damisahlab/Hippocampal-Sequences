from utils import pdfTextSet
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import RipleysKEstimator

# Update global font size
plt.rcParams.update({
    'font.size': 14,          # Default font size for text
    'axes.titlesize': 18,     # Font size for titles
    'axes.labelsize': 16,     # Font size for axis labels
    'xtick.labelsize': 14,    # Font size for x-axis tick labels
    'ytick.labelsize': 14     # Font size for y-axis tick labels
    })

# Set random seed for reproducibility
rng = np.random.default_rng(seed=42)

# Number of points and iterations
num_points = 300  # Updated to 300 as per the last assignment
num_iterations = 30
MIN_PERC=0.1
MAX_PERC=99.9

# Parameters for the circular distribution
circle_center = np.array([7.5, 7.5])  # Center of the circle (x, y)
circle_center = np.array([10,10])
circle_radius = 2                     # Radius of the circle
circle_radius = 6                     # Radius of the circle
circle_noise_std = 0.15               # Standard deviation of the Gaussian noise
circle_noise_std = 0.1               # Standard deviation of the Gaussian noise
circle_noise_std = circle_radius/10
#circle_noise_std = 0.02               # Standard deviation of the Gaussian noise

# Parameters for the clustered distribution
num_clusters = 20              # Increased number of clusters
points_per_cluster = 20        # Points per cluster
cluster_noise_std = 0.8        # Standard deviation of points around cluster centers

# Define the range of radii to evaluate the H function
r = np.linspace(0, circle_radius*2, 100)  # Adjusted radii based on circle radius
r = np.linspace(0, circle_radius*4, 100)  # Adjusted radii based on circle radius

# Initialize arrays to store H(r) for each iteration and data arrangement
# Data Arrangements: 0 - Poisson, 1 - Circular, 2 - Clustered
H_all = np.full((3, num_iterations, len(r)), np.nan)

# Initialize variables to store the first iteration's data for plotting
first_data = [None, None, None]          # [Poisson, Circular, Clustered]
first_study_areas = [None, None, None]  # [(x_min, x_max, y_min, y_max), ...]

# Perform iterations
for i in range(num_iterations):
    # --- 1. Generate Poisson (Completely Random) Data ---
    poisson_data = rng.uniform(low=0, high=20, size=(num_points, 2))  # Initial generation with a broad range

    # Derive study area from 1st and 99th percentiles
    x_min_p, x_max_p = np.percentile(poisson_data[:, 0], [MIN_PERC, MAX_PERC])
    y_min_p, y_max_p = np.percentile(poisson_data[:, 1], [MIN_PERC, MAX_PERC])

    # Clip data to the derived study area
    poisson_data_clipped = poisson_data.copy()
    poisson_data_clipped[:, 0] = np.clip(poisson_data_clipped[:, 0], x_min_p, x_max_p)
    poisson_data_clipped[:, 1] = np.clip(poisson_data_clipped[:, 1], y_min_p, y_max_p)

    if i == 0:
        first_data[0] = poisson_data_clipped.copy()
        first_study_areas[0] = (x_min_p, x_max_p, y_min_p, y_max_p)

    # Calculate area
    area_p = (x_max_p - x_min_p) * (y_max_p - y_min_p)

    # Initialize Ripley's H function estimator with dynamic study area for Poisson
    Hest_poisson = RipleysKEstimator(area=area_p, x_min=x_min_p, x_max=x_max_p, y_min=y_min_p, y_max=y_max_p)

    try:
        # Calculate Ripley's H function for Poisson data
        H_poisson = Hest_poisson.Hfunction(poisson_data_clipped, r, mode='ripley')
        H_all[0, i, :] = H_poisson
    except Exception as e:
        print(f"Iteration {i+1}: Error computing H(r) for Poisson data - {e}")

    # --- 2. Generate Noisy Circular Data ---
    angles = rng.uniform(0, 2 * np.pi, num_points)
    x_circle = circle_center[0] + circle_radius * np.cos(angles) + rng.normal(0, circle_noise_std, num_points)
    y_circle = circle_center[1] + circle_radius * np.sin(angles) + rng.normal(0, circle_noise_std, num_points)
    noisy_circle_data = np.column_stack((x_circle, y_circle))

    # Derive study area from 1st and 99th percentiles
    x_min_c, x_max_c = np.percentile(noisy_circle_data[:, 0], [MIN_PERC,MAX_PERC])
    y_min_c, y_max_c = np.percentile(noisy_circle_data[:, 1], [MIN_PERC, MAX_PERC])

    # Clip data to the derived study area
    noisy_circle_data_clipped = noisy_circle_data.copy()
    noisy_circle_data_clipped[:, 0] = np.clip(noisy_circle_data_clipped[:, 0], x_min_c, x_max_c)
    noisy_circle_data_clipped[:, 1] = np.clip(noisy_circle_data_clipped[:, 1], y_min_c, y_max_c)

    if i == 0:
        first_data[1] = noisy_circle_data_clipped.copy()
        first_study_areas[1] = (x_min_c, x_max_c, y_min_c, y_max_c)

    # Calculate area
    area_c = (x_max_c - x_min_c) * (y_max_c - y_min_c)

    # Initialize Ripley's H function estimator with dynamic study area for Circular
    Hest_circle = RipleysKEstimator(area=area_c, x_min=x_min_c, x_max=x_max_c, y_min=y_min_c, y_max=y_max_c)

    try:
        # Calculate Ripley's H function for Noisy Circular data
        H_circle = Hest_circle.Hfunction(noisy_circle_data_clipped, r, mode='ripley')
        H_all[1, i, :] = H_circle
    except Exception as e:
        print(f"Iteration {i+1}: Error computing H(r) for Circular data - {e}")

    # --- 3. Generate Randomly Arranged Clustered Data ---
    # Randomly select cluster centers within a broad range to allow dynamic study area
    cluster_centers = rng.uniform(low=0, high=20, size=(num_clusters, 2))
    clustered_data = []
    for center_cluster in cluster_centers:
        # Generate points around each cluster center with Gaussian noise
        cluster_points = rng.normal(loc=center_cluster, scale=cluster_noise_std, size=(points_per_cluster, 2))
        clustered_data.append(cluster_points)
    clustered_data = np.vstack(clustered_data)

    # Derive study area from 1st and 99th percentiles
    x_min_cl, x_max_cl = np.percentile(clustered_data[:, 0], [MIN_PERC, MAX_PERC])
    y_min_cl, y_max_cl = np.percentile(clustered_data[:, 1], [MIN_PERC, MAX_PERC])

    # Clip data to the derived study area
    clustered_data_clipped = clustered_data.copy()
    clustered_data_clipped[:, 0] = np.clip(clustered_data_clipped[:, 0], x_min_cl, x_max_cl)
    clustered_data_clipped[:, 1] = np.clip(clustered_data_clipped[:, 1], y_min_cl, y_max_cl)

    if i == 0:
        first_data[2] = clustered_data_clipped.copy()
        first_study_areas[2] = (x_min_cl, x_max_cl, y_min_cl, y_max_cl)

    # Calculate area
    area_cl = (x_max_cl - x_min_cl) * (y_max_cl - y_min_cl)

    # Initialize Ripley's H function estimator with dynamic study area for Clustered
    Hest_clusters = RipleysKEstimator(area=area_cl, x_min=x_min_cl, x_max=x_max_cl, y_min=y_min_cl, y_max=y_max_cl)

    try:
        # Calculate Ripley's H function for Clustered data
        H_clusters = Hest_clusters.Hfunction(clustered_data_clipped, r, mode='ripley')
        H_all[2, i, :] = H_clusters
    except Exception as e:
        print(f"Iteration {i+1}: Error computing H(r) for Clustered data - {e}")

# Compute mean and SEM for H functions using nanmean and nanstd
H_mean = np.nanmean(H_all, axis=1)  # Shape: (3, len(r))
H_std = np.nanstd(H_all, axis=1, ddof=1)
H_sem = H_std / np.sqrt(np.sum(~np.isnan(H_all), axis=1))

# Theoretical Ripley's H function under CSR for Poisson is 0
H_theoretical = np.zeros_like(r)

# Compute difference between observed mean H(r) and theoretical H(r)
diff_mean = H_mean - H_theoretical  # Observed - Theoretical
diff_sem = H_sem  # Assuming SEM is symmetric for differences

# --- Plotting ---
fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # Changed to 2 rows and 3 columns

# Data Arrangement Names
data_names = ['Poisson (Random)', 'Noisy Circular', 'Randomly Clustered']

# Colors for plots
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Distinct colors for each dataset

for col in range(3):
    # --- Scatter Plot (Top Row) ---
    axes[0, col].scatter(first_data[col][:, 0], first_data[col][:, 1],
                        color=colors[col], edgecolor=None, alpha=0.7, s=50, marker='o')
    axes[0, col].set_title(f'{data_names[col]}', fontsize=16, fontweight='bold')
    # Set limits based on the first iteration's study area
    x_min_plot, x_max_plot, y_min_plot, y_max_plot = first_study_areas[col]
    #axes[0, col].set_xlim(x_min_plot, x_max_plot)
    #axes[0, col].set_ylim(y_min_plot, y_max_plot)
    axes[0, col].set_xlim(2,18)
    axes[0, col].set_ylim(2,18)
    
    axes[0, col].set_xlabel('X Coordinate', fontsize=14)
    axes[0, col].set_ylabel('Y Coordinate', fontsize=14)

    # --- Ripley's H Function Plot (Bottom Row) ---
    axes[1, col].plot(r, H_mean[col], label='Mean Observed H(r)', color=colors[col], linewidth=2)
    axes[1, col].fill_between(r, H_mean[col] - H_sem[col], H_mean[col] + H_sem[col],
                              color=colors[col], alpha=0.3, label='Mean ± SEM')
    axes[1, col].plot(r, H_theoretical, label='Theoretical CSR H(r)', color='black', linestyle='--', linewidth=2)
    axes[1, col].set_title(f"Ripley's H Function for {data_names[col]} Data", fontsize=16, fontweight='bold')
    axes[1, col].set_xlabel('Distance (r)', fontsize=14)
    axes[1, col].set_ylabel("H(r)", fontsize=14)
    axes[1, col].legend(fontsize=12)
    axes[1, col].set_xlim([0,11])

    # Optionally, set y-limits for H function plots for consistency
    if col == 2:
        axes[1, col].set_ylim(-1.5, 1.5)  # Adjust as needed
    else:
        #axes[1, col].set_ylim(-0.5, 0.5)  # Adjust as needed
        #axes[1, col].set_ylim(None, None)  # Adjust as needed
        axes[1, col].set_ylim(-1.5, 1.5)  # Adjust as needed

# Add overall titles for better interpretation
fig.suptitle("Comparison of Simulated Datasets and Ripley's H Function", fontsize=20, fontweight='bold', y=0.95)

# Adjust layout for better spacing
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save the combined plot to a file
plt.savefig('ripley_h_comparison_mean_sem_2rows_3columns.pdf', dpi=300)

