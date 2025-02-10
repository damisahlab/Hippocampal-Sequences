from utils import tdaHelpers as tda
from utils import pdfTextSet
import pandas as pd
import pdb
import numpy as np
from ripser import ripser
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap  # Ensure Isomap is imported
import os
from persim import bottleneck  # Retained import, but not used in gap calculation
from scipy.ndimage import gaussian_filter  # Added for density heatmap
import matplotlib.patches as patches  # Added for ellipse plotting
from matplotlib.patches import Ellipse
import math  # Added for angle calculations
from scipy.stats import circmean, sem  # Added sem for standard error of the mean
from utils import kempter  # **Added for circular-linear regression**
from utils import ripley_module_3 as ripley  # [ADDED] Import Ripley Module
from scipy.interpolate import interp1d  # [ADDED] Import interp1d for interpolation
from sklearn.metrics import pairwise_distances  # **Added for nt-TDA preprocessing**

# -----------------------
# Configuration
# -----------------------

# Flag to toggle between raw change and percent change
use_percent_change = False  # Set to True for percent change, False for raw change

FIT_ELLIPSE = False
#USE_PCA = True  # Ensure PCA is enabled
#USE_PCA = False  # Ensure PCA is enabled
desired_components = 10
desired_components = 3
#desired_components =6
#desired_components =5
desired_components =4
#desired_components = 8
#desired_components = 7
#desired_components = 8
#desired_components = 15
#desired_components = 30
#desired_components = 16
#desired_components = 20
N_TRIALS=5
N_TRIALS=6

#N_TRIALS=5
#N_TRIALS=7
#N_TRIALS=3
#N_TRIALS=6
#N_TRIALS=4

SMOOTH_RIPLEY=True
SMOOTH_RIPLEY=False

# Optional: Flags for Isomap and Visualization
USE_ISOMAP = True  # Set to True to enable Isomap after PCA
#USE_ISOMAP = False# Set to True to enable Isomap after PCA
ISOMAP_COMPONENTS = 2  # Number of components for Isomap
#ISOMAP_COMPONENTS = 5  # Number of components for Isomap
ISOMAP_NEIGHBORS = 2  # Number of neighbors for Isomap (default is 5)
#ISOMAP_NEIGHBORS = 1  # Number of neighbors for Isomap (default is 5)
#ISOMAP_NEIGHBORS = 3  # Number of neighbors for Isomap (default is 5)
#ISOMAP_NEIGHBORS = 4  # Number of neighbors for Isomap (default is 5)
VISUALIZE_STEPS = True  # Set to True to enable intermediate visualizations

# Optional: Flag to enable coloring by scene
COLOR_BY_SCENE = True  # Set to True to color data points by scene
COLOR_BY_SCENE = False  # Overwritten to False as per original code

# Optional: Flag to enable standardization
STANDARDIZE_DATA = True  # Set to True to standardize data by dividing by std in each dimension
#STANDARDIZE_DATA = False# Set to True to standardize data by dividing by std in each dimension

TDA_FILTER=False
TDA_FILTER=True
# Permutation Test Parameters

#num_permutation = 200  # Number of permutation samples for testing
#num_bootstrap=200
num_permutation = 1000  # Number of permutation samples for testing
num_bootstrap=1000
# -----------------------
# New Global Variables for nt-TDA Preprocessing
# -----------------------
DISTANCE_PERCENTILE_TO_DROP = 20# Percentile to determine neighborhood radius
NEIGHBOR_PERCENTILE = 5        # Percentile to set neighbor count threshold
import numpy as np
from sklearn.metrics import pairwise_distances

#def nt_tda_preprocessing(data, radius=40, min_neighbors=1):
#def nt_tda_preprocessing(data, radius=0.5, min_neighbors=3):
def nt_tda_preprocessing(data, radius=0.5, min_neighbors=1):
#def nt_tda_preprocessing(data, radius=1, min_neighbors=1):
#def nt_tda_preprocessing(data, radius=0.75, min_neighbors=2):
#def nt_tda_preprocessing(data, radius=50, min_neighbors=3):
#def nt_tda_preprocessing(data, radius=50, min_neighbors=2):
    """
    Perform nt-TDA preprocessing by excluding outliers based on a fixed neighborhood radius and minimum neighbor count.
    
    Parameters:
    - data (np.ndarray): 2D array of shape (n_samples, n_features).
    - radius (float): Fixed radius to define the neighborhood around each point (default: 50).
    - min_neighbors (int): Minimum number of neighbors required to keep a point (default: 2).
    
    Returns:
    - filtered_data (np.ndarray): Data after excluding outliers.
    """
    # Compute pairwise Euclidean distances
    print("Computing pairwise distances...")
    pairwise_dist = pairwise_distances(data, metric='euclidean')
    
    # Count neighbors within the fixed radius for each point
    print(f"Counting neighbors within radius: {radius}")
    # For each point, count how many distances are <= radius
    neighbor_counts = np.sum(pairwise_dist <= radius, axis=1) - 1  # subtract 1 to exclude the point itself
    
    # Identify points to keep (those with at least min_neighbors)
    print(f"Keeping points with >= {min_neighbors} neighbors...")
    points_to_keep = neighbor_counts >= min_neighbors
    num_removed = np.sum(~points_to_keep)
    print(f"Removing {num_removed} outliers based on neighbor count.")
    
    # Filter the data
    filtered_data = data[points_to_keep]
    print(f"Data shape before preprocessing: {data.shape}")
    print(f"Data shape after nt-TDA preprocessing: {filtered_data.shape}")
    
    return filtered_data



def main():
    # Set random seed for reproducibility
    np.random.seed(2)

    # Define regions to process
    regions = ['aic', 'hpc']

    # Define the grouping columns
    # **Change**: Remove 'phase' from group_cols to group by condition and region_type only
    group_cols = ['condition', 'region_type']  # Modified grouping

    collected_ripley_data = {
        'hpc': {
            'random': [],
            'structured': []
        },
        'aic': {
            'random': [],
            'structured': []
        }
    }


    # Define output directories
    barcode_output_dir = 'barcode_plots_combined_Jan16a_2025'
    gap_output_dir = 'gap_distributions_plotted_Jan16a_2025'
    change_plot_dir = 'change_in_max_persistence_Jan16a_2025'
    projection_plot_dir = 'projection_plots_Jan16a_2025'  # New directory for projections

    # [ADDED] Define output directories for Ripley's H analyses
    ripley_plots_dir = 'ripley_h_diff_plots_Jan16a_2025'
    ripley_summary_plot_dir = 'ripley_h_diff_summary_plots_Jan16a_2025'
    os.makedirs(barcode_output_dir, exist_ok=True)
    os.makedirs(gap_output_dir, exist_ok=True)
    os.makedirs(change_plot_dir, exist_ok=True)
    if VISUALIZE_STEPS:
        os.makedirs(projection_plot_dir, exist_ok=True)  # Create if visualizations are enabled
    os.makedirs(ripley_plots_dir, exist_ok=True)  # [ADDED] Create Ripley's H plots directory
    os.makedirs(ripley_summary_plot_dir, exist_ok=True)  # [ADDED] Create Ripley's H summary plots directory

    # Permutation Test Parameters
    betti_nums_of_interest = [1]  # Betti-1 only as per your adaptation

    # Define color mapping based on region and condition
    color_mapping = {
        'hpc': {
            'structured': 'red',
            'random': 'lightcoral'
        },
        'aic': {
            'structured': 'brown',
            'random': '#C4A484'
        }
    }

    # If coloring by scene is enabled, define scene colors
    if COLOR_BY_SCENE:
        cmap = plt.get_cmap('twilight')
        scene_colors = {}
        num_scenes = 10  # Assuming 10 scenes
        for sceneIdx in range(num_scenes):
            # Normalize scene index to [0, 1] for the colormap
            scene_colors[sceneIdx + 1] = cmap(sceneIdx / (num_scenes - 1))
        # Display the colors for verification
        for scene, color in scene_colors.items():
            print(f"Scene {scene}: {color}")

    # [ADDED] Initialize dictionaries to accumulate H_diff per condition
    all_h_diffs = defaultdict(list)  # condition: list of (distances, H_diff) tuples

    # Iterate over each region
    for REGION in regions:
        print(f"\n{'='*20} Processing Region: {REGION.upper()} {'='*20}\n")

        # Load the DataFrame for the current region
        #data_frame_path = f'super_sess_data_frame_{REGION}.pkl'
        #data_frame_path = f'super_sess_data_frame_{REGION}_AvgTrials_False.pkl'
        data_frame_path = f'super_sess_data_frame_{REGION}_AvgTrials_False_N_5.pkl'
        data_frame_path = f'super_sess_data_frame_{REGION}_AvgTrials_False_N_{N_TRIALS}.pkl'
        #data_frame_path = f'super_sess_data_frame_{REGION}_AvgTrials_False_N_5_NoMacros.pkl'
        #data_frame_path = f'super_sess_data_frame_{REGION}_AvgTrials_True.pkl'
        if not os.path.exists(data_frame_path):
            print(f"Data file {data_frame_path} does not exist. Skipping region {REGION}.\n")
            continue

        df_records = pd.read_pickle(data_frame_path)

        # Initialize dictionaries for the current region
        barcodes = defaultdict(dict)
        max_persistence = defaultdict(lambda: defaultdict(dict))

        # Collect barcode data per group
        groups_data = {}

        # **New Structure**: Group by condition and region_type only
        for group_name, group_df in df_records.groupby(group_cols):
            condition, region_type = group_name  # Removed phase
            print(f"Processing group: Condition={condition}, Region Type={region_type}")

            # Separate early and late phases within the group
            early_df = group_df[group_df['phase'] == 'early']
            late_df = group_df[group_df['phase'] == 'late']

            if early_df.empty or late_df.empty:
                print(f"One of the phases is missing for group {group_name}. Skipping this group.\n")
                continue

            # Collect all feature matrices for early and late
            feature_matrices_early = early_df['super_sess_feature_matrix'].values
            feature_matrices_late = late_df['super_sess_feature_matrix'].values

            # Concatenate all feature matrices along the first axis (sessions)
            try:
                data_early = np.vstack(feature_matrices_early)
                data_late = np.vstack(feature_matrices_late)
            except ValueError as e:
                print(f"Error concatenating feature matrices for group {group_name}: {e}. Skipping this group.\n")
                continue

            # **Ensure that both early and late have the same number of columns for permutation**
            n_samples_early, n_features_early = data_early.shape
            n_samples_late, n_features_late = data_late.shape

            # It's essential that n_features_early == n_features_late
            if n_features_early != n_features_late:
                print(f"Mismatch in number of features between early and late for group {group_name}. Skipping.\n")
                continue

            # ---------------------------
            # Dimensionality Reduction
            # ---------------------------

            # **Early Phase: PCA and Isomap**
            max_components_early = min(desired_components, n_samples_early - 1, n_features_early)
            if max_components_early < desired_components:
                print(f"Adjusting n_components from {desired_components} to {max_components_early} for early phase in group {group_name}.")

            if max_components_early < 2:
                print(f"Not enough samples/features to perform PCA for early phase in group {group_name}. Skipping this group.\n")
                continue  # Skip this group

            ## Perform PCA on early data
            #pca_early = PCA(n_components=max_components_early, random_state=42)
            pca_early = PCA(n_components=max_components_early, svd_solver='full')
            #pca_early = PCA(n_components=max_components_early)
            data_reduced_early = pca_early.fit_transform(data_early)
            print(f"Early Phase: Data shape after PCA: {data_reduced_early.shape}")

            # Apply Isomap to early data
            if USE_ISOMAP:
                isomap_early = Isomap(n_neighbors=ISOMAP_NEIGHBORS, n_components=ISOMAP_COMPONENTS)
                try:
                    data_reduced_early = isomap_early.fit_transform(data_reduced_early)
                    print(f"Early Phase: Data shape after Isomap: {data_reduced_early.shape}")
                except Exception as e:
                    print(f"Isomap failed for early phase in group {group_name}: {e}. Skipping Isomap.\n")
                    continue

            # **Late Phase: PCA and Isomap**
            max_components_late = min(desired_components, n_samples_late - 1, n_features_late)
            if max_components_late < desired_components:
                print(f"Adjusting n_components from {desired_components} to {max_components_late} for late phase in group {group_name}.")

            if max_components_late < 2:
                print(f"Not enough samples/features to perform PCA for late phase in group {group_name}. Skipping this group.\n")
                continue  # Skip this group

            # Perform PCA on late data
            #pca_late = PCA(n_components=max_components_late, random_state=42)
            #pca_late = PCA(n_components=max_components_late)
            #pca_late = PCA(n_components=max_components_late)
            pca_late = PCA(n_components=max_components_early, svd_solver='full')
            data_reduced_late = pca_late.fit_transform(data_late)
            print(f"Late Phase: Data shape after PCA: {data_reduced_late.shape}")

            # Apply Isomap to late data
            if USE_ISOMAP:
                isomap_late = Isomap(n_neighbors=ISOMAP_NEIGHBORS, n_components=ISOMAP_COMPONENTS)
                try:
                    data_reduced_late = isomap_late.fit_transform(data_reduced_late)
                    print(f"Late Phase: Data shape after Isomap: {data_reduced_late.shape}")
                except Exception as e:
                    print(f"Isomap failed for late phase in group {group_name}: {e}. Skipping Isomap.\n")
                    continue

            
            # ---------------------------
            # [ADDED] Standardization
            # ---------------------------
            if STANDARDIZE_DATA:
                # Standardize early data
                std_early = data_reduced_early.std(axis=0)
                # To avoid division by zero, set std to 1 where std is zero
                std_early[std_early == 0] = 1
                data_reduced_early = data_reduced_early / std_early
                print(f"Early Phase: Data standardized by dividing by std in each dimension.")

                # Standardize late data
                std_late = data_reduced_late.std(axis=0)
                # To avoid division by zero, set std to 1 where std is zero
                std_late[std_late == 0] = 1
                data_reduced_late = data_reduced_late / std_late
                print(f"Late Phase: Data standardized by dividing by std in each dimension.")
            # ---------------------------
            
            # ---------------------------
            # [ADDED] nt-TDA Preprocessing to Remove Outliers
            # ---------------------------
            print("    Performing nt-TDA preprocessing to remove outliers based on neighborhood criteria.")
            if TDA_FILTER:
                data_reduced_early = nt_tda_preprocessing(data_reduced_early)
                                                         #distance_percentile=DISTANCE_PERCENTILE_TO_DROP, 
                                                         #neighbor_percentile=NEIGHBOR_PERCENTILE)
                data_reduced_late = nt_tda_preprocessing(data_reduced_late)
                                                        #distance_percentile=DISTANCE_PERCENTILE_TO_DROP, 
                                                        #neighbor_percentile=NEIGHBOR_PERCENTILE)
            # ---------------------------

            # Optional: Visualize PCA and Isomap projections for early and late phases separately
            if VISUALIZE_STEPS:
                # Early Phase Projection Plot
                if ISOMAP_COMPONENTS >= 2:
                    if COLOR_BY_SCENE:
                        # Assign scene numbers based on data ordering
                        # Assuming every 10 entries correspond to the same scene
                        n_points_early = data_reduced_early.shape[0]
                        scene_numbers_early = [(i % 10) + 1 for i in range(n_points_early)]
                        colors_early = [scene_colors.get(scene, 'k') for scene in scene_numbers_early]
                    else:
                        colors_early = color_mapping.get(region_type.lower(), {}).get(condition.lower(), 'k')  # Default color

                    early_proj_filename = os.path.join(
                        projection_plot_dir,
                        f'early_projection_{REGION}_{condition}_{region_type}.pdf'
                    )
                    '''
                    plt.figure(figsize=(6, 6))
                    plt.scatter(data_reduced_early[:,0], data_reduced_early[:,1], c=colors_early, alpha=1)
                    plt.hexbin(data_reduced_early[:,0], data_reduced_early[:,1], gridsize=100, cmap='plasma', alpha=0.5)

                    plt.title(f'PCA + Isomap Projection - Early Phase - {group_name}')
                    plt.xlabel('Component 1')
                    plt.ylabel('Component 2')
                    
                    plt.savefig(early_proj_filename, dpi=300)
                    plt.close()
                    print(f"Saved early phase projection plot to {early_proj_filename}")
                    '''

                    tda.plot_projection(data_reduced_early, colors_early, 'Early', early_proj_filename, group_name,STANDARDIZE_DATA=STANDARDIZE_DATA)

                # Late Phase Projection Plot
                if ISOMAP_COMPONENTS >= 2:
                    if COLOR_BY_SCENE:
                        # Assign scene numbers based on data ordering
                        n_points_late = data_reduced_late.shape[0]
                        scene_numbers_late = [(i % 10) + 1 for i in range(n_points_late)]
                        colors_late = [scene_colors.get(scene, 'k') for scene in scene_numbers_late]
                    else:
                        colors_late = color_mapping.get(region_type.lower(), {}).get(condition.lower(), 'k')  # Default color
                    late_proj_filename = os.path.join(
                        projection_plot_dir,
                        f'late_projection_{REGION}_{condition}_{region_type}.pdf'
                    )
                    '''
                    plt.figure(figsize=(6, 6))
                    plt.scatter(data_reduced_late[:,0], data_reduced_late[:,1], c=colors_late, alpha=1)
                    plt.hexbin(data_reduced_late[:,0], data_reduced_late[:,1], gridsize=100, cmap='plasma', alpha=0.5)

                    plt.title(f'PCA + Isomap Projection - Late Phase - {group_name}')
                    plt.xlabel('Component 1')
                    plt.ylabel('Component 2')
                    
                    plt.savefig(late_proj_filename, dpi=300)
                    plt.close()
                    print(f"Saved late phase projection plot to {late_proj_filename}")
                    '''

                    tda.plot_projection(data_reduced_late, colors_late, 'Late', late_proj_filename, group_name,STANDARDIZE_DATA=STANDARDIZE_DATA)

            # -----------------------
            # [ADDED] Ripley's H Analysis
            # -----------------------
            '''
            if 'rand' in condition:
                print(condition)
                pdb.set_trace()
            else:
                print(condition)
                pdb.set_trace()
            '''
            # **Compute Ripley's H for Early and Late Phases**
            # Extract first two components for Ripley's H computation
            points_early = data_reduced_early[:, :ISOMAP_COMPONENTS]
            points_late = data_reduced_late[:, :ISOMAP_COMPONENTS]

            # Compute Ripley's H_deviation for early and late
            distances_early, H_deviation_early = ripley.compute_ripleys_H(points_early, SMOOTH_RIPLEY=SMOOTH_RIPLEY)
            distances_late, H_deviation_late = ripley.compute_ripleys_H(points_late,SMOOTH_RIPLEY=SMOOTH_RIPLEY)

            # Interpolate both H_deviation to common distances
            min_distance = max(distances_early.min(), distances_late.min())
            max_distance = min(distances_early.max(), distances_late.max())
            common_distances = np.linspace(min_distance, max_distance, num=1000)  # Adjust num as needed for smoothness

            interp_early = interp1d(distances_early, H_deviation_early, kind='linear', fill_value='extrapolate')
            interp_late = interp1d(distances_late, H_deviation_late, kind='linear', fill_value='extrapolate')
            H_deviation_early_interp = interp_early(common_distances)
            H_deviation_late_interp = interp_late(common_distances)

            # Compute H_diff = H_late - H_early
            H_diff = H_deviation_late_interp - H_deviation_early_interp
            distances_to_use = common_distances  # Use raw distances as normalize_distance is False

            # Check if H_diff is not all zeros
            if not np.allclose(H_diff, 0):
                # Accumulate all H_diff differences for plotting per condition
                all_h_diffs[condition].append((common_distances, H_diff))

                # [ADDED] Perform Bootstrap Resampling to Compute Confidence Intervals
                print(f"    Starting bootstrap resampling with {num_bootstrap} samples for group {group_name}.")

                # Initialize array to store H_diff from each bootstrap sample
                H_diff_bootstrap = np.zeros((num_bootstrap, len(common_distances)))

                for boot in range(num_bootstrap):
                    if (boot + 1) % 1000 == 0:
                        print(f"        Bootstrap sample {boot + 1}/{num_bootstrap}")

                    # Bootstrap resample early data with replacement
                    indices_early = np.random.choice(points_early.shape[0], size=points_early.shape[0], replace=True)
                    resampled_early = points_early[indices_early]

                    # Bootstrap resample late data with replacement
                    indices_late = np.random.choice(points_late.shape[0], size=points_late.shape[0], replace=True)
                    resampled_late = points_late[indices_late]

                    # Compute Ripley's H for resampled data
                    distances_boot_early, H_deviation_boot_early = ripley.compute_ripleys_H(resampled_early, SMOOTH_RIPLEY=SMOOTH_RIPLEY)
                    distances_boot_late, H_deviation_boot_late = ripley.compute_ripleys_H(resampled_late, SMOOTH_RIPLEY=SMOOTH_RIPLEY)

                    # Interpolate to common distances
                    interp_boot_early = interp1d(distances_boot_early, H_deviation_boot_early, kind='linear', fill_value='extrapolate')
                    interp_boot_late = interp1d(distances_boot_late, H_deviation_boot_late, kind='linear', fill_value='extrapolate')
                    H_deviation_boot_early_interp = interp_boot_early(common_distances)
                    H_deviation_boot_late_interp = interp_boot_late(common_distances)

                    # Compute H_diff for bootstrapped data
                    H_diff_boot = H_deviation_boot_late_interp - H_deviation_boot_early_interp

                    # Store the bootstrapped H_diff
                    H_diff_bootstrap[boot, :] = H_diff_boot

                # Remove any failed bootstraps (rows with NaNs)
                valid_boots = ~np.isnan(H_diff_bootstrap).any(axis=1)
                H_diff_bootstrap = H_diff_bootstrap[valid_boots]
                actual_num_boots = H_diff_bootstrap.shape[0]
                print(f"    Completed bootstrap resampling. Valid samples: {actual_num_boots}/{num_bootstrap}")

                # Compute 2.5th and 97.5th percentiles for 95% confidence interval
                lower_ci = np.percentile(H_diff_bootstrap, 2.5, axis=0)
                upper_ci = np.percentile(H_diff_bootstrap, 97.5, axis=0)

                '''
                # [ADDED] Plot Ripley's H_diff with Confidence Intervals
                ripley.plot_ripleys_H_diff(
                    common_distances=common_distances,
                    observed_diff=H_diff,
                    condition=condition,
                    region_type=region_type,
                    session_id='AllPatients',  # Fixed session ID as per super session
                    output_dir=ripley_plots_dir,
                    points_early=points_early,  # [ADDED] Pass early points
                    points_late=points_late,    # [ADDED] Pass late points
                    lower_ci=lower_ci,          # [ADDED] Pass lower confidence interval
                    upper_ci=upper_ci           # [ADDED] Pass upper confidence interval
                )

                print(f"    Saved Ripley's H difference plot with confidence intervals.")
                '''



            # Collect data instead of plotting
            ripley_data = ripley.collect_ripleys_H_diff_data(
                common_distances=common_distances,
                observed_diff=H_diff,
                condition=condition,
                region_type=region_type,
                session_id='AllPatients',  # Fixed session ID as per super session
                points_early=points_early,  # Optional: can be omitted if not needed
                points_late=points_late,    # Optional: can be omitted if not needed
                lower_ci=lower_ci,
                upper_ci=upper_ci,
                normalize_distance=True,
                normalize_y=False,
            )
            
            # Append the data to the appropriate list in the dictionary
            if region_type in collected_ripley_data:
                if condition in collected_ripley_data[region_type]:
                    collected_ripley_data[region_type][condition].append(ripley_data)
                else:
                    print(f"Warning: Condition '{condition}' not recognized for region '{region_type}'.")
            else:
                print(f"Warning: Region type '{region_type}' not recognized.")

                
            # ---------------------------
            # Persistent Homology Computation
            # ---------------------------

            # **Compute persistent homology for early and late data**
            # Only Betti-1 is of interest
            result_early = tda.compute_persistent_homology(data_reduced_early, maxdim=1)
            result_late = tda.compute_persistent_homology(data_reduced_late, maxdim=1)

            # Store the barcodes
            barcodes[group_name]['betti_1_early'] = result_early['dgms'][1]
            barcodes[group_name]['betti_1_late'] = result_late['dgms'][1]

            print(f"Computed Betti-1 numbers for group: {group_name}")

            # -----------------------
            # Permutation Procedure
            # -----------------------

            # Original persistence diagrams for Betti-1
            original_dgm_1_early = result_early['dgms'][1]
            original_dgm_1_late = result_late['dgms'][1]

            # Compute observed gaps for Betti-1
            observed_gap_1_early = tda.compute_gap(original_dgm_1_early)
            observed_gap_1_late = tda.compute_gap(original_dgm_1_late)

            # Check for NaNs in observed gaps
            if np.isnan(observed_gap_1_early) or np.isnan(observed_gap_1_late):
                print(f"Observed gap for Betti 1 is NaN for group {group_name}. Skipping permutation for Betti 1.\n")
                observed_gap_diff_1 = np.nan
                p_value_1 = np.nan
            else:
                # Compute observed difference in gaps (late - early)
                observed_gap_diff_1 = observed_gap_1_late - observed_gap_1_early

            # Initialize list to store permutation gap differences for Betti-1
            permutation_gap_diff_1 = []

            # Perform permutation testing
            for i in range(num_permutation):
                # Permute early data
                permuted_data_early = np.empty_like(data_reduced_early)
                for col in range(data_reduced_early.shape[1]):
                    perm = np.random.permutation(data_reduced_early.shape[0])
                    permuted_data_early[:, col] = data_reduced_early[perm, col]

                # Permute late data
                permuted_data_late = np.empty_like(data_reduced_late)
                for col in range(data_reduced_late.shape[1]):
                    perm = np.random.permutation(data_reduced_late.shape[0])
                    permuted_data_late[:, col] = data_reduced_late[perm, col]

                # Compute persistent homology for permuted early and late data
                permutation_result_early = tda.compute_persistent_homology(permuted_data_early, maxdim=1)
                permutation_result_late = tda.compute_persistent_homology(permuted_data_late, maxdim=1)

                # Extract permutation persistence diagrams for Betti-1
                permutation_dgm_1_early = permutation_result_early['dgms'][1]
                permutation_dgm_1_late = permutation_result_late['dgms'][1]

                # Compute gaps for Betti-1
                gap_1_early = tda.compute_gap(permutation_dgm_1_early)
                gap_1_late = tda.compute_gap(permutation_dgm_1_late)

                # Compute difference in gaps (late - early)
                if not np.isnan(gap_1_early) and not np.isnan(gap_1_late):
                    gap_diff_1 = gap_1_late - gap_1_early
                    permutation_gap_diff_1.append(gap_diff_1)
                else:
                    # Skip this permutation if any gap is NaN
                    pass

            # Convert gap differences to numpy array
            permutation_gap_diff_1 = np.array(permutation_gap_diff_1)

            # Compute p-value for Betti-1
            if not np.isnan(observed_gap_diff_1) and permutation_gap_diff_1.size > 0:
                p_value_1 = np.sum(permutation_gap_diff_1 >= observed_gap_diff_1) / permutation_gap_diff_1.size
            else:
                p_value_1 = np.nan

            print(f"Computed p_value for Betti 1 difference: {p_value_1}")

            # Store the significance thresholds and p-values
            barcodes[group_name]['observed_gap_1_early'] = observed_gap_1_early
            barcodes[group_name]['observed_gap_1_late'] = observed_gap_1_late
            barcodes[group_name]['permutation_gap_diff_1'] = permutation_gap_diff_1
            barcodes[group_name]['observed_gap_diff_1'] = observed_gap_diff_1
            barcodes[group_name]['p_value_diff_1'] = p_value_1

            # Collect data for combined plotting with significance thresholds
            groups_data[group_name] = {
                'betti_dgm_1_early': original_dgm_1_early,
                'betti_dgm_1_late': original_dgm_1_late,
                'observed_gap_diff_1': observed_gap_diff_1,
                'permutation_gap_diff_1': permutation_gap_diff_1,
                'p_value_diff_1': p_value_1
            }

            # -----------------------
            # Plot Persistence Diagrams (Betti-1)
            # -----------------------
            # **Addition**: Plotting Betti-1 persistence diagrams using tda.plot_barcode

            # Assign color based on region and condition
            # Ensure condition labels are in lowercase
            condition_lower = condition.lower()
            region_lower = REGION.lower()
            color = color_mapping.get(region_lower, {}).get(condition_lower, 'k')  # Default to black if not found

            # Create a figure and axis for Early Phase
            fig_early, ax_early = plt.subplots(figsize=(6, 6))
            tda.plot_barcode(
                betti_num=1,
                betti_dgm=original_dgm_1_early,
                group_name=group_name,
                region=REGION,
                output_dir=barcode_output_dir,
                observed_gap=observed_gap_1_early,
                permutation_gaps=permutation_gap_diff_1,  # Assuming this is relevant; adjust if needed
                p_value=p_value_1,
                ax=ax_early,
                color=color,
                phase='early'
            )
            betti1_early_filename = os.path.join(
                barcode_output_dir,
                f'betti1_early_{REGION}_{condition}_{region_type}.pdf'
            )
            fig_early.savefig(betti1_early_filename, dpi=300)
            plt.close(fig_early)
            print(f"Saved Betti-1 Early Phase persistence diagram to {betti1_early_filename}")

            # Create a figure and axis for Late Phase
            fig_late, ax_late = plt.subplots(figsize=(6, 6))
            tda.plot_barcode(
                betti_num=1,
                betti_dgm=original_dgm_1_late,
                group_name=group_name,
                region=REGION,
                output_dir=barcode_output_dir,
                observed_gap=observed_gap_1_late,
                permutation_gaps=permutation_gap_diff_1,  # Assuming this is relevant; adjust if needed
                p_value=p_value_1,
                ax=ax_late,
                color=color,
                phase='late'
            )
            betti1_late_filename = os.path.join(
                barcode_output_dir,
                f'betti1_late_{REGION}_{condition}_{region_type}.pdf'
            )
            fig_late.savefig(betti1_late_filename, dpi=300)
            plt.close(fig_late)
            print(f"Saved Betti-1 Late Phase persistence diagram to {betti1_late_filename}")
            
        ##############################
        #Gap difference analaysis
        ##############################
        # Create a figure with subplots for gap difference distributions
        num_groups = len(groups_data)
        fig_gap_diff, axs_gap_diff = plt.subplots(num_groups, 1, figsize=(2.5, 5 ))
        fig_gap_diff.suptitle(f'{REGION.upper()} Region Betti-1 Gap Difference Distributions (Late - Early)', fontsize=9)

        # Ensure axs_gap_diff is iterable
        if num_groups == 1:
            axs_gap_diff = [axs_gap_diff]

        # Iterate over the groups and plot in respective subplots
        for idx, (group_name, dgms) in enumerate(groups_data.items()):
            conditionLoc, region_type = group_name

            ax = axs_gap_diff[idx]
            permutation_gaps = dgms['permutation_gap_diff_1']
            observed_diff = dgms['observed_gap_diff_1']
            p_value = dgms['p_value_diff_1']
            betti_label = 'Betti 1'

            if not np.isnan(observed_diff) and permutation_gaps.size > 0:
                # Plot histogram of gap differences
                ax.hist(permutation_gaps, bins=30, color='lightgrey', edgecolor='black')
                # Plot vertical line for observed difference
                ax.axvline(observed_diff, color=color, linestyle='dashed', linewidth=4)
                ax.set_title(f'{betti_label} - Condition={conditionLoc}, Region={region_type}')
                ax.set_xlabel('Gap Difference (Late - Early)')
                ax.set_ylabel('Frequency')
                ax.text(0.95, 0.95, f'p-value: {p_value:.6f}',
                        horizontalalignment='right',
                        verticalalignment='top',
                        transform=ax.transAxes,
                        fontsize=12,
                        bbox=dict(facecolor='white', alpha=0.5))
            else:
                # If observed_diff is NaN or no valid permutations, indicate that no valid data is available
                if np.isnan(observed_diff):
                    message = 'Observed gap difference is NaN'
                elif permutation_gaps.size == 0:
                    message = 'No valid permutations'
                else:
                    message = 'No valid data'
                ax.text(0.5, 0.5, message,
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=ax.transAxes,
                        fontsize=12)
                ax.set_title(f'{betti_label} - Condition={conditionLoc}, Region={region_type}')
                ax.set_xlabel('Gap Difference (Late - Early)')
                ax.set_ylabel('Frequency')

        # Adjust layout and save the gap difference distribution figure
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust rect to make space for suptitle
        combined_gap_diff_filename = f"{gap_output_dir}/combined_gap_differences_{REGION}_permutation.pdf"
        fig_gap_diff.savefig(combined_gap_diff_filename, dpi=300)
        plt.close(fig_gap_diff)

        print(f"Saved combined gap difference distribution plots for region {REGION} to {combined_gap_diff_filename}\n")

        

    # -----------------------
    # After Processing All Groups
    # -----------------------
    ripley.compile_and_plot_ripleys_H_diff(
        collected_data=collected_ripley_data,
        output_dir=ripley_summary_plot_dir
    )

    '''
    # After processing all groups within the region, create combined Ripley's H_diff plots
    # **[ADDED] Plot all H_diff differences on the same plot with average ± SEM per condition**
    if all_h_diffs:
        ripley.plot_all_ripleys_H_diffs(
            all_h_diffs=all_h_diffs,
            output_dir=ripley_summary_plot_dir
        )
        print(f"Saved combined Ripley's H_diff summary plots to {ripley_summary_plot_dir}\n")
    else:
        print(f"No Ripley's H_diff data accumulated for region {REGION}. Skipping summary plots.\n")
    '''

    print("\n" + "="*20 + " Analysis Complete " + "="*20 + "\n")

# Run the main function
if __name__ == "__main__":
    main()

