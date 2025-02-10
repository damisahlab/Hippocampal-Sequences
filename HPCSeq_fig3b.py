from utils import dataFrameOperations as dfo
from utils import lfp_analysis as lfpA
from utils import region_info_loader as ril  
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend suitable for scripts
import matplotlib.pyplot as plt
import seaborn as sns
import pdb
import os
from collections import defaultdict
from sklearn.decomposition import PCA  # For PCA
from sklearn.manifold import MDS  # For MDS (optional)
from sklearn.preprocessing import StandardScaler  # For Z-scoring

PLOT_INDIV_SESS = True
NUM_SHUFFLES = 1000
N_TRIALS_IN_PHASE = 3
REGION_TYPE = 'hpc'  # Set to 'hpc' or any region type
ANNOTATE_POINTS = True  # Set to True to annotate points with session IDs

# Define the settings
settings = [
    {
        'description': 'Correlation Change per Distance_1-5',
        'label_text': 'Correlation Change per Distance_1-5'
    }
]

def clean_string(input_string):
    """
    Clean up spaces and replace slashes '/' with underscores '_' in a string.

    Args:
    - input_string (str): The string to be cleaned.

    Returns:
    - str: The cleaned string with spaces and slashes replaced by underscores.
    """
    if isinstance(input_string, str):
        cleaned_string = input_string.replace(' ', '_').replace('/', '_').replace('(','_').replace(')','_')
        return cleaned_string
    return str(input_string)

# Load the saved DataFrame
def load_data(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

# Load the spectrogram DataFrame
spectrogram_df = load_data('updated_spectrogram_df_with_region_subfield.pkl')

# Unite pt1 sessions (random and structured) to allow 2D point plotting
specialSessName = 'pt1_sess'
spectrogram_df['session'] = spectrogram_df['session'].str.replace(f'{specialSessName}_2', specialSessName)
spectrogram_df['session'] = spectrogram_df['session'].str.replace(f'{specialSessName}_1', specialSessName)

# pt1 localization 
new_value = 'left hippocampus anterior'  # Replace 'reset_value' with your desired value
# Use .loc to modify the DataFrame in place
spectrogram_df.loc[
    (spectrogram_df['session'] == specialSessName) & (spectrogram_df['region_type'] == 'hpc'),
    'regionName'
] = new_value

# Function to compute the correlation matrix per trial by concatenating spectrograms across channels in the same group
def compute_correlation_per_trial_grouped(df_condition, num_trials, num_scenes):
    """
    Computes the pairwise correlation between scenes for each trial by concatenating the spectrograms
    across channels in the same group.

    Args:
        df_condition: DataFrame containing spectrogram data for the specific condition and region type.
        num_trials: Number of trials.
        num_scenes: Number of scenes per trial.

    Returns:
        correlations_per_trial: A list of correlation matrices, one for each trial.
    """
    correlations_per_trial = []

    # For each trial
    for trial_idx in range(num_trials):
        # For each scene, collect the concatenated spectrograms from all channels in the group
        scene_vectors = []
        for scene_idx in range(num_scenes):
            concatenated_spectrogram = np.array([])
            # Iterate over each channel in the group
            for idx, row in df_condition.iterrows():
                # Get the spectrogram data for this row
                spectrogram_data = row['spectrogram_per_trial_per_scenePosWithinTrial']
                # Get the spectrogram for this trial and scene
                spectrogram = spectrogram_data[trial_idx, scene_idx, :, :]
                # Flatten it
                flattened_spectrogram = spectrogram.flatten()
                # Concatenate it
                concatenated_spectrogram = np.concatenate((concatenated_spectrogram, flattened_spectrogram))
            # Append the concatenated spectrogram for this scene
            scene_vectors.append(concatenated_spectrogram)
        # Compute the correlation matrix between scenes for this trial
        corr_matrix = np.corrcoef(scene_vectors)
        correlations_per_trial.append(corr_matrix)

    return correlations_per_trial

# New Function: Perform PCA and Plot Spectrograms in 2D Space with Separate Subplots
def plot_spectrogram_pca(session_id, group_name, condition, early_spectrograms, late_spectrograms, freq_bins, time_bins, output_dir='PCA_Plots'):
    """
    Performs PCA on early and late spectrograms and plots them in 2D space with separate subplots,
    including spectrogram representations of the principal components and singular values.

    Args:
        session_id (str): Identifier for the session.
        group_name (str): 'anterior' or 'posterior'.
        condition (str): 'structured' or 'random'.
        early_spectrograms (dict): {scene_number: spectrogram_array}
        late_spectrograms (dict): {scene_number: spectrogram_array}
        freq_bins (int): Number of frequency bins in the spectrogram.
        time_bins (int): Number of time bins in the spectrogram.
        output_dir (str): Directory to save the plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for PCA
    # Each spectrogram should be a 1D vector
    scenes = sorted(early_spectrograms.keys())  # Assuming scenes are labeled 1-10
    early_vectors = [early_spectrograms[scene] for scene in scenes]
    late_vectors = [late_spectrograms[scene] for scene in scenes]
    
    # Combine early and late vectors
    combined_vectors = early_vectors + late_vectors
    labels = ['Early'] * len(scenes) + ['Late'] * len(scenes)
    scene_numbers = scenes * 2  # Duplicate for early and late
    
    # Check for NaNs in combined_vectors
    if np.isnan(combined_vectors).any():
        print(f"Warning: NaN values found in combined_vectors for session {session_id}, group {group_name}, condition {condition}. Imputing missing values.")
        # Option 1: Impute NaNs with the mean of the feature (column-wise)
        nan_indices = np.where(np.isnan(combined_vectors))
        for i in range(combined_vectors.shape[1]):
            col = combined_vectors[:, i]
            if np.isnan(col).any():
                mean_val = np.nanmean(col)
                combined_vectors[np.isnan(col), i] = mean_val
    else:
        print(f"No NaN values found in combined_vectors for session {session_id}, group {group_name}, condition {condition}.")
    
    # Perform PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(combined_vectors)
    
    # Extract explained variance
    explained_variance = pca.explained_variance_ratio_ * 100  # Convert to percentage
    
    # Split back into early and late
    early_pcs = principal_components[:len(scenes)]
    late_pcs = principal_components[len(scenes):]
   
    # Create DataFrames for plotting
    plot_df_early = pd.DataFrame({
        'PC1': early_pcs[:,0],
        'PC2': early_pcs[:,1],
        'Scene': scenes
    })
    
    plot_df_late = pd.DataFrame({
        'PC1': late_pcs[:,0],
        'PC2': late_pcs[:,1],
        'Scene': scenes
    })
    
    # Normalize scene numbers for colormap
    norm = plt.Normalize(min(scenes), max(scenes))
    cmap = plt.cm.coolwarm
    
    # Determine global axis limits for consistent axes across subplots
    all_pc1 = np.concatenate([early_pcs[:,0], late_pcs[:,0]])
    all_pc2 = np.concatenate([early_pcs[:,1], late_pcs[:,1]])
    pc1_min, pc1_max = all_pc1.min() - 1, all_pc1.max() + 1
    pc2_min, pc2_max = all_pc2.min() - 1, all_pc2.max() + 1
    
    # Create a single figure with two PCA scatter subplots and two PC spectrogram subplots
    fig = plt.figure(figsize=(25, 20))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    
    # Top Row: Early and Late Trials PCA Scatter Plots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Plot Early Trials PCA Scatter Plot
    scatter1 = ax1.scatter(
        plot_df_early['PC1'], plot_df_early['PC2'],
        c=plot_df_early['Scene'], cmap='hsv', norm=norm,
        edgecolor='k', s=500, alpha=0.7
    )
    # Annotations for scenes
    for idx, row in plot_df_early.iterrows():
        ax1.annotate(int(row['Scene']),
                    (row['PC1'] + 0.02, row['PC2'] + 0.02),
                    fontsize=9)
    ax1.set_title(f'Early Trials PCA for Session {session_id}, Group {group_name}, Condition {condition}\nPC1: {explained_variance[0]:.2f}% Variance, PC2: {explained_variance[1]:.2f}% Variance', fontsize=16)
    ax1.set_xlabel('Principal Component 1', fontsize=14)
    ax1.set_ylabel('Principal Component 2', fontsize=14)
    ax1.set_xlim(pc1_min, pc1_max)
    ax1.set_ylim(pc2_min, pc2_max)
    ax1.grid(True)
    
    # Plot Late Trials PCA Scatter Plot
    scatter2 = ax2.scatter(
        plot_df_late['PC1'], plot_df_late['PC2'],
        c=plot_df_late['Scene'], cmap='hsv', norm=norm,
        edgecolor='k', s=500, alpha=0.7
    )
    # Annotations for scenes
    for idx, row in plot_df_late.iterrows():
        ax2.annotate(int(row['Scene']),
                    (row['PC1'] + 0.02, row['PC2'] + 0.02),
                    fontsize=9)
    ax2.set_title(f'Late Trials PCA for Session {session_id}, Group {group_name}, Condition {condition}\nPC1: {explained_variance[0]:.2f}% Variance, PC2: {explained_variance[1]:.2f}% Variance', fontsize=16)
    ax2.set_xlabel('Principal Component 1', fontsize=14)
    ax2.set_ylabel('Principal Component 2', fontsize=14)
    ax2.set_xlim(pc1_min, pc1_max)
    ax2.set_ylim(pc2_min, pc2_max)
    ax2.grid(True)
    
    # Colorbar for both scatter plots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(scatter1, cax=cbar_ax)
    cbar_ax.set_ylabel('Scene Number', fontsize=14)
    
    # Middle Row: Spectrogram Representations of PC1 and PC2
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Retrieve PCA components
    pc1 = pca.components_[0]  # First principal component
    pc2 = pca.components_[1]  # Second principal component
    
    # Reshape PCs to spectrogram dimensions
    pc1_spectrogram = pc1.reshape(freq_bins, time_bins)
    pc2_spectrogram = pc2.reshape(freq_bins, time_bins)
    
    # Plot PC1 Spectrogram
    im1 = ax3.imshow(pc1_spectrogram, aspect='auto', origin='lower', cmap='viridis')
    ax3.set_title(f'Principal Component 1 Spectrogram for {condition.capitalize()} Condition', fontsize=16)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Frequency')
    fig.colorbar(im1, ax=ax3, fraction=0.046, pad=0.04, label='Power')
    
    # Plot PC2 Spectrogram
    im2 = ax4.imshow(pc2_spectrogram, aspect='auto', origin='lower', cmap='viridis')
    ax4.set_title(f'Principal Component 2 Spectrogram for {condition.capitalize()} Condition', fontsize=16)
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Frequency')
    fig.colorbar(im2, ax=ax4, fraction=0.046, pad=0.04, label='Power')
   
    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    # Save the plot
    output_filename = os.path.join(output_dir, f'PCA_Early_Late_{clean_string(session_id)}_{clean_string(group_name)}_{clean_string(condition)}.png')
    plt.savefig(output_filename)
    plt.close()
    print(f"PCA plot with separate subplots and PC spectrograms saved to {output_filename}")

print("Starting analysis...")

# Loop through the settings
for setting in settings:
    description = setting['description']
    label_text = setting['label_text']
    meta_data_str = description  # Updated as per your change

    print(f"Processing setting: {description}")

    # Ensure the 'condition_name' column is in lowercase for consistent comparisons
    spectrogram_df['condition_name'] = spectrogram_df['condition_name'].str.lower()

    # Initialize dictionaries to collect correlation matrices across sessions
    all_region_correlation_matrices = {}  # Key: group_name, Value: dict of condition keys and matrices

    # Initialize a list to collect data points for plotting
    plot_data = []

    # Initialize a dictionary to collect correlation changes per distance
    distance_corr_changes = defaultdict(list)  # Key: distance (1-5), Value: list of corr_change differences

    # NEW SECTION: Initialize dictionaries to collect early and late vectors across all sessions
    aggregated_early_vectors = defaultdict(list)  # Key: scene_number, Value: list of early vectors across sessions
    aggregated_late_vectors = defaultdict(list)   # Key: scene_number, Value: list of late vectors across sessions

    # Get the list of unique session IDs
    session_ids = spectrogram_df['session'].unique()

    # For each session ID
    for sessID in session_ids:
        print(f"Processing session: {sessID}")

        # Get the subset of the DataFrame for this session and region type
        df_session = spectrogram_df[(spectrogram_df['session'] == sessID) & (spectrogram_df['region_type'] == REGION_TYPE)]

        # Create groupings based on 'anterior' and 'posterior' in 'regionName'
        groups = {
            'anterior': df_session[df_session['regionName'].str.contains('anterior', case=False, na=False)]#,
            #'posterior': df_session[df_session['regionName'].str.contains('posterior', case=False, na=False)]
        }

        # Remove empty groups
        groups = {k: v for k, v in groups.items() if not v.empty}

        if not groups:
            print(f"No anterior or posterior data found for session {sessID}. Skipping.")
            continue

        # For each group within the session
        for group_name, df_region in groups.items():
            print(f"Processing group: {group_name} in session: {sessID}")

            # Ensure the 'condition_name' column is in lowercase for consistent comparisons
            df_region['condition_name'] = df_region['condition_name'].str.lower()

            # Initialize the dictionary to store the average correlation matrices for this session-group
            correlation_matrices = {}

            # Get the channels in this group
            channels_in_group = df_region['channel'].unique()
            # Number of channels in this group
            num_channels = len(channels_in_group)

            # List of conditions
            conditions = ['structured', 'random']

            # Dictionary to store per-session-group correlation change differences for both conditions
            per_session_group_diff = {}  # Key: condition, Value: dict with session-group data

            # For storing average spectrograms for early and late trials per condition
            spectrograms_early = {}  # Key: condition, Value: {scene_number: average_spectrogram}
            spectrograms_late = {}   # Key: condition, Value: {scene_number: average_spectrogram}

            # For each condition
            for condition in conditions:
                # Get the subset of the dataframe for this condition within the session and group
                df_condition = df_region[df_region['condition_name'] == condition]

                # Check if df_condition is empty
                if df_condition.empty:
                    print(f"No data for condition {condition} in session {sessID}, group {group_name}")
                    continue

                # Assuming all channels have the same number of trials and scenes
                first_row = df_condition.iloc[0]
                spectrogram_data = first_row['spectrogram_per_trial_per_scenePosWithinTrial']
                num_trials, num_scenes = spectrogram_data.shape[:2]
                freq_bins, time_bins = spectrogram_data.shape[2], spectrogram_data.shape[3]

                # Compute correlations per trial by concatenating spectrograms across channels in the group
                correlations_per_trial = compute_correlation_per_trial_grouped(df_condition, num_trials, num_scenes)

                # Ensure that we have at least N_TRIALS_IN_PHASE trials
                if len(correlations_per_trial) < N_TRIALS_IN_PHASE:
                    print(f"Not enough trials for condition {condition} in session {sessID}, group {group_name}")
                    continue

                # Get the 'Early' matrices (average of first N_TRIALS_IN_PHASE trials)
                early_trials = correlations_per_trial[:N_TRIALS_IN_PHASE]
                early_mean = np.mean(early_trials, axis=0)

                # Get the 'Late' matrices (average of last N_TRIALS_IN_PHASE trials)
                late_trials = correlations_per_trial[-N_TRIALS_IN_PHASE:]
                late_mean = np.mean(late_trials, axis=0)

                # Compute the change matrix
                change_matrix = late_mean - early_mean

                # Store in the dictionary
                key_early = f"{condition}_early"
                key_late = f"{condition}_late"
                key_change = f"{condition}_change"
                correlation_matrices[key_early] = early_mean
                correlation_matrices[key_late] = late_mean
                correlation_matrices[key_change] = change_matrix

                # Collect matrices across sessions
                region_key = group_name
                if region_key not in all_region_correlation_matrices:
                    all_region_correlation_matrices[region_key] = {}

                # Initialize keys if they don't exist
                for key, matrix in zip([key_early, key_late, key_change], [early_mean, late_mean, change_matrix]):
                    if key not in all_region_correlation_matrices[region_key]:
                        all_region_correlation_matrices[region_key][key] = []
                    all_region_correlation_matrices[region_key][key].append(matrix)

                # Compute per-scene correlation change differences for all distances
                for scene_i in range(num_scenes):
                    for distance in range(1, 6):  # Distances 1 to 5
                        scene_j = (scene_i + distance) % num_scenes  # Circular distance
                        corr_change = change_matrix[scene_i, scene_j]
                        distance_corr_changes[distance].append(corr_change)

                # Compute average spectrograms for early and late trials
                # Initialize dictionaries
                if condition not in spectrograms_early:
                    spectrograms_early[condition] = {}
                if condition not in spectrograms_late:
                    spectrograms_late[condition] = {}

                # Compute average spectrogram per scene for early trials
                for scene_idx in range(num_scenes):
                    scene_number = scene_idx + 1  # Assuming scenes are 1-indexed
                    # Collect spectrograms across channels and early trials
                    spectrograms = []
                    for row_idx, row in df_condition.iterrows():
                        spectrogram_data = row['spectrogram_per_trial_per_scenePosWithinTrial']
                        # Early trials
                        for trial in range(N_TRIALS_IN_PHASE):
                            spectrogram = spectrogram_data[trial, scene_idx, :, :]
                            spectrograms.append(spectrogram.flatten())
                    # Average spectrogram
                    if spectrograms:
                        avg_spectrogram = np.mean(spectrograms, axis=0)
                        spectrograms_early[condition][scene_number] = avg_spectrogram
                    else:
                        spectrograms_early[condition][scene_number] = np.zeros(freq_bins * time_bins)

                # Compute average spectrogram per scene for late trials
                for scene_idx in range(num_scenes):
                    scene_number = scene_idx + 1  # Assuming scenes are 1-indexed
                    # Collect spectrograms across channels and late trials
                    spectrograms = []
                    for row_idx, row in df_condition.iterrows():
                        spectrogram_data = row['spectrogram_per_trial_per_scenePosWithinTrial']
                        # Late trials
                        for trial in range(-N_TRIALS_IN_PHASE, 0):
                            spectrogram = spectrogram_data[trial, scene_idx, :, :]
                            spectrograms.append(spectrogram.flatten())
                    # Average spectrogram
                    if spectrograms:
                        avg_spectrogram = np.mean(spectrograms, axis=0)
                        spectrograms_late[condition][scene_number] = avg_spectrogram
                    else:
                        spectrograms_late[condition][scene_number] = np.zeros(freq_bins * time_bins)

        if 'pt7' in sessID:
            sys.exit()

        # Proceed to call the plotting function for the session-group if needed
        if PLOT_INDIV_SESS and correlation_matrices:
            region_sessID = f"{sessID}_{group_name}"
            min_freq = 0.5  # Example frequency range
            max_freq = 70.0
            min_time = None  # If not applicable
            max_time = None
            region_type = group_name  # Use the group_name ('anterior' or 'posterior')
            num_sessions = 1  # Since we're processing one session at a time

            # Call the plotting function for this session-group
            last_distances, session_averages,_ = lfpA.plot_correlation_matrices(
                correlation_matrices=correlation_matrices,
                sessID=region_sessID,
                min_freq=min_freq,
                max_freq=max_freq,
                min_time=min_time,
                max_time=max_time,
                region_type=region_type,
                num_channels=num_channels,
                num_sessions=num_sessions,
                num_shuffles=NUM_SHUFFLES,
                meta_data_str=meta_data_str  # Updated as per your change
            )
        else:
            print(f"No correlation matrices to plot for session {sessID}, group {group_name}. Skipping.")

        # Perform PCA and plot spectrograms in 2D space for each condition
        for condition in conditions:
            if condition in spectrograms_early and condition in spectrograms_late:
                plot_spectrogram_pca(
                    session_id=sessID,
                    group_name=group_name,
                    condition=condition,
                    early_spectrograms=spectrograms_early[condition],
                    late_spectrograms=spectrograms_late[condition],
                    freq_bins=freq_bins,
                    time_bins=time_bins,
                    output_dir='PCA_Plots'
                )
            else:
                print(f"Insufficient spectrogram data for PCA plotting in session {sessID}, group {group_name}, condition {condition}")

    # NEW SECTION: Aggregate Early and Late Vectors Across All Sessions to Create Supersession PCA
    print("Aggregating early and late vectors across all sessions for supersession PCA...")

    # Loop through all sessions and groups to collect early and late vectors
    for sessID in session_ids:
        # Get the subset of the DataFrame for this session and region type
        df_session = spectrogram_df[(spectrogram_df['session'] == sessID) & (spectrogram_df['region_type'] == REGION_TYPE)]

        # Create groupings based on 'anterior' and 'posterior' in 'regionName'
        groups = {
            'anterior': df_session[df_session['regionName'].str.contains('anterior', case=False, na=False)],
            'posterior': df_session[df_session['regionName'].str.contains('posterior', case=False, na=False)]
        }

        # Remove empty groups
        groups = {k: v for k, v in groups.items() if not v.empty}

        for group_name, df_region in groups.items():
            # Ensure the 'condition_name' column is in lowercase for consistent comparisons
            df_region['condition_name'] = df_region['condition_name'].str.lower()

            # Initialize dictionaries to store average spectrograms per condition
            spectrograms_early = {}
            spectrograms_late = {}

            # List of conditions
            conditions = ['structured', 'random']

            for condition in conditions:
                df_condition = df_region[df_region['condition_name'] == condition]

                if df_condition.empty:
                    continue

                # Assuming all channels have the same number of trials and scenes
                first_row = df_condition.iloc[0]
                spectrogram_data = first_row['spectrogram_per_trial_per_scenePosWithinTrial']
                num_trials, num_scenes = spectrogram_data.shape[:2]

                # Collect early and late vectors for this condition and group
                early_vectors = []
                late_vectors = []

                for row_idx, row in df_condition.iterrows():
                    spectrogram_data = row['spectrogram_per_trial_per_scenePosWithinTrial']
                    # Early trials
                    for trial in range(N_TRIALS_IN_PHASE):
                        for scene_idx in range(num_scenes):
                            spectrogram = spectrogram_data[trial, scene_idx, :, :]
                            early_vectors.append(spectrogram.flatten())
                    # Late trials
                    for trial in range(-N_TRIALS_IN_PHASE, 0):
                        for scene_idx in range(num_scenes):
                            spectrogram = spectrogram_data[trial, scene_idx, :, :]
                            late_vectors.append(spectrogram.flatten())

                # Convert lists to numpy arrays
                early_vectors = np.array(early_vectors)
                late_vectors = np.array(late_vectors)

                # Check for NaNs in early_vectors and late_vectors
                if np.isnan(early_vectors).any():
                    print(f"Warning: NaN values found in early_vectors for session {sessID}, group {group_name}, condition {condition}. Imputing missing values.")
                    # Impute NaNs with the mean of each feature
                    col_means = np.nanmean(early_vectors, axis=0)
                    inds = np.where(np.isnan(early_vectors))
                    early_vectors[inds] = np.take(col_means, inds[1])

                if np.isnan(late_vectors).any():
                    print(f"Warning: NaN values found in late_vectors for session {sessID}, group {group_name}, condition {condition}. Imputing missing values.")
                    # Impute NaNs with the mean of each feature
                    col_means = np.nanmean(late_vectors, axis=0)
                    inds = np.where(np.isnan(late_vectors))
                    late_vectors[inds] = np.take(col_means, inds[1])

                # Average early and late vectors per scene
                for scene_number in range(1, num_scenes + 1):
                    # Indices for the current scene
                    scene_idx = scene_number - 1
                    # Early vectors for the current scene
                    scene_early_vectors = early_vectors[:, scene_idx * spectrogram_data.shape[3] * spectrogram_data.shape[2] : (scene_idx + 1) * spectrogram_data.shape[3] * spectrogram_data.shape[2]]
                    # Late vectors for the current scene
                    scene_late_vectors = late_vectors[:, scene_idx * spectrogram_data.shape[3] * spectrogram_data.shape[2] : (scene_idx + 1) * spectrogram_data.shape[3] * spectrogram_data.shape[2]]

                    # Check if scene_early_vectors and scene_late_vectors have the expected shape
                    if scene_early_vectors.size == 0 or scene_late_vectors.size == 0:
                        print(f"Scene {scene_number} in session {sessID}, group {group_name}, condition {condition} has insufficient data. Skipping.")
                        continue

                    # Average across trials and channels
                    avg_early = np.mean(scene_early_vectors, axis=0)
                    avg_late = np.mean(scene_late_vectors, axis=0)

                    # Check for NaNs in avg_early and avg_late
                    if np.isnan(avg_early).any() or np.isnan(avg_late).any():
                        print(f"Scene {scene_number} in session {sessID}, group {group_name}, condition {condition} has NaNs after averaging. Skipping.")
                        continue

                    # Append to aggregated dictionaries
                    aggregated_early_vectors[scene_number].append(avg_early)
                    aggregated_late_vectors[scene_number].append(avg_late)

    # Now, for each scene, average across all sessions
    supersession_vectors = {}  # Key: scene_number, Value: average vector

    for scene_number in sorted(aggregated_early_vectors.keys()):
        # Check if there are any vectors for early and late
        if aggregated_early_vectors[scene_number] and aggregated_late_vectors[scene_number]:
            # Average early vectors across sessions
            early_avg = np.mean(aggregated_early_vectors[scene_number], axis=0)
            # Average late vectors across sessions
            late_avg = np.mean(aggregated_late_vectors[scene_number], axis=0)
            # Combine early and late by averaging them
            combined_avg = np.mean([early_avg, late_avg], axis=0)
            supersession_vectors[scene_number] = combined_avg
        else:
            print(f"Scene {scene_number} has insufficient data for supersession PCA. Skipping.")
            # Optionally, assign NaNs or a default vector
            # supersession_vectors[scene_number] = np.full_like(next(iter(aggregated_early_vectors.values()))[0], np.nan)
            # Here, we'll skip adding this scene to supersession_vectors
            continue

    # Convert supersession vectors to a DataFrame
    supersession_df = pd.DataFrame.from_dict(supersession_vectors, orient='index', columns=[f'feature_{i}' for i in range(1, len(next(iter(supersession_vectors.values()))) + 1)])
    supersession_df.index.name = 'Scene'
    supersession_df.reset_index(inplace=True)

    # Check for any remaining NaNs
    if supersession_df.isnull().values.any():
        nan_scenes = supersession_df[supersession_df.isnull().any(axis=1)]['Scene'].tolist()
        print(f"Scenes with NaNs after aggregation and averaging: {nan_scenes}")
        # Decide how to handle them: drop or impute
        # For this example, we'll drop them
        supersession_df = supersession_df.dropna()
        if supersession_df.empty:
            raise ValueError("All supersession vectors contain NaNs. Cannot perform PCA.")
        else:
            print(f"Dropped scenes with NaNs. Remaining scenes: {supersession_df['Scene'].tolist()}")

    # Z-score the data
    scaler = StandardScaler()
    supersession_features = scaler.fit_transform(supersession_df.drop('Scene', axis=1))

    if np.max(supersession_features)==0:
        continue
    # Perform PCA
    pca_supersession = PCA(n_components=2)
    principal_components_supersession = pca_supersession.fit_transform(supersession_features)

    # Create a DataFrame for PCA results
    pca_supersession_df = pd.DataFrame({
        'Scene': supersession_df['Scene'],
        'PC1': principal_components_supersession[:, 0],
        'PC2': principal_components_supersession[:, 1]
    })

    # Extract explained variance
    explained_variance_supersession = pca_supersession.explained_variance_ratio_ * 100  # Convert to percentage

    # Plot PCA for Supersession
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        pca_supersession_df['PC1'], pca_supersession_df['PC2'],
        c=pca_supersession_df['Scene'], cmap='coolwarm', edgecolor='k', s=200, alpha=0.7
    )

    # Annotations for scenes
    for idx, row in pca_supersession_df.iterrows():
        plt.annotate(int(row['Scene']),
                    (row['PC1'] + 0.02, row['PC2'] + 0.02),
                    fontsize=12)

    # Title with explained variance
    plt.title(f'Supersession PCA: Average Early & Late Trials\nPC1: {explained_variance_supersession[0]:.2f}% Variance, PC2: {explained_variance_supersession[1]:.2f}% Variance', fontsize=16)
    plt.xlabel('Principal Component 1', fontsize=14)
    plt.ylabel('Principal Component 2', fontsize=14)

    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Scene Number', fontsize=14)

    plt.grid(True)
    plt.tight_layout()

    # Save the Supersession PCA plot
    supersession_output_dir = 'Supersession_PCA_Plots'
    os.makedirs(supersession_output_dir, exist_ok=True)
    supersession_output_filename = os.path.join(supersession_output_dir, 'Supersession_PCA.png')
    plt.savefig(supersession_output_filename)
    plt.close()
    print(f"Supersession PCA plot saved to {supersession_output_filename}")

    # After processing all session-groups, compute the average correlation matrices across all sessions for each group
    print("Computing average correlation matrices across all sessions...")

    for group_name, matrices_dict in all_region_correlation_matrices.items():
        # Initialize the dictionary to store the averaged matrices
        avg_correlation_matrices = {}

        for key in ['structured_early', 'structured_late', 'structured_change',
                    'random_early', 'random_late', 'random_change']:
            if key in matrices_dict and matrices_dict[key]:
                # Stack the matrices and compute the average
                stacked_matrices = np.stack(matrices_dict[key], axis=0)
                avg_matrix = np.mean(stacked_matrices, axis=0)
                avg_correlation_matrices[key] = avg_matrix

        # Now, we have the averaged correlation matrices for this group across all sessions
        # Proceed to call the plotting function
        region_sessID = f"AllSessions_{group_name}"

        # Use the number of channels across all sessions
        num_channels = len(spectrogram_df[spectrogram_df['regionName'].str.contains(group_name, case=False, na=False)]['channel'].unique())
        num_sessions = len(session_ids)

        # Call the plotting function
        if avg_correlation_matrices:
            last_distances, session_averages = lfpA.plot_correlation_matrices(
                correlation_matrices=avg_correlation_matrices,
                sessID=region_sessID,
                min_freq=0.5,
                max_freq=70.0,
                min_time=None,  # Defined as None above
                max_time=None,
                region_type=group_name,
                num_channels=num_channels,
                num_sessions=num_sessions,
                num_shuffles=NUM_SHUFFLES,
                meta_data_str=description
            )
        else:
            print(f"No average correlation matrices to plot for group {group_name}. Skipping.")

    # Now, proceed to plot the correlation change per distance
    # Compute average and SEM for each distance
    distances = sorted(distance_corr_changes.keys())  # [1,2,3,4,5]
    mean_corr_changes = []
    sem_corr_changes = []

    for distance in distances:
        corr_changes = distance_corr_changes[distance]
        mean = np.mean(corr_changes)
        sem = np.std(corr_changes, ddof=1) / np.sqrt(len(corr_changes))
        mean_corr_changes.append(mean)
        sem_corr_changes.append(sem)
        print(f"Distance {distance}: Mean Change = {mean:.4f}, SEM = {sem:.4f}")

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plotting the average correlation change with SEM as error bars
    sns.lineplot(x=distances, y=mean_corr_changes, marker='o', label='Mean Correlation Change')
    plt.errorbar(distances, mean_corr_changes, yerr=sem_corr_changes, fmt='none', ecolor='red', capsize=5)

    # Customize the plot
    plt.title(f'{label_text} per Distance (1-5)')
    plt.xlabel('Scene Distance')
    plt.ylabel('Correlation Change')
    plt.xticks(distances)
    plt.grid(True)

    # Save the plot to a file
    output_filename = f'CorrelationChangePerDistance_{clean_string(description)}.png'
    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")

    # Close the figure to free up memory
    plt.close()


# Debugging breakpoint
# pdb.set_trace()  # Commented out to allow script to run uninterrupted

