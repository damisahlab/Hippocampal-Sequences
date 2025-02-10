from utils import single_unit_checkpoint as suc
from scipy.signal import welch
from scipy.signal import butter, filtfilt
from scipy.stats import pearsonr
from utils import region_info_loader as ril
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import patchworklib as pw
from scipy.signal import welch, spectrogram
import sys
from utils import lfp_loader as ll
from utils import lfp_analysis as lfpA
from utils import dataPathDict as dpd
import pdb
from utils import taskInfoFunctions as tif
import os
import pickle
from utils import aggregation_bootstrap as ab
from scipy.signal import spectrogram
from scipy.signal import convolve2d
from utils import local_paths

from matplotlib.cm import get_cmap
import scipy.stats
np.random.seed(0)  # Set a seed for reproducibility
brainSideStr='left side'
brainSideStr='right side'
brainSideStr='both sides'
SMOOTH_MATRIX_FIRST=True
#SMOOTH_MATRIX_FIRST=False

zeroTol=1e-15

def save_data(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

def load_data(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def plot_average_changes(distances, average_values, sem_values, num_sessions):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 7), sharey=True)

    categories = ['random','structured']
    colors = ['black','red']

    for ax, category, color in zip(axes, categories, colors):
        avg = average_values[category]
        sem = sem_values[category]
        ax.errorbar(distances, avg, yerr=sem, fmt='-o', capsize=5, color=color, label=f'{category.capitalize()} Change')
        ax.axhline(0, color='black', linewidth=3, linestyle='--')
        ax.set_xlim([0, 5.1])
        #ax.set_xlabel('Scene Distance')
        ax.set_xlabel('Within-trial scene separation (no. scenes)')
        ax.set_title(f'{category.capitalize()} scene order')

    # Common ylabel and title
    axes[0].set_ylabel('Average Correlation Change (scene shuffle Z-score per session)')
    fig.suptitle(f'Change in LFP scene code similarity by distance (n={num_sessions} sessions)')

    # Only add the legend to the first subplot
    #axes[0].legend()

    # Save the figure
    #plt.savefig('allPts_allChannelSpectralCorrelationMatrixChanges.png')
    plt.savefig(f'allPts_{region_type}_allChannelSpectralCorrelationMatrixChanges.png')


def calculate_avg_and_sem(values_list):
    # Convert list of arrays into a 2D numpy array for easier computation
    values_array = np.array(values_list)
    avg = np.nanmean(values_array, axis=0)
    sem = np.nanstd(values_array, axis=0) / np.sqrt(values_array.shape[0])
    return avg, sem


# Function to calculate average matrices
def average_matrices(matrices_dict):
    average_matrix = {}
    for key in matrices_dict[next(iter(matrices_dict))]:
        sum_matrix = sum(matrices_dict[session_id][key] for session_id in matrices_dict)
        average_matrix[key] = sum_matrix / len(matrices_dict)
    return average_matrix


# Main code
#March 22, 2024
sessNames=dpd.getSessNames()
#sessNames=dpd.getSessNamesNoNRS()
#sessNames=dpd.getSessNamesWithMircoLFP()
#sessNames=dpd.getLatestSessNames()
#sessNames=dpd.getWithUnitSessNamesSubset()
#max_channel_number=16
max_channel_number=24

processed_data_dir=local_paths.get_processed_data_dir()
smoothAdjTrials=True 
smoothAdjTrials=False
#min_time=-0.1
min_time=0
#max_time=0.5
#max_time=0.3
max_time=0.3
max_time=0.4
max_time=1
max_time=0.75
max_time=0.6
#TEST - SHOULD BE 0.6!!!!!
#max_time=0.8

#min_freq=1
#min_freq=5
#min_freq=15
#min_freq=0.5
min_freq=0.5
#min_freq=5
max_freq=60
max_freq=50
#max_freq=20
max_freq=70
#max_freq=40
#max_freq=80

all_sessions_correlation_matrices = {}
all_sessions_change_matrices = {}
accumulated_averages = {'structured': [], 'random': []}
accumulated_slopes= {'structured': [], 'random': []}
accumulated_slopes_sessNames= {'structured': [], 'random': []}
accumulated_averages_4_matrices = {f'{cond}_{timing}': [] for cond in ['structured', 'random'] for timing in ['early', 'late']}

num_trials_in_timing=30 #June 19 2024


#region_type='hpc'
#region_type='non-hpc'

region_type = sys.argv[1]

# Define the file paths
data_file_path = f'lfp_scene_code_change_stats_all_sess_{region_type}_Aug2024.pkl'

# Check if data already exists
if  False and os.path.exists(data_file_path):
    #if True and os.path.exists(data_file_path):
    # Load data
    all_slopes, correlation_matrices, accumulated_averages,accumulated_averages_4_matrices = load_data(data_file_path)
else:

    all_slopes = {'structured': [], 'random': []}

    #sessID = sys.argv[1]
    #channel_range = map(int, sys.argv[2].split(','))  # Expecting a comma-separated list of channels
    for sessID in sessNames:
        
        #if 'pt8' in sessID:
        #    continue
        # Initializing the scene_spectrograms dictionary with an additional level for channels
        scene_spectrograms = {
            scene: {
                channel: {"structured_early": [], "structured_late": [], "random_early": [], "random_late": []}
                for channel in range(max_channel_number)  # Replace with the max number of channels
            } 
            for scene in range(1, 11)
        }

        if 'non' in region_type:
            channel_range = ril.get_non_hpc_lfp_channels(sessID, brainSideStr)
        elif 'hpc' in region_type:
            channel_range = ril.get_hpc_lfp_channels(sessID, brainSideStr)
        elif 'myg' in region_type:
            channel_range = ril.get_amygdala_channels(sessID, brainSideStr)
        elif 'nsul' in region_type:
            channel_range = ril.get_insula_channels(sessID, brainSideStr)
        elif 'ingul' in region_type:
            channel_range = ril.get_cingulate_channels(sessID, brainSideStr)
        elif 'aic' in region_type:
            channel_range = ril.get_combined_channels(sessID, 'amygdala','insula','cingulate')

        if len(channel_range) == 0:
            continue

        num_conditions=len(tif.get_condition_names_for_sessID(sessID))
        filename = f"{sessID}_{region_type}_flattened_trial_scene_spectrograms.pkl"
        if os.path.exists(filename):
            with open(filename,'rb') as f:
                flattened_trial_scene_spectrograms=pickle.load(f)
        else:
            # Initialize the data structure for storing spectrograms
            # This dictionary will have trial groups as keys, which then contain trials, which then contain scenes
            trial_scene_spectrograms = {
                trial_group: {trial: {scene: [] for scene in range(1, 11)} for trial in range(15)} 
                for trial_group in range(num_conditions)
            }

            for channel_idx, channel in enumerate(channel_range):
                downsampled_lfp, all_event_times, originalFs, dsFact = ll.load_downsampled_lfp_and_events_with_cache(sessID, int(channel),dataDir=processed_data_dir)
                
                if downsampled_lfp is None:
                    continue

                for scene_num in range(1, 11):  # For each scene
                    classified_scene_event_times = tif.classify_events_for_scene_with_trial_num(sessID, scene_num, all_event_times, early_threshold=num_trials_in_timing, late_threshold=num_trials_in_timing)

                    for idx, (event_time, condition, timing, trial_within_group) in enumerate(classified_scene_event_times):
                        # Compute and filter spectrogram
                        #f, t, f_edges, t_edges, Sxx = lfpA.compute_event_triggered_spectrogram(downsampled_lfp, event_time, originalFs / dsFact, min_freq=min_freq, max_freq=max_freq,nperseg=512)
                        f, t, f_edges, t_edges, Sxx = lfpA.compute_event_triggered_spectrogram(downsampled_lfp, event_time, originalFs / dsFact, min_freq=min_freq, max_freq=max_freq,nperseg=256)
                        #f, t, f_edges, t_edges, Sxx = lfpA.compute_event_triggered_spectrogram(downsampled_lfp, event_time, originalFs / dsFact, min_freq=min_freq, max_freq=max_freq,nperseg=300)
                        Sxx_filtered = Sxx[:, (t >= min_time) & (t <= max_time)]

                        if Sxx_filtered is not None and Sxx_filtered.size > 0:
                            # Determine the trial within the trial group based on the index
                            # Each set of 15 presentations (classified events for a scene) is considered a different trial within the same trial group
                            trial_group_idx_local = idx // 15

                            if 'pt1' in sessID: #only patient with repeating consecutive conditions
                                trial_group_idx_local= 0
                            # Store raw spectrogram for the scene in the specific trial within the trial group
                            trial_scene_spectrograms[trial_group_idx_local][trial_within_group-1][scene_num].append(Sxx_filtered)


            # Initialize a structure for flattened spectrograms
            flattened_trial_scene_spectrograms = {
                trial_group: {
                    trial: {scene: None for scene in range(1, 11)} for trial in trial_scene_spectrograms[trial_group]
                } for trial_group in trial_scene_spectrograms
            }
            for trial_group in trial_scene_spectrograms:
                for trial in trial_scene_spectrograms[trial_group]:
                    for scene in trial_scene_spectrograms[trial_group][trial]:
                        concatenated_vectors = []
                        for spectrogram in trial_scene_spectrograms[trial_group][trial][scene]:
                            concatenated_vectors.append(spectrogram.flatten())

                        if concatenated_vectors:
                            # Concatenate across channels
                            flattened_trial_scene_spectrograms[trial_group][trial][scene] = np.concatenate(concatenated_vectors)

            # Save the flattened_trial_scene_spectrograms to a file
            with open(filename, 'wb') as file:
                pickle.dump(flattened_trial_scene_spectrograms, file)

            print(f"Saved flattened_trial_scene_spectrograms for session {sessID} to {filename}")

        # Initialize correlation matrices for each trial within each trial group
        scene_code_corrMatrx_per_trial_gp_trial = {
            trial_group: {
                trial: np.zeros((10, 10)) for trial in flattened_trial_scene_spectrograms[trial_group]
            } for trial_group in flattened_trial_scene_spectrograms
        }

        for trial_group in flattened_trial_scene_spectrograms:
            for trial in flattened_trial_scene_spectrograms[trial_group]:
                for scene1 in range(1, 11):
                    for scene2 in range(1, 11):
                        #smooth consecutive trials?
                        if smoothAdjTrials:
                            # Gather vectors for the current, preceding, and following trial for both scenes
                            vectors1 = []
                            vectors2 = []
                            for adj_trial in [trial - 1, trial, trial + 1]:
                                if adj_trial in flattened_trial_scene_spectrograms[trial_group]:
                                    if scene1 in flattened_trial_scene_spectrograms[trial_group][adj_trial]:
                                        vectors1.append(flattened_trial_scene_spectrograms[trial_group][adj_trial][scene1])
                                    if scene2 in flattened_trial_scene_spectrograms[trial_group][adj_trial]:
                                        vectors2.append(flattened_trial_scene_spectrograms[trial_group][adj_trial][scene2])

                            # Only proceed if both scenes have vectors in the current or adjacent trials
                            if vectors1 and vectors2:
                                # Average vectors across adjacent trials
                                vector1 = np.mean(vectors1, axis=0)
                                vector2 = np.mean(vectors2, axis=0)
                                correlation, _ = pearsonr(vector1, vector2)
                                scene_code_corrMatrx_per_trial_gp_trial[trial_group][trial][scene1 - 1, scene2 - 1] = correlation
                        else:    
                            vector1 = flattened_trial_scene_spectrograms[trial_group][trial].get(scene1)
                            vector2 = flattened_trial_scene_spectrograms[trial_group][trial].get(scene2)
                            if vector1 is not None and vector2 is not None and (np.isfinite(vector1).all() and np.isfinite(vector2).all()):
                                # Calculate Pearson correlation
                                correlation, _ = pearsonr(vector1, vector2)
                                scene_code_corrMatrx_per_trial_gp_trial[trial_group][trial][scene1 - 1, scene2 - 1] = correlation


        scene_order_matrix= tif.get_scene_num_per_disp_from_mat(sessID)
       

        N = int(num_trials_in_timing/10) # Assuming N is defined as the cutoff for early vs. late trials
        conditions = ['random', 'structured']  # Assuming this maps trial groups to conditions
        

        condition_keys = [f'{cond}_{timing}' for cond in conditions for timing in ['early', 'late']]
        
        # Trial group conditions mapping
        trial_group_conditions = tif.get_condition_names_for_sessID(sessID)

        ########################################################################
        
        # Initialize matrices for sums and counts
        sum_matrices = {key: np.zeros((10, 10)) for key in condition_keys}  # Assuming 10 scenes per trial
        count_matrices = {key: np.zeros((10, 10), dtype=int) for key in condition_keys}

        #scene_order_matrix= tif.get_scene_num_per_disp_from_mat(sessID)
        ##########################################################################
        #ASSUMING CORRELATION MATRICES PER TRIAL ARE IN TERMS OF SCENE IDENTITY, NOT ORDER WITHIN TRIAL
        ##########################################################################
        for trial_group_idx, trial_group in enumerate(scene_code_corrMatrx_per_trial_gp_trial):
            condition = trial_group_conditions[trial_group_idx]  # Map trial group to condition
            trials = list(scene_code_corrMatrx_per_trial_gp_trial[trial_group].keys())
            early_trials = trials[:N]
            #TRYING IGNORE FIRST TRIAL -TJ Sept 23 2024
            #early_trials = trials[1:(N)]
            late_trials = trials[-N:]

            for timing, selected_trials in [('early', early_trials), ('late', late_trials)]:
                key = f"{condition}_{timing}"
                for trial in selected_trials:
                    sceneIdentity_per_pos_currTrial =scene_order_matrix[trial_group][:, trial]  # Scene position order
                    corr_matrix_by_sceneIdentity = scene_code_corrMatrx_per_trial_gp_trial[trial_group][trial]
                    
                    for i in range(len(sceneIdentity_per_pos_currTrial)):
                        for j in range(len(sceneIdentity_per_pos_currTrial)):
                            # Use scene positions to index into the correlation matrix
                            correlation = corr_matrix_by_sceneIdentity[sceneIdentity_per_pos_currTrial[i] - 1, sceneIdentity_per_pos_currTrial[j] - 1]
                            sum_matrices[key][i, j] += correlation
                            count_matrices[key][i, j] += 1

        # Normalize sums by counts to get average correlation matrices
        correlation_matrices = {key: np.divide(sum_matrices[key], count_matrices[key], 
                                                   where=count_matrices[key] != 0) 
                                    for key in condition_keys}

        # Now avg_correlation_matrices contains the average correlation matrices based on scene position
        #distances,session_averages=lfpA.plot_correlation_matrices(correlation_matrices,sessID,min_freq=min_freq,max_freq=max_freq,region_type=region_type)
        distances,session_averages,session_slopes=lfpA.plot_correlation_matrices(correlation_matrices,sessID,SMOOTH_MATRIX_FIRST=SMOOTH_MATRIX_FIRST,min_freq=min_freq,max_freq=max_freq,region_type=region_type)

        # Accumulate averages for each category
        for category in ['structured', 'random']:
            if not any(np.isnan(session_averages[category]['change'])):
                accumulated_averages[category].append(session_averages[category])
            if not np.isnan(session_slopes[category]['change']):
                accumulated_slopes[category].append(session_slopes[category])
                accumulated_slopes_sessNames[category].append(sessID)
             # Mask to filter out NaN values and non-positive distances
            
        # Accumulate averages for each category
        for category in accumulated_averages_4_matrices.keys():
            if np.abs(np.sum(correlation_matrices[category]))>zeroTol:
                accumulated_averages_4_matrices[category].append(correlation_matrices[category])

    # Save data
    data_to_save = (all_slopes, correlation_matrices, accumulated_averages,accumulated_averages_4_matrices)
    save_data(data_to_save, data_file_path)
    #end session loop

'''
for condName in ['structured','random']:
    p_value, observed_stat, null_distribution = ab.bootstrap_test(all_slopes[condName], ab.mean_difference)
    ab.plot_bootstrap_results(observed_stat,null_distribution,condName)
'''
#average_correlation_matrices = average_matrices(all_sessions_correlation_matrices)
#average_change_matrices = average_matrices(all_sessions_change_matrices)

# Assuming accumulated_averages is structured like: 
# {'random_early': [matrix1, matrix2, ...], 'random_late': [...], ...}
# where matrix1, matrix2, ... are the average matrices for each session
overall_averages = {}
overall_slope_changes = {'structured':[],'random':[]}
overall_slope_changes_sessNames= {'structured':[],'random':[]}
overall_sems = {}

for category, values in accumulated_averages_4_matrices.items():
    # Calculate overall average and SEM across sessions for each category
    overall_averages[category] = np.mean(values, axis=0)
    #overall_averages[category] = np.mean(values[:-1], axis=0) #last is all 0s

for category in ['structured', 'random']:
    for si in range(len(accumulated_slopes[category])):
        #overall_slope_changes[category].append(accumulated_slopes[category][si]['late'] - accumulated_slopes[category][si]['early'])
        overall_slope_changes[category].append(accumulated_slopes[category][si]['change'])
        overall_slope_changes_sessNames[category].append(accumulated_slopes_sessNames[category][si])

lfpA.save_overall_slope_changes(overall_slope_changes,overall_slope_changes_sessNames,'All_Patients', region_type,SMOOTH_MATRIX_FIRST=SMOOTH_MATRIX_FIRST)

#lfpA.plot_correlation_matrices(average_correlation_matrices,'allPts')
# Call the adjusted plot function with global averages
lfpA.plot_correlation_matrices(overall_averages, 'All_Patients', SMOOTH_MATRIX_FIRST=SMOOTH_MATRIX_FIRST,min_freq=min_freq, max_freq=max_freq,region_type=region_type)

'''
average_values = {}
sem_values = {}

for category in ['structured', 'random']:
    average_values[category], sem_values[category] = calculate_avg_and_sem(accumulated_averages[category])

num_sessions=len(sessNames)
# Example usage
plot_average_changes(distances, average_values, sem_values,num_sessions)
'''
