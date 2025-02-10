import os
import pandas as pd
import numpy as np
import pickle
from scipy.stats import wilcoxon
from utils import local_paths

processed_data_dir=local_paths.get_processed_data_dir()

def load_summary_dataframe(csv_path=None, pickle_path=None):
    """
    Load the summary DataFrame from a CSV or pickle file.

    Args:
        csv_path (str, optional): Path to the CSV file.
        pickle_path (str, optional): Path to the pickle file.

    Returns:
        pd.DataFrame: The loaded summary DataFrame.

    Raises:
        FileNotFoundError: If neither CSV nor pickle file is found.
    """
    if pickle_path and os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            summary_df = pickle.load(f)
        print(f"Loaded summary DataFrame from pickle: {pickle_path}")
    elif csv_path and os.path.exists(csv_path):
        summary_df = pd.read_csv(csv_path)
        # Convert 'coherence_change' from string to list if loaded from CSV
        summary_df['coherence_change'] = summary_df['coherence_change'].apply(eval).apply(np.array)
        print(f"Loaded summary DataFrame from CSV: {csv_path}")
    else:
        raise FileNotFoundError("Neither CSV nor pickle file was found. Please provide a valid path.")
    return summary_df

def compute_coherence_average(coherence_array, freqs, frequency_band=(3,6)):
    """
    Compute the average of coherence_change values within a specified frequency band.

    Args:
        coherence_array (np.ndarray): Array of coherence changes.
        freqs (np.ndarray): Array of frequency values corresponding to coherence_change.
        frequency_band (tuple): Frequency band as (low_freq, high_freq).

    Returns:
        float: The average of coherence_change values within the frequency band.
    """
    low_freq, high_freq = frequency_band
    indices = np.where((freqs >= low_freq) & (freqs <= high_freq))[0]
    
    if len(indices) == 0:
        print(f"No coherence_change values found in the frequency band {frequency_band}. Returning NaN.")
        return np.nan

    return np.mean(coherence_array[indices])

def perform_wilcoxon_tests(summary_df, freqs, frequency_band=(3,6), output_csv_path=None):
    """
    Perform Wilcoxon signed-rank tests on 'coh_avg' for both 'structured' and 'random' conditions
    within each region to test if their medians differ from zero.

    Args:
        summary_df (pd.DataFrame): The summary DataFrame containing the data.
        freqs (np.ndarray): Array of frequency values corresponding to coherence_change.
        frequency_band (tuple, optional): Frequency band as (low_freq, high_freq).
        output_csv_path (str, optional): Path to save the test results as a CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the test results for each region and condition.
    """
    # Identify unique regions
    unique_regions = summary_df['region_name'].unique()
    print(f"Unique regions found: {unique_regions}")

    # Initialize list to store results
    results = []

    # Process each region separately
    for region in unique_regions:
        print(f"\nProcessing region: {region}")

        # Subset data for the current region
        region_df = summary_df[summary_df['region_name'] == region]

        # Separate 'structured' and 'random' conditions
        for condition in ['structured', 'random']:
            condition_df = region_df[region_df['condition'] == condition].copy()

            if condition_df.empty:
                print(f"No data for condition '{condition}' in region '{region}'. Skipping.")
                continue

            # Compute the average of coherence_change within the frequency band
            condition_df['coh_avg'] = condition_df['coherence_change'].apply(
                lambda x: compute_coherence_average(x, freqs, frequency_band=frequency_band)
            )

            # Extract the averaged coherence changes, excluding NaN values
            coh_avg = condition_df['coh_avg'].dropna().values

            if len(coh_avg) == 0:
                print(f"No valid coh_avg data for condition '{condition}' in region '{region}'. Skipping.")
                stat, p = np.nan, np.nan
            else:
                # Perform Wilcoxon signed-rank test against zero
                try:
                    if len(coh_avg) < 10:
                        stat, p = wilcoxon(coh_avg, zero_method='pratt', alternative='greater', mode='exact')
                    else:
                        stat, p = wilcoxon(coh_avg, zero_method='wilcox', alternative='greater')
                    # stat, p = wilcoxon(coh_avg, zero_method='wilcox', alternative='greater')
                    print(f"Wilcoxon signed-rank test for '{condition}_coh_avg' in region '{region}': statistic={stat}, p-value={p}")
                except ValueError as e:
                    print(f"Wilcoxon signed-rank test could not be performed for '{condition}_coh_avg' in region '{region}': {e}")
                    stat, p = np.nan, np.nan

            # Append results
            results.append({
                'region': region,
                'condition': condition,
                'wilcoxon_statistic': stat,
                'wilcoxon_p_value': p,
                'num_samples': len(coh_avg)
            })

    # Create a results DataFrame
    results_df = pd.DataFrame(results)

    # Optionally save the results to a CSV file
    if output_csv_path:
        results_df.to_csv(output_csv_path, index=False)
        print(f"\nWilcoxon signed-rank test results saved to {output_csv_path}")

    return results_df

def main():
    # Define paths (modify these paths as needed)
    output_dir = processed_data_dir 
    summary_csv = "coherence_summary.csv"              
    summary_pickle = "coherence_summary.pkl"           # Replace with your actual pickle filename

    summary_csv_path = os.path.join(output_dir, summary_csv)
    summary_pickle_path = os.path.join(output_dir, summary_pickle)

    # Load the summary DataFrame
    try:
        summary_df = load_summary_dataframe(pickle_path=summary_pickle_path)
    except FileNotFoundError:
        # If pickle not found, try loading from CSV
        summary_df = load_summary_dataframe(csv_path=summary_csv_path)

    # Define frequency array
    start_freq = 1.953125
    end_freq = 68.359375
    step_size = 1.953125
    freqs = np.arange(start_freq, end_freq + step_size, step_size)

    # Define the frequency band to analyze
    frequency_band = (1, 4)  # Example: 3-6 Hz

    # Perform Wilcoxon signed-rank tests
    results_df = perform_wilcoxon_tests(
        summary_df, 
        freqs=freqs,
        frequency_band=frequency_band,
        output_csv_path=os.path.join(".", "wilcoxon_results.csv")
    )

    # Display the results
    print("\nWilcoxon Signed-Rank Test Results:")
    print(results_df)

if __name__ == "__main__":
    main()

