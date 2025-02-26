import h5py
import numpy as np

# Paths to the input and output HDF5 files
input_file_path = "/homes/marcus/Dokumenter/LDM_precomputed_latens/AdaIN_computations/data/all_patients_V3H2O.h5"
output_file_path = "/homes/marcus/Dokumenter/LDM_precomputed_latens/AdaIN_computations/data/all_patients_V3H2O_agg.h5"

# Dictionary to store aggregated medians for each group
aggregated_stats = {}
all_means = []
all_variances = []

# Open the original HDF5 file and aggregate statistics for each group
with h5py.File(input_file_path, "r") as h5f:
    for label_key in h5f.keys():
        # Read means and variances for the current group
        means = h5f[label_key]['means'][:]  # Shape: (num_samples, num_channels)
        variances = h5f[label_key]['variances'][:]  # Shape: (num_samples, num_channels)
        
        # Calculate aggregated median per channel
        aggregated_mean = np.median(means, axis=0)  # Shape: (num_channels,)
        aggregated_variance = np.median(variances, axis=0)  # Shape: (num_channels,)
        
        # Store in the dictionary with label_key as the identifier
        aggregated_stats[label_key] = (aggregated_mean, aggregated_variance)
        
        # Collect means and variances for overall aggregation, per channel
        all_means.append(means)
        all_variances.append(variances)

# Concatenate all means and variances across groups, then compute global channel-wise statistics
all_means = np.concatenate(all_means, axis=0)  # Shape: (total_samples, num_channels)
all_variances = np.concatenate(all_variances, axis=0)  # Shape: (total_samples, num_channels)
overall_mean_of_means = np.mean(all_means, axis=0)  # Shape: (num_channels,)
overall_mean_of_variances = np.mean(all_variances, axis=0)  # Shape: (num_channels,)

# Save the aggregated statistics to a new HDF5 file
with h5py.File(output_file_path, "w") as h5f_agg:
    for label_key, (agg_mean, agg_var) in aggregated_stats.items():
        group = h5f_agg.create_group(label_key)
        group.create_dataset("aggregated_mean", data=agg_mean)  # Channel-wise aggregated means
        group.create_dataset("aggregated_variance", data=agg_var)  # Channel-wise aggregated variances
    
    # Add the "all" group with the global statistics
    all_group = h5f_agg.create_group("all")
    all_group.create_dataset("aggregated_mean", data=overall_mean_of_means)
    all_group.create_dataset("aggregated_variance", data=overall_mean_of_variances)

print(f"Aggregated channel-wise statistics saved to {output_file_path}")
