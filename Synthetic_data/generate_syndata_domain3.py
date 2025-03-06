import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def generate_third_domain_data(data_t1, data_t2, n_components=600):
    """
    Generate data for the third domain by applying PCA to the combined data of two domains.

    Parameters:
    - data_t1 (numpy.ndarray): Data from the first domain (shape: [samples, features])
    - data_t2 (numpy.ndarray): Data from the second domain (shape: [samples, features])
    - n_components (int): Number of principal components for PCA. Default is 600.

    Returns:
    - numpy.ndarray: Transformed data for the third domain with added Gaussian noise.
    """

    # Ensure inputs are NumPy arrays
    data_t1, data_t2 = np.asarray(data_t1), np.asarray(data_t2)

    # Check if inputs have at least one feature
    assert (
        data_t1.shape[1] > 0 and data_t2.shape[1] > 0
    ), "Input data must have at least one feature."

    # Equalize dimensions by truncating
    min_dimensions = min(data_t1.shape[1], data_t2.shape[1])
    data_t1, data_t2 = data_t1[:, :min_dimensions], data_t2[:, :min_dimensions]

    # Ensure n_components does not exceed min_dimensions
    n_components = min(n_components, min_dimensions)

    # Combine the data
    combined_data = np.vstack((data_t1, data_t2))

    # Standardize data for better PCA performance
    scaler = StandardScaler()
    combined_data = scaler.fit_transform(combined_data)

    # Apply PCA to combined data
    pca = PCA(n_components=n_components)
    pca.fit(combined_data)

    # Transform data from the 1st domain using learned PCA components
    data_t3 = pca.transform(data_t1)

    # Add Gaussian noise
    noise_std = 0.05 * np.std(data_t3, axis=0)  # Scale noise to feature variability
    data_t3 += np.random.normal(0, noise_std, data_t3.shape)

    return data_t3
