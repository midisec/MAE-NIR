import numpy as np
import pandas as pd


def simplified_lhs(n, samples):
    # Generate a regular grid
    grid = np.linspace(0, 1, samples + 1)

    # Shuffle the grid values
    permuted_values = [np.random.permutation(grid[:-1]) for _ in range(n)]

    # Take random points from intervals
    points = []
    for i in range(samples):
        sample = [permuted_values[j][i] + np.random.rand() * (grid[1] - grid[0]) for j in range(n)]
        points.append(sample)

    return np.array(points)


def clhs_fixed_simplified_lhs(data, train_fraction=0.666):
    n_samples = data.shape[0]
    n_train = int(train_fraction * n_samples)

    # LHS sampling
    lhd = simplified_lhs(data.shape[1], n_train)

    # Scale LHS values to data range
    lhd_scaled = np.zeros(lhd.shape)
    for i in range(data.shape[1]):
        col_data = data.iloc[:, i].values
        lhd_scaled[:, i] = lhd[:, i] * (col_data.max() - col_data.min()) + col_data.min()

    # Find closest rows in data to the LHS samples without repetition
    train_indices = []
    remaining_data = data.copy().values
    remaining_indices = np.array(range(n_samples))
    for i in range(n_train):
        closest_index = np.argmin(np.linalg.norm(remaining_data - lhd_scaled[i, :], axis=1))
        train_indices.append(remaining_indices[closest_index])

        # Remove the selected data point
        remaining_data = np.delete(remaining_data, closest_index, axis=0)
        remaining_indices = np.delete(remaining_indices, closest_index, axis=0)

    train_indices = np.array(train_indices)
    test_indices = np.array(list(set(range(n_samples)) - set(train_indices)))

    return data.iloc[train_indices, :], data.iloc[test_indices, :]


# Load the dataset
data = pd.read_csv('AnHui.HuangShan.SOIL.csv')

# Split the data using the CLHS method
train_data_clhs_fixed, test_data_clhs_fixed = clhs_fixed_simplified_lhs(data)

# Save the train and test sets
train_data_clhs_fixed.to_csv('AnHui.HuangShan.SOIL.train.csv', index=False)
test_data_clhs_fixed.to_csv('AnHui.HuangShan.SOIL.test.csv', index=False)
