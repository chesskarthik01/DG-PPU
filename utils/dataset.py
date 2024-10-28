import os
import glob
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


class CATMAUS(torch.utils.data.Dataset):
    def __init__(self, root, monte_carlo=True, **kwargs):
        """
        Initialises the dataset.

        Args:
            root (str): Root directory containing the .csv files
            monte_carlo (bool): Perform Monte Carlo sampling. Default is True
            kwargs: Additional arguments for num_points and interations when monte_carlo is True.
        """

        super(CATMAUS, self).__init__()
        self.root = root
        self.monte_carlo = monte_carlo
        self.num_points = kwargs.get('num_points', 1024)
        self.iterations = kwargs.get('iterations', 500)
        self.batch_size = kwargs.get('batch_size', 1024)
        self.translation_range = kwargs.get('translation_range', 0.02)
        self.jitter_std = kwargs.get('jitter_std', 0.01)
        self.jitter_clip = kwargs.get('jitter_clip', 0.02)
        self.data_list = self.load_data()

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.data_list)

    def __getitem__(self, idx):
        # Return a single Data object from the list
        return self.data_list[idx]

    def load_data(self):
        """
        Load all CSV files, combine the data, perform Monte Carlo sampling, and convert them into
        PyTorch Geometric Data objects.
        """
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        DATA_DIR = os.path.join(BASE_DIR, self.root)

        # Collect all data into a single array
        all_X = []
        all_y = []

        # Read each CSV file and concatenate into one dataset
        csv_files = glob.glob(os.path.join(DATA_DIR, '*.csv'))

        if not csv_files:
            raise ValueError("No CSV files found in the data directory!")

        for file_path in csv_files:
            df = pd.read_csv(file_path)
            X_data = df[['x', 'y', 'z']].values
            y_label = df['label'].values

            all_X.append(X_data)
            all_y.append(y_label)

        # Combine all point cloud data (X) and labels (y)
        X = np.vstack(all_X)
        y = np.hstack(all_y)

        print(f"Combined Data X shape: {X.shape}, y shape: {y.shape}")

        # Identify the minority class (class with the least number of entries)
        unique, counts = np.unique(y, return_counts=True)
        # Find the class with the least entries
        minority_class = unique[np.argmin(counts)]
        print(f"Minority class identified: {minority_class}")

        if not self.monte_carlo:
            return self.load_data_in_batches(X, y)
        else:
            return self.monte_carlo_sampling(X, y, self.num_points, self.iterations, minority_class)

    def load_data_in_batches(self, X, y):
        """
        Split the entire dataset into smaller batches when monte_carlo=False.
        """
        num_rows = X.shape[0]
        samples_list = []

        # Split the dataset into batchs (without replacement)
        num_batchs = num_rows // self.batch_size
        for i in range(num_batchs):
            start_idx = i * self.batch_size
            end_idx = (i + 1) * self.batch_size
            X_batch = X[start_idx:end_idx, :]
            y_batch = y[start_idx:end_idx]

            # Convert batch data to PyTorch Geometric Data object
            pos = torch.tensor(X_batch, dtype=torch.float)
            y_tensor = torch.tensor(y_batch, dtype=torch.long)
            data = Data(pos=pos, y=y_tensor)
            samples_list.append(data)

        # Handle the remainder if it doesn't perfectly divide into batch_size
        remainder = num_rows % self.batch_size
        if remainder > 0:
            X_batch = X[-remainder:, :]
            y_batch = y[-remainder:]
            pos = torch.tensor(X_batch, dtype=torch.float)
            y_tensor = torch.tensor(y_batch, dtype=torch.long)
            data = Data(pos=pos, y=y_tensor)
            samples_list.append(data)

        print(f"Dataset split into {len(samples_list)} batches.")
        return samples_list

    def monte_carlo_sampling(self, X, y, sample_size=1024, iterations=500, minority_class=None):
        """
        Monte Carlo sampling with verbose logging to show progress.
        """
        num_rows = X.shape[0]
        samples_list = []
        total_dataset_size = 0

        for i in range(iterations):
            # Randomly sample points (with replacement)
            random_indices = np.random.choice(
                num_rows, size=sample_size, replace=True)
            X_sampled = X[random_indices, :]
            y_sampled = y[random_indices]

            # Apply translation and jittering to the minority class
            X_sampled_augmented = X_sampled.copy()
            minority_mask = (y_sampled == minority_class)

            # Log the number of Patella points selected for augmentation
            num_patella_points = np.sum(minority_mask)
            print(
                f'Iteration {i+1}: Number of Patella points before augmentation: {num_patella_points}')

            # Ensure translation and jittering are applied to the minority class
            if np.any(minority_mask):  # Check if there are any minority class points
                # Augment the Patella points
                X_augmented = self.apply_translation(
                    X_sampled_augmented[minority_mask])
                X_augmented = self.apply_jitter(X_augmented)

                # Concatenate the augmented points to the existing dataset
                X_sampled_augmented = np.vstack(
                    (X_sampled_augmented, X_augmented))
                # Same labels for augmented points
                y_augmented = np.full(X_augmented.shape[0], minority_class)
                y_sampled = np.hstack((y_sampled, y_augmented))

            # Log the number of Patella points after augmentation (should increase)
            num_augmented_patella_points = np.sum(y_sampled == minority_class)
            print(
                f'Iteration {i+1}: Number of Patella points after augmentation: {num_augmented_patella_points}')

            # Convert sampled data to PyTorch Geometric Data object
            pos = torch.tensor(X_sampled_augmented, dtype=torch.float)
            y_tensor = torch.tensor(y_sampled, dtype=torch.long)
            data = Data(pos=pos, y=y_tensor)
            samples_list.append(data)

            total_dataset_size += X_sampled_augmented.shape[0]

            # Verbose logging: show the progress of Monte Carlo sampling
            print(f"Iteration {i+1}/{iterations}:")
            print(
                f"Total size of dataset so far: {total_dataset_size} points")
            print("First 5 data points (X):")
            print(X_sampled[:5])
            print("First 5 labels (y):")
            print(y_sampled[:5])

        return samples_list

    def apply_translation(self, X):
        """
        Apply a random translation to the point cloud within a specified range.
        """
        translation = np.random.uniform(-self.translation_range,
                                        self.translation_range, size=(X.shape[0], 3))
        return X + translation

    def apply_jitter(self, X):
        """
        Apply jittering (Gaussian noise) to the point cloud.
        """
        noise = np.random.normal(0, self.jitter_std, size=(X.shape[0], 3))
        noise = np.clip(noise, -self.jitter_clip, self.jitter_clip)
        return X + noise
