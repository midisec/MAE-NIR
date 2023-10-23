import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from util.common_utils import SG_smoothing, FD, SD, SNV, MSC, MC, LG


class SpectralDataSet(Dataset):
    def __init__(self, dataset_type, preprocessing=None, to_2d=False, matrix_size=None):
        self.LUCAS_SOIL_DATA_PATH = "./data/datasets/LUCAS2009SoilDatasets/LUCAS.SOIL_corr.csv"
        self.AnHui_SOIL_DATA_PATH = "./data/datasets/AnhuiSoilDatasets/AnHui.HuangShan.SOIL.csv"

        if dataset_type == "AnhuiSoil":
            self.file_path = self.AnHui_SOIL_DATA_PATH
            data_raw = pd.read_csv(self.file_path, low_memory=False)
            self.data = data_raw.iloc[:, :-3]  # Extracting spectral data
            self.labels = data_raw.iloc[:, -3:].to_numpy()  # Extracting N, P, K labels

        if dataset_type == "LucasSoil":
            self.file_path = self.LUCAS_SOIL_DATA_PATH
            data_raw = pd.read_csv(self.file_path, low_memory=False)
            self.data = data_raw.iloc[:, 4:-84]
            # self.labels = data_raw.iloc[:, -84:].to_numpy()  # Extracting labels
            self.labels = np.zeros((self.data.shape[0], 1))

        if preprocessing:
            if preprocessing == "SG":
                self.data = SG_smoothing(self.data)
            elif preprocessing == "FD":
                self.data = FD(self.data)
            elif preprocessing == "SD":
                self.data = SD(self.data)
            elif preprocessing == "SNV":
                self.data = SNV(self.data)
            elif preprocessing == "MSC":
                self.data = MSC(self.data)
            elif preprocessing == "MC":
                self.data = MC(self.data)
            elif preprocessing == "LG":
                self.data = LG(self.data)
            else:
                raise ValueError(f"Unknown preprocessing method: {preprocessing}")

        self.data = self.data.to_numpy()

        if to_2d:
            if not matrix_size:
                raise ValueError("Please specify matrix_size for 2D conversion.")
            self.data = self.convert_to_2d(self.data, matrix_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        spectral_data = self.data[index, :-1]
        label = self.labels[index]
        return np.array(spectral_data, dtype=np.float32), label

    def convert_to_2d(self, data, matrix_size):
        new_data = []
        for spectrum in data:
            if len(spectrum) > matrix_size * matrix_size:
                raise ValueError("Matrix size too small to accommodate the spectral data.")
            two_d_spectrum = np.zeros((matrix_size, matrix_size))
            rows = len(spectrum) // matrix_size
            for i in range(rows):
                two_d_spectrum[i, :len(spectrum[i*matrix_size:(i+1)*matrix_size])] = spectrum[i*matrix_size:(i+1)*matrix_size]
            new_data.append(two_d_spectrum)
        return np.array(new_data)


if __name__ == '__main__':

    # spectral_data = SpectralDataSet("AnhuiSoil", preprocessing="MSC")
    # spectral_data = SpectralDataSet("LucasSoil", preprocessing="MSC")
    spectral_data = SpectralDataSet("LucasSoil")
    spectral_loader = DataLoader(spectral_data, shuffle=True, batch_size=16, num_workers=0)


    for batch_data, batch_label in spectral_loader:
        print(batch_data.shape)
        print(batch_label.shape)
        break


