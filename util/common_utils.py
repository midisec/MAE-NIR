import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import math


# This file contains common utility functions for data preprocessing in spectral analysis.


# ---- Savitzky-Golay Smoothing (SG) ----
def SG_smoothing(data_x, window_size=15, rank=2):
    """
    Applies Savitzky-Golay smoothing on the input data.

    Parameters:
        data_x (pd.DataFrame): The input data.
        window_size (int): The window size for smoothing. Default is 15.
        rank (int): The polynomial rank for smoothing. Default is 2.

    Returns:
        pd.DataFrame: The smoothed data.
    """

    # Helper function to create the matrix X for Savitzky-Golay algorithm
    def create_x(size, rank):
        x = []
        for i in range(2 * size + 1):
            m = i - size
            row = [m ** j for j in range(rank)]
            x.append(row)
        return np.mat(x)

    # Helper function to apply Savitzky-Golay smoothing
    def sg(data_x, window_size, rank):
        m = int((window_size - 1) / 2)
        odata = data_x[:]
        for i in range(m):
            odata.insert(0, odata[0])
            odata.insert(len(odata), odata[len(odata) - 1])
        x = create_x(m, rank)
        b = (x * (x.T * x).I) * x.T
        a0 = b[m]
        a0 = a0.T
        ndata = []
        for i in range(len(data_x)):
            y = [odata[i + j] for j in range(window_size)]
            y1 = np.mat(y) * a0
            y1 = float(y1)
            ndata.append(y1)
        return ndata

    # Main function logic for SG smoothing
    n = data_x.shape[0]
    data = np.zeros_like(np.array(data_x))
    for i in range(n):
        data[i, :] = data_x.iloc[i, :]
    ans = []
    for i in range(data.shape[0]):
        ans.append(sg(list(data[i, :]), window_size, rank))
    return pd.DataFrame(np.array(ans), columns=data_x.columns)


# ---- First Derivative (FD) ----
def FD(data_x):
    """
    Applies first derivative on the input data.

    Parameters:
        data_x (pd.DataFrame): The input data.

    Returns:
        pd.DataFrame: The differentiated data.
    """
    temp2 = data_x.diff(axis=1)
    temp3 = temp2.values
    return pd.DataFrame(np.delete(temp3, 0, axis=1), columns=data_x.columns[1:])


# ---- Second Derivative (SD) ----
def SD(data_x):
    """
    Applies second derivative on the input data.

    Parameters:
        data_x (pd.DataFrame): The input data.

    Returns:
        pd.DataFrame: The twice differentiated data.
    """
    temp2 = data_x.diff(axis=1)
    temp3 = np.delete(temp2.values, 0, axis=1)
    temp4 = (pd.DataFrame(temp3)).diff(axis=1)
    spec_D2 = np.delete(temp4.values, 0, axis=1)
    return pd.DataFrame(spec_D2, columns=data_x.columns[2:])


# ---- Standard Normal Variate (SNV) ----
def SNV(data):
    """
    Applies Standard Normal Variate on the input data.

    Parameters:
        data (pd.DataFrame): The input data.

    Returns:
        pd.DataFrame: The standardized data.
    """
    n = data.shape[0]
    data_x = np.zeros_like(np.array(data))
    for i in range(n):
        data_x[i, :] = data.iloc[i, :]
    n, p = data_x.shape
    snv_x = np.ones((n, p))
    data_std = np.std(data_x, axis=1)
    data_average = np.mean(data_x, axis=1)
    for i in range(n):
        for j in range(p):
            snv_x[i][j] = (data_x[i][j] - data_average[i]) / data_std[i]
    return pd.DataFrame(snv_x, columns=data.columns)


# ---- Multiple Scatter Correction (MSC) ----
def MSC(data):
    """
    Applies Multiple Scatter Correction on the input data.

    Parameters:
        data (pd.DataFrame): The input data.

    Returns:
        pd.DataFrame: The scatter corrected data.
    """
    n = data.shape[0]
    data_x = np.zeros_like(np.array(data))
    for i in range(n):
        data_x[i, :] = data.iloc[i, :]
    mean = np.mean(data_x, axis=0)
    n, p = data_x.shape
    msc_x = np.ones((n, p))
    for i in range(n):
        y = data_x[i, :]
        lin = LinearRegression()
        lin.fit(mean.reshape(-1, 1), y.reshape(-1, 1))
        k = lin.coef_
        b = lin.intercept_
        msc_x[i, :] = (y - b) / k
    return pd.DataFrame(msc_x, columns=data.columns)


# ---- Mean Centering (MC) ----
def MC(data):
    """
    Applies Mean Centering on the input data.

    Parameters:
        data (pd.DataFrame): The input data.

    Returns:
        pd.DataFrame: The mean centered data.
    """
    return pd.DataFrame(data - np.mean(data, axis=0))


# ---- Logarithm Transformation (LG) ----
def LG(data):
    """
    Applies Logarithm Transformation on the input data.

    Parameters:
        data (pd.DataFrame): The input data.

    Returns:
        pd.DataFrame: The logarithm transformed data.
    """
    n = data.shape[0]
    data_x = np.zeros_like(np.array(data))
    for i in range(n):
        data_x[i, :] = data.iloc[i, :]
    n, p = data_x.shape
    LG_x = np.ones((n, p))
    for i in range(n):
        for j in range(p):
            LG_x[i][j] = (math.log(1 / data_x[i][j], 10))
    return pd.DataFrame(LG_x, columns=data.columns)
