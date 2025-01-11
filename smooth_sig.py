import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from pathlib import Path

def detect_isolated_items(bool_arr, isolation_len=1):
    """
    Detects isolated NaNs in the signal. Isolated NaNs can have a length of 1 or 2 within a sequence of numbers.

    Parameters:
        signal (numpy.ndarray): Input 1D array with numbers and NaNs.

    Returns:
        isolated_nans (list of tuple): List of tuples, where each tuple represents the start and end indices of isolated NaNs.
    """
    if not isinstance(bool_arr, np.ndarray):
        raise ValueError("Signal must be a numpy array.")

    # Detect NaN locations
    isolated_items = []

    start = None
    for i, val in enumerate(bool_arr):
        if val and start is None:  # Start of a NaN sequence
            start = i
        elif not val and start is not None:  # End of a NaN sequence
            end = i
            length = end - start
            # Check if the NaN sequence is isolated
            if length <= isolation_len and (start == 0 or not bool_arr[start - 1]) and (end == len(bool_arr) or not bool_arr[end]):
                isolated_items.append((start, end - 1))
            start = None

    # Handle case where the signal ends with NaNs
    if start is not None:
        end = len(bool_arr)
        length = end - start
        if length <= 2 and (start == 0 or not bool_arr[start - 1]):
            isolated_items.append((start, end - 1))

    return isolated_items


def handle_isolated_items(x):
    y = x.copy()
    isolated_nans = detect_isolated_items(np.isnan(x))
    isolated_values = detect_isolated_items(~np.isnan(x))
    if len(isolated_values) > 0:
        for i1, i2 in isolated_values:
            pupil_l[i1:i2] = np.nan
    if len(isolated_nans) > 0:
        win = np.arange(-3, 4, 1)
        for i1, i2 in isolated_nans:
            idx = win + i1
            if i1 == i2:
                idx = np.setxor1d(idx, i1)
                y[i1] = x[idx].mean()
            else:
                idx = np.setxor1d(idx, [i1, i2])
                y[i1:i2] = x[idx].mean()
    return x

def split_signal(signal):    
    # Identify where NaN values are
    is_nan = np.isnan(signal)
    
    # Find indices where the signal transitions from non-NaN to NaN and vice versa
    split_points = np.where(np.diff(is_nan.astype(int)) != 0)[0] + 1
    
    # Split the signal at these indices
    segments = np.split(signal, split_points)
    
    # Filter out segments that are all NaNs
    filtered_segments = [seg[~np.isnan(seg)] for seg in segments if not np.all(np.isnan(seg))]
    
    return filtered_segments


def interp_signal(signal):
    signal2 = signal.copy()
    # Identify the indices of valid (non-NaN) values
    valid_indices = ~np.isnan(signal)
    
    # Indices where NaNs are present
    nan_indices = np.isnan(signal)
    
    # Interpolate all NaN values using numpy.interp
    signal2[nan_indices] = np.interp(np.flatnonzero(nan_indices), np.flatnonzero(valid_indices), signal[valid_indices])
    
    return signal2

if __name__ == "__main__":
    fnames = sorted(glob.glob(r'../../datasets/eyetracking/Eye-tracking-Kaggle/*.csv'))
    data = pd.read_csv(fnames[0],  low_memory=False)
    idx_good = (data['Category Left'] == data['Category Right'])
    data = data[idx_good]
    t = data['RecordingTime [ms]']
    pupil_l = pd.to_numeric(data['Pupil Diameter Left [mm]'].values[1:], errors='coerce', downcast='float')
    pupil_r = pd.to_numeric(data['Pupil Diameter Right [mm]'].values[1:], errors='coerce', downcast='float')    
    
    pupil_l2 = handle_isolated_items(pupil_l)
    pupil_l3 = interp_signal(pupil_l2)


    pupil_r2 = handle_isolated_items(pupil_r)    
    pupil_r3 = interp_signal(pupil_r2)

    i1 = np.where(pupil_l3 != pupil_l)[0]
    i2 = np.where(pupil_l3 != pupil_l2)[0]
    
    fig, ax = plt.subplots(2)
    ax[0].plot(pupil_l[:3000])
    ax[0].plot(pupil_l2[:3000])
    ax[0].plot(pupil_l3[:3000])

    ax[1].plot(pupil_r[:3000])
    # ax[1].plot(pupil_r2[:3000])
    # ax[1].plot(pupil_r3[:3000])

    plt.show()


    