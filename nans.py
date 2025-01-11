import numpy as np

def detect_isolated_items(is_nan, isolation_len=1):
    """
    Detects isolated NaNs in the signal. Isolated NaNs can have a length of 1 or 2 within a sequence of numbers.

    Parameters:
        signal (numpy.ndarray): Input 1D array with numbers and NaNs.

    Returns:
        isolated_nans (list of tuple): List of tuples, where each tuple represents the start and end indices of isolated NaNs.
    """
    if not isinstance(signal, np.ndarray):
        raise ValueError("Signal must be a numpy array.")

    # Detect NaN locations
    # is_nan = np.isnan(nan_sig)
    isolated_nans = []

    start = None
    for i, val in enumerate(is_nan):
        if val and start is None:  # Start of a NaN sequence
            start = i
        elif not val and start is not None:  # End of a NaN sequence
            end = i
            length = end - start
            # Check if the NaN sequence is isolated
            if length <= isolation_len and (start == 0 or not is_nan[start - 1]) and (end == len(is_nan) or not is_nan[end]):
                isolated_nans.append((start, end - 1))
            start = None

    # Handle case where the signal ends with NaNs
    if start is not None:
        end = len(is_nan)
        length = end - start
        if length <= 2 and (start == 0 or not is_nan[start - 1]):
            isolated_nans.append((start, end - 1))

    return isolated_nans

# Example usage
signal = np.array([1, 1, 11, 11, np.nan, 1, 11, 11, 11, np.nan, 2, 3, np.nan, np.nan, np.nan, 5])
isolated_nans = detect_isolated_items(np.isnan(signal), isolation_len=3)

print("Isolated NaNs:", isolated_nans)

isolated_nans = detect_isolated_items(~np.isnan(signal), isolation_len=3)

print("Isolated NaNs:", isolated_nans)
