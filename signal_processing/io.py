# signal_processing/io.py
import numpy as np
import pandas as pd


def read_cts_file(filepath):
    """Reads a .cts file, skipping header lines starting with # or CTS."""
    skip_rows = 0
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or line.startswith('CTS'):
                skip_rows += 1
            else:
                break
    return pd.read_csv(filepath, sep='\t', skiprows=skip_rows,
                       decimal=',', na_values=['NaN'])


import numpy as np

def trim_by_derivative(toco_signal, fhr_signal, threshold=1.0):
    """
    Finds the start of real data by detecting the first significant
    change in the TOCO signal, and then ensuring the FHR signal is available.
    """
    diffs = np.abs(np.diff(toco_signal, prepend=toco_signal[0]))
    toco_changes = np.where(diffs > threshold)[0]
    
    toco_start = toco_changes[0] + 240 if len(toco_changes) > 0 else 0
    
    toco_start = min(toco_start, len(fhr_signal) - 1)

    fhr_slice = fhr_signal[toco_start:]
    valid_fhr_indices = np.where((~np.isnan(fhr_slice)) & (fhr_slice > 0))[0]
    
    if len(valid_fhr_indices) > 0:
        final_start = toco_start + valid_fhr_indices[0]
        return final_start
    else:
        return toco_start