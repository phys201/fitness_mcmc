import pandas as pd
import numpy as np
import os
import sys
sys.path.append('..')

def _get_file_path(filename, data_dir):
    """
    Takes the file name and returns the absolute data file path given that
    the data file is found in data_dir

    Params:
        file_name[str]: file name of data file
        data_dir[str]: the directory where the file is located

        Returns:
        data_path[str]: the absolute path to file_name
    """
    start = os.path.abspath(__file__)
    start_dir = os.path.dirname(start)
    data_dir = os.path.join(start_dir, data_dir)
    data_path = os.path.join(start_dir, data_dir, filename)

    return data_path

def load_data(filename, data_dir = "experimental_data", load_metadata = False):
    #the default is data_directory is experimental_data
    """
    Takes data file path and outputs time generations and counts as numpy arrays

    Params:
        data_file[str]: path to data_file

    Returns:
        data [pandas_dataframe]: actual data where time and counts is stored
        time [array_like]: time generations when sampling would occur
        counts [array_like]: counts of genotypes
        frequencies [array_like]: normalized counts of genotypes
        load_metada [bool]: load true s and f0 values from simulated data
    """
    data_file = _get_file_path(filename, data_dir)
    data = pd.read_csv(data_file, delimiter = "\t")
    time = [int(i) for i in data.columns[1:]]
    counts = data.loc[:,data.columns[1:]].to_numpy()
    frequencies = np.zeros(np.shape(counts.T))
    if load_metadata:
        metadata_file = _get_file_path(
            filename.split(".txt")[0] + "_metadata.txt", data_dir
        )
        metadata = pd.read_csv(metadata_file, delimiter = "\t").to_numpy()[:,1:]
        s_vals = metadata[:, 0]
        f0_vals = metadata[:, 1]
    idx = np.flipud(np.argsort(np.sum(counts, axis = 1)))
    ordered_counts = counts[idx, :].astype("float")

    if load_metadata:
        s_vals = s_vals[idx]
        f0_vals = f0_vals[idx]
        return data, time, ordered_counts, s_vals, f0_vals
    else:
        return data, time, ordered_counts
