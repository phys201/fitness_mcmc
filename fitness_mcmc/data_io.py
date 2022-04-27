import pandas as pd
import numpy as np
import os

import sys 
sys.path.append('..')

from fitness_mcmc import create_trajectories, sample_lineages

def _get_file_path(filename, data_dir):
    
    start = os.path.abspath(__file__)
    start_dir = os.path.dirname(start)
    data_dir = os.path.join(start_dir, data_dir)
    data_path = os.path.join(start_dir, data_dir, filename)
    
    return data_path


def load_data(filename, data_dir = 'experimental_data'):
    
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
    """
    data_file = _get_file_path(filename, data_dir)
    data = pd.read_csv(data_file, delimiter = '\t')
    time = [int(i) for i in data.columns[1:]]
    counts = data.loc[:,data.columns[1:]].to_numpy()
    frequencies = np.zeros(np.shape(counts.T))
    for i in range(len(counts[0,:])):
        frequencies[i] = counts[:,i]/np.sum(counts[:,i])
    idx = np.flipud(np.argsort(np.sum(frequencies.T, axis = 1)))
    ordered_frequencies = frequencies.T[idx,:]

    return data, time, ordered_frequencies
