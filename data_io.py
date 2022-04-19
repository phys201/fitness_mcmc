import pandas as pd
import numpy as np 
from pathlib import Path

def _get_file_path(filename):
    
    # Path.cwd() returns the location of the source file currently in use (so
    # in this case io.py). We can use it as base path to construct
    # other paths from that should end up correct on other machines or
    # when the package is installed
    current_directory = Path.cwd()
    data_path = Path(current_directory, filename)
    return data_path


def load_data(filename):
    data_file = _get_file_path(filename)
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
    
    data = pd.read_csv(data_file, delimiter = '\t')
    time = [int(i) for i in data.columns[1:]]
    counts = data.loc[:,data.columns[1:]].to_numpy()
    frequencies = np.zeros(np.shape(counts.T))
    for i in range(len(counts[0,:])):
        frequencies[i] = counts[:,i]/np.sum(counts[:,i])
    idx = np.flipud(np.argsort(np.sum(frequencies.T, axis = 1)))
    ordered_frequencies = frequencies.T[idx,:]
    
    return data, time, ordered_frequencies 