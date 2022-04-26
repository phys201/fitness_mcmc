import pandas as pd
import numpy as np
from pathlib import Path
from fitness_mcmc import create_trajectories, sample_lineages

def _get_file_path(filename):

    # Path.cwd() returns the location of the source file currently in use (so
    # in this case io.py). We can use it as base path to construct
    # other paths from that should end up correct on other machines or
    # when the package is installed
    current_directory = Path.cwd()
    data_path = Path(current_directory, filename)
    return data_path


def load_data(filename):
    
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
    data_file = _get_file_path(filename)
    data = pd.read_csv(data_file, delimiter = '\t')
    time = [int(i) for i in data.columns[1:]]
    counts = data.loc[:,data.columns[1:]].to_numpy()
    frequencies = np.zeros(np.shape(counts.T))
    for i in range(len(counts[0,:])):
        frequencies[i] = counts[:,i]/np.sum(counts[:,i])
    idx = np.flipud(np.argsort(np.sum(frequencies.T, axis = 1)))
    ordered_frequencies = frequencies.T[idx,:]

    return data, time, ordered_frequencies

def write_simulated_datafile(filename, N = 40, times = -1, s_range = 0.1,
                             depth = 100):
    """
    Creates a textfile of simulated trajectories formated like a real datafile

    Params:
        filename [str]: name of the output file
        N [int]: population size
        times [array_like]: times, in generations, to sample lineages.
        s_range [float]: range of fitness values
        depth [int_or_float]: Simulated read depth, affects noise
    """
    f0_vals = np.random.random(N)
    s_vals = np.random.random(N) * s_range
    if times == -1:
        times = np.array([5, 10, 25, 40, 45])
    else:
        times = np.array(times)

    trajectory = create_trajectories(f0_vals, s_vals, times)
    sampled = pd.DataFrame(sample_lineages(trajectory, depth * N),
                           columns = times)

    sampled.to_csv(filename, sep="\t", index_label = "BC")
