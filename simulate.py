import numpy as np

def normalize_func(x):
    """
    Normalizes frequencies to sum to 1

    Parameters:
        x: frequency vector to be normalized
    """
    return x / np.sum(x, axis = 0)

def create_trajectories(f0, s, times, normalize = True):
    """
    Simulates lineage trajectores given initial frequency and fitnesses

    Parameters:
        f0: initial lineage frequencies
        s: lineage fitnesses
        num_gens: number of generations to simulate
        dt: discretized time step for forward integration
    """
    f0 = f0.reshape([len(f0), -1])
    s = s.reshape([len(s), -1])
    times = times.reshape([-1, len(times)])
    f_traj = f0 * np.exp(s * times)

    if normalize:
        f_traj = normalize_func(f_traj)

    return f_traj

def sample_lineages(f, num_samples):
    """
    Returns lineage counts Poisson sampled from their true frequencies

    Parameters:
        f: true lineage frequencies
        num_samples: total number of samples to draw, should be order
            100 * num_lineages
    Returns:
        n_sampled: number of samples measured from each lineage
    """
    n_expected = f * num_samples
    n_sampled = np.random.poisson(n_expected)
    return n_sampled

def extract_at_time_pts(trajectory, tps = [7, 14, 28, 42, 49],
                        dt = 0.1, num_gens = -1):
    if num_gens == -1:
        num_gens = tps[-1]
    N = len(trajectory[0, :])
    time_array = np.arange(0, num_gens + .00001, dt)
    samples = np.zeros([5, N])

    for i, tp in enumerate(tps):
        j = np.where(time_array == tp)[0][0]
        samples[i, :] = trajectory[j, :]
    return samples
