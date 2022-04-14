import numpy as np
import pymc3 as pm

class Fitness_Model:

    def __init__(self, data, times = -1):
        if times == -1:
            self.times = np.array([7, 14, 28, 42, 49]).reshape([1, -1])
        else:
            self.times = np.array(times).reshape([1, -1])
        self.num_times = len(self.times[0,:])
        self.data = np.array(data).reshape([-1, self.num_times])
        self.N = len(data[:, 0])
        self.model = pm.Model()
        with self.model:
            self.s = pm.Flat("s", shape = (self.N, 1))
            self.f0 = pm.HalfFlat("f0", shape = (self.N, 1))
            self.f = (self.f0 * pm.math.exp(self.s * self.times) /
                      pm.math.sum(self.f0 * pm.math.exp(self.s * self.times),
                      axis = 0) )
            self.f_obs = pm.Poisson("f_obs", mu = 100 * 1000 * self.f,
                                    observed = 100 * 1000 * self.data)

    def find_MAP(self):
        self.map_estimate = pm.find_MAP(model = self.model)

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
