import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import arviz as az

class Fitness_Model:

    def __init__(self, data, times = -1, s_ref = 0):
        """
        Initializes the Fitness_Model class

        Params:
            data [array-like]: lineage counts over time. Shape:
                [# lineages, # times]
            times [array-like]: times, in generations, where lineages were
                sampled
        """
        if times == -1:
            self.times = np.array([7, 14, 28, 42, 49]).reshape([1, -1])
        else:
            self.times = np.array(times).reshape([1, -1])
        self.num_times = len(self.times[0,:])
        self.data = np.array(data).reshape([-1, self.num_times])
        self.N = len(data[:, 0])
        self.model = pm.Model()
        self.s_ref_val = 0
        with self.model:
            self.s_ref = pm.math.constant(s_ref, ndim = 2)
            self.f0_ref = pm.math.constant(1, ndim = 2)
            self.s = pm.Flat("s", shape = (self.N - 1, 1))
            self.f0 = pm.HalfFlat("f0", shape = (self.N - 1, 1))

            self.f_ref = (self.f0_ref * pm.math.exp(self.s_ref_val * self.times) /
                          (self.f0_ref * pm.math.exp(self.s_ref * times)
                          + pm.math.sum(self.f0 * pm.math.exp(self.s * self.times),
                          axis = 0)))
            self.f = (self.f0 * pm.math.exp(self.s * self.times) /
                      (self.f0_ref * pm.math.exp(self.s_ref * times)
                      + pm.math.sum(self.f0 * pm.math.exp(self.s * self.times),
                      axis = 0)))
            self.f_tot = pm.math.concatenate((self.f_ref, self.f))
            self.f_obs = pm.Poisson("f_obs", mu = 100 * 1000 * self.f_tot,
                                    observed = 100 * 1000 * self.data)

    def mcmc_sample(self):
        """
        Markov-chain Monte Carlo sample the posterior distribution of the
        pymc3 model
        """
        with self.model:
            self.trace = pm.sample(5000, return_inferencedata=True)

    def plot_mcmc_posterior(self):
        """
        Plots the posterior for several MCMC sampled parameters
        """
        with self.model:
            az.plot_posterior(self.trace)
    
    def plot_mcmc_trace(self):
        
        with self.model: 
            az.plot_trace(self.trace)
        

    def find_MAP(self):
        """
        Finds the MAP estimate for lineage fitnesses and starting frequencies
        """
        self.map_estimate = pm.find_MAP(model = self.model)

    def plot_MAP_estimate(self, type="log_y"):
        """
        Plots lineage trajectories from the MAP estimate

        Parameters:
            type [str]: either "log_y" or "lin", sets the y axis scale
        """
        f_pred = np.zeros_like(self.data)
        f_pred[1:, :] =  self.map_estimate["f0"] * np.exp(
                            self.map_estimate["s"] * self.times)
        f_pred[0] = np.exp(self.s_ref_val * self.times)
        f_pred /= np.sum(f_pred, axis = 0)

        fig, axs = plt.subplots(1,2)
        if type == "log_y":
            axs[0].semilogy(self.times.T, self.data.T)
            axs[1].semilogy(self.times.T, f_pred.T)
        elif type == "lin":
            axs[0].plot(self.times.T, self.data.T)
            axs[1].plot(self.times.T, f_pred.T)
        axs[0].set_xlabel("Generations")
        axs[1].set_xlabel("Generations")
        axs[0].set_ylabel("Lineage frequency")
        axs[0].set_title("Data")
        axs[1].set_title("Reconstructed")
        plt.show()

def normalize_func(x):
    """
    Normalizes lineage frequencies to sum to 1

    Parameters:
        x [array_like]: frequency vector to be normalized
    """
    return x / np.sum(x, axis = 0)

def create_trajectories(f0, s, times, normalize = True):
    """
    Simulates lineage trajectores given initial frequency and fitnesses

    Parameters:
        f0 [array-like]: initial lineage frequencies
        s [array-like]: lineage fitnesses
        times [array-like]: times, in generations, to sample lineage frequencies
        normalize [bool]: if True, normalizes lineage frequencies at each
            sampling time
    Returns:
        f_traj [numpy array]: array of lineage frequencies sampled at times
            given by "times"
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
        f [array_like]: true lineage frequencies
        num_samples [int]: total number of samples to draw, should be order
            100 * num_lineages
    Returns:
        n_sampled [numpy array]: number of samples measured from each lineage
    """
    n_expected = f * num_samples
    n_sampled = np.random.poisson(n_expected)
    return n_sampled
