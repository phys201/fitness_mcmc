from unittest import TestCase
import fitness_mcmc as fm
import fitness_mcmc.data_io as io
import numpy as np
import pymc3 as pm 

data, time, ordered_frequencies  = io.load_data('filtered_counts_ypd_temp_30.txt')
fitness_model = fm.Fitness_Model(ordered_frequencies[0:10],time)

class TestFunc(TestCase):
    
    def test_normalize_func(self):
        """
        tests the normalize_fun function from fitness_mcmc.py to see if it agrees with the 
        expected value 1
        """
        
        x=np.array(10)
        t=fm.normalize_func(x**2)
        
        message = "First value and second value are not equal !"
       
        self.assertEqual(t, 1, message)
        
    def test_is_sample_lineages(self):
        """
        tests whether the package "fitness_mcmc" loads correctly
        """
        data, time, ordered_frequencies  = io.load_data('filtered_counts_ypd_temp_30.txt')

        num_samples=40
        q=fm.sample_lineages(ordered_frequencies, num_samples)
        
        self.assertTrue(isinstance(q,  np.ndarray))
        
    def test_create_trajectories(self): 
        """
        tests whether create_lineages produces sensible output
        """
        N = 40 
        s_range = 0.1
        f0_vals = np.random.random(N)
        s_vals = np.random.random(N) * s_range
        times = np.array([5, 10, 25, 40, 45])
        trajectory = fm.create_trajectories(f0_vals, s_vals, times)
        
        self.assertEqual(len(trajectory),N)
        
    def test_mcmc_sample(self):
        MAP = fitness_model.mcmc_sample()
        self.assertEqual(len(fitness_model.trace),4)
    def test_plot_mcmc_posterior(self):
        fitness_model.plot_mcmc_posterior()
        self.assertEqual(len(fitness_model.trace),4)
    def test_plot_mcmc_trace(self):
        trace = fitness_model.plot_mcmc_trace()
        self.assertEqual(len(fitness_model.trace.observed_data.f_obs_dim_0),len(ordered_frequencies[0:10]))
    def test_find_MAP(self):
        fitness_model.find_MAP() 
        self.assertTrue(isinstance(fitness_model.map_estimate["f0"],np.ndarray))
    def test_plot_MAP_estimate(self): 
        plot = fitness_model.plot_MAP_estimate(type = "lin")
        self.assertTrue(isinstance(fitness_model.f_pred, np.ndarray))