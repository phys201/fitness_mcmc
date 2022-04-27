from unittest import TestCase

import fitness_mcmc

import fitness_mcmc.data_io as io 
import fitness_mcmc.fitness_mcmc as m
import pymc3 as pm
import numpy as np

class Test(TestCase):
        
    def test_is_sample_lineages(self):
        """
        tests whether the package "fitness_mcmc" loads correctly
        """
        data, time, ordered_frequencies  = io.load_data('filtered_counts_ypd_temp_30.txt')
        
        
        num_samples=40
        q=m.sample_lineages(ordered_frequencies, num_samples)
        
        self.assertTrue(isinstance(q,  np.ndarray))
       

        
       
