from unittest import TestCase
import fitness_mcmc.fitness_mcmc as fm 
import fitness_mcmc.data_io as di
from fitness_mcmc.data_io import load_data
import numpy as np
import pymc3 as pm 

data, time, ordered_frequencies= di.load_data('filtered_counts_ypd_temp_30.txt')

class TestIo(TestCase):
    
    fitness_model = fm.Fitness_Model(ordered_frequencies,time) 

    def test_model_is_valid(self):
        message = "something went wrong with model loading" 
        self.assertTrue(isinstance(self.fitness_model.model, pm.Model),message)
    
    def test_sample_lineages(self): 
        message = "sampling went wrong" 
        
        sampled_lineages = fm.sample_lineages(ordered_frequencies,len(ordered_frequencies))
        
        self.assertEqual(len(sampled_lineages),len(ordered_frequencies), message)
        
    def test_normalize_func(self):
        """
        tests the normalize_fun function from fitness_mcmc.py to see if it agrees with the 
        expected value 1
        """
        
        x=np.array(10)
        t=fm.normalize_func(x**2)
        
        message = "First value and second value are not equal !"
       
        self.assertEqual(t, 1, message)