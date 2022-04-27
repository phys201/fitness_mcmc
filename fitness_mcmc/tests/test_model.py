from unittest import TestCase
from fitness_mcmc.fitness_mcmc import normalize_func, Fitness_Model
from fitness_mcmc.data_io import load_data
import numpy as np
import pymc3 as pm 

data, time, ordered_frequencies= load_data('filtered_counts_ypd_temp_30.txt')

class TestIo(TestCase):
    
    fitness_model = Fitness_Model(ordered_frequencies,time) 

    def test_model_is_valid(self):
        message = "something went wrong with model loading" 
        self.assertTrue(isinstance(self.fitness_model.model, pm.Model),message)
        
    