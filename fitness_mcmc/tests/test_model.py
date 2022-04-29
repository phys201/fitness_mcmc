from unittest import TestCase
import fitness_mcmc as fm
import fitness_mcmc.data_io as di
import numpy as np
import pymc3 as pm 

data, time, ordered_frequencies= di.load_data('filtered_counts_ypd_temp_30.txt')

class TestModel(TestCase):
    
    fitness_model = fm.Fitness_Model(ordered_frequencies,time) 

    def test_model_is_valid(self):
        message = "something went wrong with model loading" 
        self.assertTrue(isinstance(self.fitness_model.model, pm.Model),message)
    
