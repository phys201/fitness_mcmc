from unittest import TestCase
from fitness_mcmc.fitness_mcmc import normalize_func
import numpy as np

class TestIo(TestCase):
    def test_normalize_func(self):
        
        x=np.array(10)
        t=normalize_func(x**2)
        
        message = "First value and second value are not equal !"
       
        self.assertEqual(t, 1, message)