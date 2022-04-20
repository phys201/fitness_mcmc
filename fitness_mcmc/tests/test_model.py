from unittest import TestCase
from fitness_mcmc.fitness_mcmc import normalize_func
import numpy as np

class TestIo(TestCase):
    def test_normalize_func(self):
        """
        tests the normalize_fun function from fitness_mcmc.py to see if it agrees with the 
        expected value 1
        """
        
        x=np.array(10)
        t=normalize_func(x**2)
        
        message = "First value and second value are not equal !"
       
        self.assertEqual(t, 1, message)