from unittest import TestCase
import pandas as pd
from fitness_mcmc.data_io import _get_file_path, load_data

#returns a tuple of data, generations and ordered_frequencies
data = load_data('filtered_counts_ypd_temp_30.txt')

class TestIo(TestCase):
            
    def test_is_dataframe(self):
        """
        Makes sure that when we load the test data, we are created a Pandas DataFrame as expected
        """
        message = "A pandas dataframe was not created" 
        
        self.assertTrue(isinstance(data[0], pd.DataFrame),message)
        
    def test_data_io(self):
        """
        gets the path of the file and loads real data, and finds first value of column 5 to compare it with the actual value
        """
        data = load_data('filtered_counts_ypd_temp_30.txt')
        
        new_dataframe=data[0]
        my_value=new_dataframe['5'][0]
        
        message = "First value and second value are not equal !"
       
        self.assertEqual(my_value, 1682, message)
        
    def test_simulated_data(self):
        """
        gets the path of the file and loads the simulated data, and finds first value of column 5 to compare it with the actual value
        """
        simulated_data = load_data('simulated_data.txt', data_dir = 'simulated_data')
        
        new_dataframe_2=simulated_data[0]
        my_value_2=new_dataframe_2['5'][0]
        
        message_2 = "First value and second value are not equal !"
       
        self.assertEqual(my_value_2, 90, message_2)
        
        
    