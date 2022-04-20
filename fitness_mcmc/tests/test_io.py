from unittest import TestCase
from fitness_mcmc.data_io import _get_file_path, load_data

class TestIo(TestCase):
    def test_data_io(self):
        """
        gets the path of the file and loads real data, and finds first value of column 5 to compare it with the actual value
        """
        file_path = _get_file_path('filtered_counts_ypd_temp_30.txt')
        data = load_data(file_path)
        
        
        new_dataframe=data[0]
        my_value=new_dataframe['5'][0]
        
        message = "First value and second value are not equal !"
       
        self.assertEqual(my_value, 1682, message)
        
    def test_simulated_data(self):
        """
        gets the path of the file and loads the simulated data, and finds first value of column 5 to compare it with the actual value
        """
        simulated_file_path = _get_file_path('simulated_data.txt')
        simulated_data = load_data(simulated_file_path)
        
        new_dataframe_simulated=simulated_data[0]
        my_value_simulated=new_dataframe_simulated['5'][0]
        
        message_simulated = "First value and second value are not equal !"
       
        self.assertEqual(my_value_simulated, 90, message_simulated)