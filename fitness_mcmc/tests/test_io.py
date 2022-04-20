from unittest import TestCase
from fitness_mcmc.data_io import _get_file_path, load_data

class TestIo(TestCase):
    def test_data_io(self):
        file_path = _get_file_path('filtered_counts_ypd_temp_30.txt')
        data = load_data(file_path)
        #data = di.load_data('Final_project/fitness_mcmc/filtered_counts_ypd_temp_30.txt')
        assert data['data'][0] == 1682
    def test_simulated_data(self):
        simulated_file_path = _get_file_path('simulated_data.txt')
        simulated_data = load_data(simulated_file_path)
        assert simulated_data['simulated_data'][0] == 90