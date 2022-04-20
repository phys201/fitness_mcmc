from unittest import TestCase
#import fitness_mcmc.data_io as di
from fitness_mcmc.fitness_mcmc.data_io  import _get_file_path, load_data

class TestIo(TestCase):
    def test_data_io(self):
        file_path = di.get_data_file_path('filtered_counts_ypd_temp_30.txt')
        data = di.load_data(file_path)
        #data = di.load_data('Final_project/fitness_mcmc/filtered_counts_ypd_temp_30.txt')
        assert data['data'][0] == 1682
    def test_simulated_data(self):
        simulated_file_path = di.get_data_file_path('simulated_filename.txt')
        simulated_data = di.load_data(simulated_file_path)
        #data = di.load_data('Final_project/fitness_mcmc/filtered_counts_ypd_temp_30.txt')
        assert simulated_data['data'][0] == 1682
        