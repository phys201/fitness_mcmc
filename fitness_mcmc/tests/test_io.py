from unittest import TestCase
from fitness_mcmc.data_io import _get_file_path, load_data

class TestIo(TestCase):
    def test_data_io(self):
        data = load_data('Final_project/fitness_mcmc/filtered_counts_ypd_temp_30.txt')
        assert data['data'][0] == 1682