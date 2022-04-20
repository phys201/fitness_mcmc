from unittest import TestCase

class TestModel(TestCase):
    def test_sample_lineages(self):
        data = load_data('Final_project/fitness_mcmc/filtered_counts_ypd_temp_30.txt')
        assert data['data'][0] == 1682

