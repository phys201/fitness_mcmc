import numpy as np
import pymc3 as pm

class Fitness_Model:

    def __init__(self, data, times = -1):
        if times == -1:
            self.times = np.array([7, 14, 28, 42, 49]).reshape([1, -1])
        else:
            self.times = np.array(times).reshape([1, -1])
        self.num_times = len(self.times[0,:])
        self.data = np.array(data).reshape([-1, self.num_times])
        self.N = len(data[:, 0])
        self.model = pm.Model()
        with self.model:
            self.s = pm.Flat("s", shape = (self.N, 1))
            self.f0 = pm.HalfFlat("f0", shape = (self.N, 1))
            self.f = (self.f0 * pm.math.exp(self.s * self.times) /
                      pm.math.sum(self.f0 * pm.math.exp(self.s * self.times),
                      axis = 0) )
            self.f_obs = pm.Poisson("f_obs", mu = 100 * 1000 * self.f,
                                    observed = 100 * 1000 * self.data)

    def find_MAP(self):
        self.map_estimate = pm.find_MAP(model = self.model)
