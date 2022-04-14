import numpy as np
import matplotlib.pyplot as plt
import time
from simulate import *

start = time.time()

N = 100000
num_gens = 49
dt = 0.1
for i in range(100):
    f0 = np.random.random(N)
    s = np.random.random(N) / 10

    times = np.array([7, 14, 28, 42, 49])

    traj = create_trajectories(f0, s, times)

print("Time taken: {}s".format(round(time.time() - start)))

plt.figure()
plt.plot(times, traj[0:100, :].T)
plt.show()

# tps = [7, 14, 28, 42, 49]
# samples = np.zeros([5, N])
#
# for i, tp in enumerate(tps):
#     j = np.where(time_array == tp)[0][0]
#     samples[i, :] = sample_lineages(traj[j, :], int(1e7))
#
# plt.figure()
# plt.semilogy(tps, samples[:, 0:100])
# plt.show()
