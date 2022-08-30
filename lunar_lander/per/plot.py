import numpy as np
from numpy import savetxt, loadtxt
import matplotlib.pyplot as plt
from scipy.integrate import simps
from numpy import trapz

areas = []
n=10
def plot(arrays, labels, path):
    for idx, array in enumerate(arrays):
        mins = np.min(array, axis=0)
        maxs = np.max(array, axis=0)
        mean = np.mean(array, axis=0)
        x = np.arange(len(mean))
        area = trapz(mean)
        areas.append(area)
        plt.plot(x[::n], mean[::n], label=labels[idx])
        plt.fill_between(x[::n], mins[::n], maxs[::n], alpha=0.2)
    plt.xlabel('Episode', fontsize=18)
    plt.ylabel('Reward', fontsize=18)
    plt.legend()
    plt.grid()
    plt.savefig(path)

gravity10 = []
gravity5 = []
g10with5 = []
g5with10 = []
for idx in range(5):
    gravity10.append(np.loadtxt(f'per_gravity_10_5runs/progress_{idx}'))
    gravity5.append(np.loadtxt(f'per_gravity_5_5runs/progress_{idx}'))
    g10with5.append(np.loadtxt(f'per_gravity10_inject5_5runs/progress_{idx}'))
    g5with10.append(np.loadtxt(f'per_gravity5_inject10_5runs/progress_{idx}'))
    # gravity10.append(np.loadtxt(f'per_gravity_10_5runs/avg_reward_{idx}'))
    # gravity5.append(np.loadtxt(f'per_gravity_5_5runs/avg_reward_{idx}'))
    # g10with5.append(np.loadtxt(f'per_gravity10_inject5_5runs/avg_reward_{idx}'))
    # g5with10.append(np.loadtxt(f'per_gravity5_inject10_5runs/avg_reward_{idx}'))

gravity10 = np.array(gravity10)
gravity5 = np.array(gravity5)
g10with5 = np.array(g10with5)
g5with10 = np.array(g5with10)

arrays = [gravity10, g10with5, gravity5, g5with10]
labels = ['gravity-10', 'gravity-10[gravity-5]', 'gravity-5',  'gravity-5[gravity-10]']
plot(arrays, labels, 'lunarlander_per')
# arrays1 = [gravity10, g10with5]
# labels1 = ['gravity-10', 'gravity-5 --> gravity-10']
# plot(arrays1, labels1, 'comparison1')