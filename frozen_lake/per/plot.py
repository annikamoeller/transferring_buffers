import numpy as np
from numpy import savetxt, loadtxt
import matplotlib.pyplot as plt
from scipy.integrate import simps
from numpy import trapz

areas = []
def plot(arrays, labels, path):
    for idx, array in enumerate(arrays):
        mins = np.min(array, axis=0)
        maxs = np.max(array, axis=0)
        mean = np.mean(array, axis=0)
        x = np.arange(len(mean))
        area = trapz(mean)
        areas.append(area)
        plt.plot(x, mean, label=labels[idx])
        plt.fill_between(x, mins, maxs, alpha=0.2)
    plt.xlabel('Episode', fontsize=18)
    plt.ylabel('Reward', fontsize=18)
    plt.legend()
    plt.savefig(path)

# noslip4x4_slip4x4 = []
# slip4x4_noslip4x4 = []
# slip4x4 = []
# noslip4x4 = []
# for idx in range(5):
#     noslip4x4_slip4x4.append(np.loadtxt(f'noslip4x4_with_slip4x4/progress_{idx}'))
#     slip4x4_noslip4x4.append(np.loadtxt(f'slip4x4_with_noslip4x4/progress_{idx}'))
#     slip4x4.append(np.loadtxt(f'slip_4x4/progress_{idx}'))
#     noslip4x4.append(np.loadtxt(f'no_slip_4x4/progress_{idx}'))

# slip4x4_noslip4x4 = np.array(slip4x4_noslip4x4)
# slip4x4 = np.array(slip4x4)
# noslip4x4 = np.array(noslip4x4)
# arrays = [slip4x4, slip4x4_noslip4x4, noslip4x4, noslip4x4_slip4x4]
# labels = ['slip', 'slip[no-slip]', 'no_slip', 'no-slip[slip]']

noslip8x8_slip_8x8 = []
slip8x8_noslip8x8 = []
slip8x8 = []
noslip8x8 = []
for idx in range(5):
    noslip8x8_slip_8x8.append(np.loadtxt(f'noslip8x8_with_slip8x8/progress_{idx}'))
    slip8x8_noslip8x8.append(np.loadtxt(f'slip8x8_with_noslip8x8/progress_{idx}'))
    slip8x8.append(np.loadtxt(f'slip_8x8/progress_{idx}'))
    noslip8x8.append(np.loadtxt(f'no_slip_8x8/progress_{idx}'))

noslip8x8_slip_8x8 = np.array(noslip8x8_slip_8x8)
slip8x8_noslip8x8 = np.array(slip8x8_noslip8x8)
slip8x8 = np.array(slip8x8)
noslip8x8 = np.array(noslip8x8)
arrays = [slip8x8, slip8x8_noslip8x8, noslip8x8, noslip8x8_slip_8x8]
labels = ['slip', 'slip[no-slip]', 'no_slip', 'no-slip[slip]']

plot(arrays, labels, 'lake8x8_per')
savetxt('areas', areas)