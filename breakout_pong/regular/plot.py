from pickletools import read_uint4
import numpy as np
from numpy import savetxt, loadtxt
import matplotlib.pyplot as plt
from scipy.integrate import simps
from numpy import trapz

areas = []
n=1000
def plot(arrays, labels, path):
    for idx, array in enumerate(arrays):
        mins = np.min(array, axis=0)
        maxs = np.max(array, axis=0)
        mean = np.mean(array, axis=0)
        x = np.arange(len(mean))
        area = trapz(mean)
        areas.append(area)
        print(areas)
        plt.plot(x[::n], mean[::n], label=labels[idx])
        plt.fill_between(x[::n], mins[::n], maxs[::n], alpha=0.2)
    plt.xlabel('Episode', fontsize=18)
    plt.ylabel('Reward', fontsize=18)
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(path)

min_bs = 10000000
min_bt = 10000000
min_ps = 10000000
min_pt = 10000000
for idx in range(1,6):
    arr_len_bs = len(np.loadtxt(f'progress_breakout/progress_{idx}'))
    if (arr_len_bs < min_bs): min_bs = arr_len_bs 

    arr_len_bt = len(np.loadtxt(f'progress_breakout_transfer/progress_{idx}'))
    if (arr_len_bt < min_bt): min_bt = arr_len_bt

    arr_len_ps = len(np.loadtxt(f'progress_pong/progress_{idx}'))
    if (arr_len_ps < min_ps): min_ps = arr_len_ps

    arr_len_pt = len(np.loadtxt(f'progress_pong_transfer/progress_{idx}'))
    if (arr_len_pt < min_pt): min_pt = arr_len_pt

print(min_bs, min_bt, min_ps, min_pt)

breakout_standard = []
breakout_transfer = []
pong_standard = []
pong_transfer = []

for idx in range(1,6):
    breakout_standard.append(np.loadtxt(f'progress_breakout/progress_{idx}')[0:min_bs])
    breakout_transfer.append(np.loadtxt(f'progress_breakout_transfer/progress_{idx}')[0:min_bt])
    pong_standard.append(np.loadtxt(f'progress_pong/progress_{idx}')[0:min_ps])
    pong_transfer.append(np.loadtxt(f'progress_pong_transfer/progress_{idx}')[0:min_pt])

breakout_standard = np.array(breakout_standard)
breakout_transfer = np.array(breakout_transfer)
pong_standard = np.array(pong_standard)
pong_transfer = np.array(pong_transfer)
print(breakout_standard.shape, breakout_transfer.shape, pong_standard.shape, pong_transfer.shape)


arrays = [breakout_standard, breakout_transfer]
arrays2 = [pong_standard, pong_transfer]
labels = ['breakout', 'breakout[pong]']
labels2 = ['pong', 'pong[breakout]']
plot(arrays, labels, 'breakout_plot')
#plot(arrays2, labels2, 'pong_plot')
