# Prepare data for adding to Aramis project

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# load data from file containing measured forces, etc.
filename = 'TTb-4c-2cm-0-TU-V1_bs4.ASC'
data = np.loadtxt(filename, delimiter=';')

# values from Aramis project
number_of_stages = 214
delay = 2.

# set time to start 0
time = data[:, 0] - data[0, 0]
force = data[:, 1]

# linear interpolation for all stages
stage_time = np.arange(0, number_of_stages * delay, delay)
load_time = interp1d(time, force)
load = load_time(stage_time)

# save data of forces as txt file divided by 100. (compatibility with previous data)
np.savetxt('AD_data.txt', load / 100., fmt='%.6e')

print 'peak value ', np.max(load)
print 'peak value at stage nb. ', np.argwhere(load == np.max(load))

# plot load-time curve
plt.figure()
plt.plot(time, force)
plt.xlabel('time [s]')
plt.ylabel('force [kN]')

plt.show()
