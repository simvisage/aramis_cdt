#-------------------------------------------------------------------------------
#
# Copyright (c) 2012
# IMB, RWTH Aachen University,
# ISM, Brno University of Technology
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in the Spirrid top directory "licence.txt" and may be
# redistributed only under the conditions described in the aforementioned
# license.
#
# Thanks for using Simvisage open source!
#
#-------------------------------------------------------------------------------

import os
import platform
import numpy as np


def unique_rows(a):
    unique_a = np.unique(a.view([('', a.dtype)] * a.shape[1]))
    print unique_a
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

a = np.array([[1, 2], [3, 4], [1, 2]])
print a.view([('', a.dtype)] * a.shape[1])
print unique_rows(a)

quit()

import time
if platform.system() == 'Linux':
    sysclock = time.time
elif platform.system() == 'Windows':
    sysclock = time.clock

if platform.system() == 'Linux':
    aramis_dir = os.path.join(r'/media/data/_linux_data/aachen/ARAMIS_data_IMB/01_ARAMIS_Daten/')
elif platform.system() == 'Windows':
    aramis_dir = os.path.join(r'E:\_linux_data\aachen\ARAMIS_data_IMB\01_ARAMIS_Daten')

data_dir = os.path.join(aramis_dir, 'Probe-1-Ausschnitt-Xf15a1-Yf5a4', 'P1-s458-Xf15a1-Yf5a4-Ausschnitt-Stufe-0-400.txt')
data_dir_npy = os.path.join(aramis_dir, 'Probe-1-Ausschnitt-Xf15a1-Yf5a4', 'npy', 'P1-s458-Xf15a1-Yf5a4-Ausschnitt-Stufe-0-400.npy')

t1, t2, t3, = 0, 0, 0
for i in range(10):
    start = sysclock()
    data = np.loadtxt(data_dir, skiprows=14, usecols=[0, 1, 2, 3, 4, 8, 9, 10])
    t1 += sysclock() - start
    print data

    start = sysclock()
    data = np.load(data_dir_npy)
    t2 += sysclock() - start
    print data

    start = sysclock()
    data = np.load(data_dir_npy, mmap_mode='r')
    t3 += sysclock() - start
    print data

print t1 / 10., t2 / 10., t3 / 10.






