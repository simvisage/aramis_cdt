#-------------------------------------------------------------------------------
#
# Copyright (c) 2013
# IMB, RWTH Aachen University,
# ISM, Brno University of Technology
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in the AramisCDT top directory "license.txt" and may be
# redistributed only under the conditions described in the aforementioned
# license.
#
# Thanks for using Simvisage open source!
#
#-------------------------------------------------------------------------------

import os

import numpy as np

import platform
import time
if platform.system() == 'Linux':
    sysclock = time.time
elif platform.system() == 'Windows':
    sysclock = time.clock

import matplotlib.pyplot as plt

from aramis_cdt.aramis_info import AramisInfo
from aramis_cdt.aramis_cdt import AramisCDT
from aramis_cdt.aramis_ui import AramisUI

aramis_dir = '.'

specimen_name = os.path.split(os.getcwd())[-1]

data_dir = os.path.join(aramis_dir, 'sample_data-Xf19s15-Yf19s15')

AI = AramisInfo(data_dir=data_dir)

AC = AramisCDT(aramis_info=AI,
               integ_radius=1,
               crack_detect_idx=0)

AUI = AramisUI(aramis_info=AI)


x = AUI.aramis_data.x_arr_undeformed[10, :]
ux = AUI.aramis_data.ux_arr[10, :]
d_ux = AUI.aramis_cdt.d_ux_arr[10, :]
dd_ux = AUI.aramis_cdt.dd_ux_arr[10, :]
ddd_ux = AUI.aramis_cdt.ddd_ux_arr[10, :]

plt.subplot(611)
plt.plot(x, ux)
plt.locator_params(axis='y', nbins=3)

plt.subplot(612)
plt.plot(x, d_ux)
plt.locator_params(axis='y', nbins=3)

plt.subplot(613)
plt.plot(x, dd_ux)
plt.locator_params(axis='y', nbins=3)

plt.subplot(614)
plt.plot(x, ddd_ux)
plt.locator_params(axis='y', nbins=3)

plt.subplot(615)
y = dd_ux[1:] * dd_ux[:-1]
plt.plot((x[1:] + x[:-1]) / 2.0, y)
plt.plot(((x[1:] + x[:-1]) / 2.0)[y < 0], y[y < 0], 'ro')
plt.locator_params(axis='y', nbins=3)

plt.subplot(616)
y = (ddd_ux[1:] + ddd_ux[:-1]) / 2.0
plt.plot(((x[1:] + x[:-1]) / 2.0), y)
plt.plot(((x[1:] + x[:-1]) / 2.0)[y < -0.005], y[y < -0.005], 'ro')
plt.locator_params(axis='y', nbins=3)

plt.tight_layout()
plt.show()




#===============================================================================
#
#===============================================================================

AUI.aramis_cdt.integ_radius = 4

x = AUI.aramis_data.x_arr_undeformed[40, :]
ux = AUI.aramis_data.ux_arr[40, :]
d_ux = AUI.aramis_cdt.d_ux_arr[40, :]
dd_ux = AUI.aramis_cdt.dd_ux_arr[40, :]
ddd_ux = AUI.aramis_cdt.ddd_ux_arr[40, :]

plt.subplot(611)
plt.plot(x, ux)
plt.locator_params(axis='y', nbins=3)

plt.subplot(612)
plt.plot(x, d_ux)
plt.locator_params(axis='y', nbins=3)

plt.subplot(613)
plt.plot(x, dd_ux)
plt.locator_params(axis='y', nbins=3)

plt.subplot(614)
plt.plot(x, ddd_ux)
plt.locator_params(axis='y', nbins=3)

plt.subplot(615)
y = dd_ux[1:] * dd_ux[:-1]
plt.plot((x[1:] + x[:-1]) / 2.0, y)
plt.plot((x[1:] + x[:-1]) / 2.0, np.zeros_like(y))
# plt.plot(((x[1:] + x[:-1]) / 2.0)[y < 0], y[y < 0], 'rx')
plt.locator_params(axis='y', nbins=3)

plt.subplot(616)
y = (ddd_ux[1:] + ddd_ux[:-1]) / 2.0
plt.plot(((x[1:] + x[:-1]) / 2.0), y)
plt.plot((x[1:] + x[:-1]) / 2.0, np.zeros_like(y) - 0.01)
# plt.plot(((x[1:] + x[:-1]) / 2.0)[y < -0.01], y[y < -0.01], 'rx')
plt.locator_params(axis='y', nbins=3)
plt.tight_layout()

plt.show()
