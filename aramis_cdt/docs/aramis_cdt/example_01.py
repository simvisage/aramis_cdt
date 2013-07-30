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

from etsproxy.traits.api import \
    HasTraits, Instance

from etsproxy.traits.ui.api import UItem, View

import os

import numpy as np

import platform
import time
if platform.system() == 'Linux':
    sysclock = time.time
elif platform.system() == 'Windows':
    sysclock = time.clock

from aramis_cdt.aramis_info import AramisInfo
from aramis_cdt.aramis_cdt import AramisCDT
from aramis_cdt.aramis_view2d import AramisView2D
from aramis_cdt.aramis_view3d import AramisView3D
from aramis_cdt.aramis_ui import AramisUI

from aramis_cdt.report_gen import report_gen

aramis_dir = '/media/data/_linux_data/aachen/Aramis_07_2013/'

specimen_name = os.path.split(os.getcwd())[-1]

data_dir = os.path.join(aramis_dir, 'TT-4c-V1-Xf19a15-Yf19a15')

AI = AramisInfo(data_dir=data_dir)
AC = AramisCDT(aramis_info=AI,
               n_px_facet_size_x=19,
               n_px_facet_size_y=19,
               n_px_facet_distance_x=15,
               n_px_facet_distance_y=15,
               integ_radius=1,
               evaluated_step=203,
               w_detect_step=203,
               transform_data=True)

AUI = AramisUI(aramis_info=AI,
                aramis_cdt=AC)

AUI.aramis_view2d.save_plot = False
AUI.aramis_view2d.show_plot = True

AUI.aramis_view2d.plot_crack_filter_crack_avg = True

x = AUI.aramis_cdt.x_arr[10, :]
ux = AUI.aramis_cdt.ux_arr[10, :]
d_ux = AUI.aramis_cdt.d_ux_arr[10, :]
dd_ux = AUI.aramis_cdt.dd_ux_arr[10, :]
ddd_ux = AUI.aramis_cdt.ddd_ux_arr[10, :]

import matplotlib.pyplot as plt
plt.subplot(611)
plt.plot(x, ux)
plt.subplot(612)
plt.plot(x, d_ux)
plt.subplot(613)
plt.plot(x, dd_ux)
plt.subplot(614)
plt.plot(x, ddd_ux)
plt.subplot(615)
y = dd_ux[1:] * dd_ux[:-1]
plt.plot((x[1:] + x[:-1]) / 2.0, y)
plt.plot(((x[1:] + x[:-1]) / 2.0)[y < 0], y[y < 0], 'ro')
plt.subplot(616)
y = (ddd_ux[1:] + ddd_ux[:-1]) / 2.0
plt.plot(((x[1:] + x[:-1]) / 2.0), y)
plt.plot(((x[1:] + x[:-1]) / 2.0)[y < -0.01], y[y < -0.01], 'ro')
plt.show()




#===============================================================================
#
#===============================================================================

data_dir = os.path.join(aramis_dir, 'TT-4c-V1-Xf19a1-Yf19a4')

AI = AramisInfo(data_dir=data_dir)
AC = AramisCDT(aramis_info=AI,
               n_px_facet_size_x=19,
               n_px_facet_size_y=19,
               n_px_facet_distance_x=15,
               n_px_facet_distance_y=15,
               integ_radius=4,
               evaluated_step=203,
               w_detect_step=203,
               transform_data=True)

AUI = AramisUI(aramis_info=AI,
                aramis_cdt=AC)

AUI.aramis_view2d.save_plot = False
AUI.aramis_view2d.show_plot = True

AUI.aramis_view2d.plot_crack_filter_crack_avg = True

x = AUI.aramis_cdt.x_arr[40, :]
ux = AUI.aramis_cdt.ux_arr[40, :]
d_ux = AUI.aramis_cdt.d_ux_arr[40, :]
dd_ux = AUI.aramis_cdt.dd_ux_arr[40, :]
ddd_ux = AUI.aramis_cdt.ddd_ux_arr[40, :]

import matplotlib.pyplot as plt
plt.subplot(611)
plt.plot(x, ux)
plt.subplot(612)
plt.plot(x, d_ux)
plt.subplot(613)
plt.plot(x, dd_ux)
plt.subplot(614)
plt.plot(x, ddd_ux)
plt.subplot(615)
y = dd_ux[1:] * dd_ux[:-1]
plt.plot((x[1:] + x[:-1]) / 2.0, y)
plt.plot((x[1:] + x[:-1]) / 2.0, np.zeros_like(y))
# plt.plot(((x[1:] + x[:-1]) / 2.0)[y < 0], y[y < 0], 'rx')
plt.subplot(616)
y = (ddd_ux[1:] + ddd_ux[:-1]) / 2.0
plt.plot(((x[1:] + x[:-1]) / 2.0), y)
plt.plot((x[1:] + x[:-1]) / 2.0, np.zeros_like(y) - 0.01)
# plt.plot(((x[1:] + x[:-1]) / 2.0)[y < -0.01], y[y < -0.01], 'rx')
plt.show()
