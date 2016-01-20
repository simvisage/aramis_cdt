#-------------------------------------------------------------------------
#
# Copyright (c) 2013
# IMB, RWTH Adic_cdthen University,
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
#-------------------------------------------------------------------------

# begin_read
import os
from aramis_cdt.api import \
    AramisCDT, AramisInfo, AramisUI

aramis_dir = '.'
specimen_name = os.path.split(os.getcwd())[-1]
data_dir = os.path.join(aramis_dir, 'sample_data-Xf19s15-Yf19s15')
dic_info = AramisInfo(data_dir=data_dir)
dic_cdt = AramisCDT(aramis_info=dic_info,
                    integ_radius=1,
                    crack_detection_step=0)
dic_ui = AramisUI(aramis_info=dic_info)
# end_read

# begin_fields
x = dic_ui.aramis_data.x_arr_0[10, :]
ux = dic_ui.aramis_data.ux_arr[10, :]
d_ux = dic_ui.aramis_data.d_ux[10, :]
dd_ux = dic_ui.aramis_data.dd_ux[10, :]
ddd_ux = dic_ui.aramis_data.ddd_ux[10, :]
# end_fields

import matplotlib.pyplot as plt
import numpy as np

plt.subplot(611)
plt.plot(x, ux)
plt.locator_params(axis='y', nbins=3)

plt.subplot(612)
plt.plot(x, d_ux)
plt.plot(x, np.convolve(d_ux, np.ones(10) / 10., 'same'))
plt.plot(x, np.convolve(d_ux, np.ones(3) / 3., 'same'))
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


#=========================================================================
#
#=========================================================================

dic_ui.aramis_cdt.integ_radius = 4

x = dic_ui.aramis_data.x_arr_0[40, :]
ux = dic_ui.aramis_data.ux_arr[40, :]
d_ux = dic_ui.aramis_data.d_ux[40, :]
dd_ux = dic_ui.aramis_data.dd_ux[40, :]
ddd_ux = dic_ui.aramis_data.ddd_ux[40, :]

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
