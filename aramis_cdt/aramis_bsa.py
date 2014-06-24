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
    HasTraits, Float, Property, cached_property, Int, Array, Bool, \
    Instance, DelegatesTo, Tuple, Button, List, Str, Event, on_trait_change, WeakRef

from etsproxy.traits.ui.api import View, Item, HGroup, EnumEditor, Group, UItem, RangeEditor

import numpy as np
from scipy import stats
import os
import re

import platform
import time
if platform.system() == 'Linux':
    sysclock = time.time
elif platform.system() == 'Windows':
    sysclock = time.clock

from aramis_info import AramisInfo
from aramis_data import AramisData
from aramis_cdt import AramisCDT

def get_d(u_arr, integ_radius):
    ir = integ_radius
    du_arr = np.zeros_like(u_arr)
    du_arr[:, ir:-ir] = (u_arr[:, 2 * ir:] - u_arr[:, :-2 * ir])
    return du_arr


class AramisBSA(AramisCDT):
    '''Crack Detection Tool for detection of cracks, etc. from Aramis data.
    '''

    d_ux_arr2 = Property(Array, depends_on='aramis_info_changed, aramis_data.+params_changed')
    '''The first derivative of displacement in x-direction
    '''
    @cached_property
    def _get_d_ux_arr2(self):
        x_und = self.aramis_data.x_arr_undeformed
        ux = self.aramis_data.ux_arr
        du_arr = np.zeros_like(x_und)
        ir = self.integ_radius
        du_arr[:, ir:-ir] = (ux[:, 2 * ir:] - ux[:, :-2 * ir]) / (x_und[:, 2 * ir:] - x_und[:, :-2 * ir])
        return du_arr


if __name__ == '__main__':
    from os.path import expanduser
    home = expanduser("~")

    data_dir = os.path.join(home, '.simdb_cache', 'exdata/bending_tests',
                             'three_point', '2013-07-09_BT-6c-2cm-0-TU_bs4-Aramis3d',
                             'aramis', 'BT-6c-V4-bs4-Xf19s15-Yf19s15')

    AI = AramisInfo(data_dir=data_dir)
    # AI = AramisInfo()
    AD = AramisData(aramis_info=AI,
                    evaluated_step_idx=10)
    AC = AramisBSA(aramis_info=AI,
                   aramis_data=AD,
                   integ_radius=10)

    for step in [225]:
        AD.evaluated_step_idx = step
        mid_idx = AC.d_ux_arr2.shape[1] / 2
        x = np.mean(AC.d_ux_arr2[:, mid_idx - 4:mid_idx + 4], axis=1)
        # x = AC.d_ux_arr2[:, mid_idx - 4:mid_idx + 4]

        y = AD.y_arr_undeformed[:, mid_idx]
        import matplotlib.pyplot as plt
        plt.plot(x, y)

    plt.plot([-0.004, 0.014], [0, 0], color='black')
    plt.plot([0, 0], [-10, 10], color='black')
    plt.show()
    # AC.configure_traits()
