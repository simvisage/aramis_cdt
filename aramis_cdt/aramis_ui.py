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

import platform
import time
if platform.system() == 'Linux':
    sysclock = time.time
elif platform.system() == 'Windows':
    sysclock = time.clock

from aramis_info import AramisInfo
from aramis_cdt import AramisCDT
from aramis_view2d import AramisView2D
from aramis_view3d import AramisView3D

class AramisView(HasTraits):
    '''This class is managing all the parts of the CDT and enable to create
    simple user interface.
    '''
    aramis_info = Instance(AramisInfo)

    aramis_cdt = Instance(AramisCDT)

    aramis_view2d = Instance(AramisView2D)
    def _aramis_view2d_default(self):
        return AramisView2D(aramis_info=self.aramis_info,
                            aramis_cdt=self.aramis_cdt)

    aramis_view3d = Instance(AramisView3D)
    def _aramis_view3d_default(self):
        return AramisView3D(aramis_cdt=self.aramis_cdt)

    view = View(
                UItem('aramis_info@'),
                UItem('aramis_cdt@'),
                UItem('aramis_view2d@'),
                UItem('aramis_view3d@'),
                title='Aramis CDT'
                )


if __name__ == '__main__':
    if platform.system() == 'Linux':
        aramis_dir = os.path.join(r'/media/data/_linux_data/aachen/ARAMIS_data_IMB/01_ARAMIS_Daten/')
    elif platform.system() == 'Windows':
        aramis_dir = os.path.join(r'E:\_linux_data\aachen\ARAMIS_data_IMB\01_ARAMIS_Daten')

    data_dir = os.path.join(aramis_dir, 'Probe-1-Ausschnitt-Xf15a1-Yf5a4')
    AI = AramisInfo(data_dir=data_dir)
    AC = AramisCDT(aramis_info=AI,
                  evaluated_step=400)
    AV = AramisView(aramis_info=AI,
                    aramis_cdt=AC)
    AV.configure_traits()
