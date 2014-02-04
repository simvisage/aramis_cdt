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
# from traits.etsconfig.api import ETSConfig
# ETSConfig.toolkit = 'qt4'

from etsproxy.traits.api import \
    HasTraits, Instance, on_trait_change

from etsproxy.traits.ui.api import UItem, View, Tabbed

import os
import sys

import platform
import time
if platform.system() == 'Linux':
    sysclock = time.time
elif platform.system() == 'Windows':
    sysclock = time.clock

from aramis_info import AramisInfo
from aramis_cdt import AramisCDT
from aramis_view2d import AramisPlot2D
from aramis_view3d import AramisView3D
from aramis_remote import AramisRemote


class AramisUI(HasTraits):
    '''This class is managing all the parts of the CDT and enable to create
    simple user interface.
    '''
    aramis_info = Instance(AramisInfo, ())

    aramis_remote = Instance(AramisRemote)
    def _aramis_remote_default(self):
        return AramisRemote(aramis_info=self.aramis_info)

    aramis_cdt = Instance(AramisCDT, ())

    aramis_view2d = Instance(AramisPlot2D)
    def _aramis_view2d_default(self):
        return AramisPlot2D(aramis_info=self.aramis_info,
                            aramis_cdt=self.aramis_cdt)

    aramis_view3d = Instance(AramisView3D)
    def _aramis_view3d_default(self):
        return AramisView3D(aramis_cdt=self.aramis_cdt)

    view = View(
                UItem('aramis_remote'),
                UItem('aramis_info@'),
                Tabbed(
                       UItem('aramis_cdt@'),
                       UItem('aramis_view2d@'),
                       UItem('aramis_view3d@'),
                       ),
                title='Aramis CDT'
                )


if __name__ == '__main__':

    if platform.system() == 'Linux':
        data_dir = r'/media/data/_linux_data/aachen/Aramis_07_2013/TTb-4c-2cm-0-TU-V2_bs4-Xf19s15-Yf19s15'
    elif platform.system() == 'Windows':
        data_dir = r'E:\_linux_data\aachen/Aramis_07_2013/TTb-4c-2cm-0-TU-V2_bs4-Xf19s15-Yf19s15'

    # data_dir = '/media/data/_linux_data/aachen/Aramis_07_2013/TTb-4c-2cm-0-TU-V1_bs4-Xf19s1-Yf5s4'

    AI = AramisInfo(data_dir=data_dir)
    AC = AramisCDT(aramis_info=AI,
                  integ_radius=1,
                  evaluated_step_idx=129,
                  crack_detect_idx=129,
                  transform_data=True)
    AUI = AramisUI(aramis_info=AI,
                    aramis_cdt=AC)
#     AUI = AramisUI()
    AUI.configure_traits()
