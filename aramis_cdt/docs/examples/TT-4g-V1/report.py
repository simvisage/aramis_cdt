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

data_dir = os.path.join(aramis_dir, specimen_name + '-Xf19a15-Yf19a15')

AI = AramisInfo(data_dir=data_dir)
AC = AramisCDT(aramis_info=AI,
               n_px_facet_size_x=19,
               n_px_facet_size_y=19,
               n_px_facet_distance_x=15,
               n_px_facet_distance_y=15,
               integ_radius=1,
               evaluated_step=185,
               w_detect_step=185,
               transform_data=True)

AUI = AramisUI(aramis_info=AI,
                aramis_cdt=AC)

report_gen(AUI)

