#-------------------------------------------------------------------------------
#
# Copyright (c) 2013
# IMB, RWTH Aachen University,
# ISM, Brno University of Technology
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in the AramisCDT top directory "licence.txt" and may be
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
from aramis_cdt.aramis_view2d import AramisPlot2D
from aramis_cdt.aramis_view3d import AramisView3D
from aramis_cdt.aramis_ui import AramisUI
from aramis_cdt.aramis_remote import AramisRemote

from aramis_cdt.report_gen import report_gen
from matresdev.db.simdb import SimDB

simdb = SimDB()

experiment_dir = os.path.join(simdb.exdata_dir, 'tensile_tests', 'buttstrap_clamping',
                              '2013-07-09_TTb-4c-2cm-0-TU_bs4-Aramis3d')

AI = AramisInfo()
AR = AramisRemote(aramis_info=AI,
                  experiment_dir=experiment_dir)
AC = AramisCDT(aramis_info=AI,
               integ_radius=1,
               evaluated_step_idx=203,
               crack_detect_idx=203,
               transform_data=True)

AUI = AramisUI(aramis_info=AI,
                aramis_cdt=AC)

report_gen(AUI)

