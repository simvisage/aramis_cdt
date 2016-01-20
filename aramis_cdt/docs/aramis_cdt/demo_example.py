#-------------------------------------------------------------------------
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
x = dic_ui.aramis_data.x_arr_0[10, :]
# end_read