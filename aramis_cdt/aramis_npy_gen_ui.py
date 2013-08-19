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

from etsproxy.traits.api import \
    HasTraits, Float, Property, cached_property, Int, Array, Bool, Directory, \
    Instance, DelegatesTo, Tuple, Button, List, Str, File, Enum, on_trait_change

from etsproxy.traits.ui.api import View, Item, HGroup, EnumEditor, Group, UItem, RangeEditor

import numpy as np
from scipy import stats
import os
import re
import ConfigParser

from matresdev.db.simdb import SimDB
from sftp_server import SFTPServer
import zipfile

import platform
import time
if platform.system() == 'Linux':
    sysclock = time.time
elif platform.system() == 'Windows':
    sysclock = time.clock

from aramis_info import AramisInfo
from aramis_npy_gen import AramisNPyGen

class AramisNPyGenUI(HasTraits):
    '''This class enables to create simple user interface for *.npy files 
    generation tool.
    '''

    aramis_info = Instance(AramisInfo)

    aramis_npy_gen = Instance(AramisNPyGen)
    def _aramis_npy_gen_default(self):
        return AramisNPyGen(aramis_info=self.aramis_info)

    view = View(
                UItem('aramis_info@'),
                UItem('aramis_npy_gen@'),
                )


if __name__ == '__main__':
    data_dir = '/media/data/_linux_data/aachen/Aramis_07_2013/TT-12c-6cm-0-TU-SH4-V1-Xf19s15-Yf19s15'
    AI = AramisInfo(data_dir=data_dir)
    AGUI = AramisNPyGenUI(aramis_info=AI)
    AGUI.configure_traits()
