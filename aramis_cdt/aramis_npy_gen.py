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


class AramisNPyGen(HasTraits):
    aramis_info = Instance(AramisInfo)

    x_idx_min_undeformed = Int
    '''Minimum value of the indices in the first column of the undeformed state.
    '''

    y_idx_min_undeformed = Int
    '''Minimum value of the indices in the second column of the undeformed state.
    '''

    x_idx_max_undeformed = Int
    '''Maximum value of the indices in the first column of the undeformed state.
    '''

    y_idx_max_undeformed = Int
    '''Maximum value of the indices in the second column of the undeformed state.
    '''

    n_x_undeformed = Int
    '''Number of facets in x-direction
    '''

    n_y_undeformed = Int
    '''Number of facets in y-direction
    '''

    generate_npy = Button
    '''Generate npy files from Aramis *.txt data 
    '''
    def _generate_npy(self):
        pass

    view = View(
                UItem('generate_npy'),
                )


if __name__ == '__main__':
    AG = AramisNPyGen()
    AG.configure_traits()
