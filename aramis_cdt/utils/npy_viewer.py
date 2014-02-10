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
from traits.api import \
    HasTraits, Float, Property, cached_property, Int, Array, Bool, \
    Instance, DelegatesTo, Tuple, Button, List, Str, File

from traitsui.api import View, Item, HGroup, EnumEditor, Group, UItem, RangeEditor

from traitsui.ui_editors.array_view_editor import ArrayViewEditor

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


class NpyData(HasTraits):
    data = Array
    traits = View(
                Item('data',
                        show_label=False,
                        editor=ArrayViewEditor(format='%.4f',
                                               )
                    ),
                )

class NpyViewer(HasTraits):
    npy_file = File

    data = Instance(NpyData)

    load_data = Button()

    def _load_data_fired(self):
        self.load_data.data = np.load(self.npy_file)

    traits = View(
                'npy_file',
                UItem('load_data'),
                Item('@data'),
                title='Array Viewer',
                width=0.3,
                height=0.8,
                resizable=True,
                )

if __name__ == '__main__':
    viewer = NpyViewer()
    viewer.configure_traits()
