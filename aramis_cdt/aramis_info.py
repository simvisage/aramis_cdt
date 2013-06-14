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

from etsproxy.traits.api import HasTraits, Int, Directory, Str, List, on_trait_change

from etsproxy.traits.ui.api import View, Item

import numpy as np

from matresdev.db.simdb import \
    SimDB

import os.path
import re

import platform
import time
if platform.system() == 'Linux':
    sysclock = time.time
elif platform.system() == 'Windows':
    sysclock = time.clock

simdb = SimDB()


class AramisInfo(HasTraits):
    '''Basic informations of Aramis database
    '''

    data_dir = Directory
    '''Directory containing *.txt data files exported by Aramis Software.
    '''
    def _data_dir_default(self):
        return os.path.join(simdb.exdata_dir, 'tensile_tests',
                            'dog_bone', '2012-04-12_TT-12c-6cm-0-TU_SH4', 'ARAMIS',
                            'Probe-1-Ausschnitt-Xf15a1-Yf5a4')

    file_list = List(Str)
    '''List of names of *.txt data files contained in the directory. 
    '''

    step_list = List(Int)
    '''List of available steps.
    '''
    @on_trait_change('data_dir')
    def _file_step_list_update(self):
        '''Update values of file_list and step_list. Sorted by the step number.
        '''
        file_list = [v for v in os.listdir(self.data_dir) if os.path.splitext(v)[1] == ".txt"]
        step_list = []
        pat = r'.*-(?P<step>\d+).txt'
        for f in file_list:
            m = re.match(pat, f)
            step_list.append(int(m.groupdict()['step']))
        # sort filenames by the step number
        file_arr = np.array(file_list, dtype=str)
        idx = np.argsort(step_list)
        step_list.sort()

        self.file_list = file_arr[idx].tolist()
        self.step_list = step_list

    number_of_steps = Int
    '''Number of time steps
    '''
    @on_trait_change('data_dir')
    def _number_of_steps_update(self):
        self.number_of_steps = len(self.step_list)

    basename = Str
    '''String with the base part of the file name "basename<step>.txt".
    '''
    @on_trait_change('file_list')
    def _basename_update(self):
        pat = r'(?P<basename>.*-)(?P<step>\d+).txt'
        m = re.match(pat, self.file_list[0])
        self.basename = m.groupdict()['basename']

    first_step = Int
    '''Number of the first step.
    '''
    @on_trait_change('step_list')
    def _first_step_update(self):
        self.first_step = np.min(self.step_list)

    last_step = Int
    '''Number of the last step.
    '''
    @on_trait_change('step_list')
    def _last_step_update(self):
        self.last_step = np.max(self.step_list)

    # step_selected = Int(auto_set=False, enter_set=True)

    view = View(
                Item('data_dir'),
                Item('basename', style='readonly'),
                Item('first_step', style='readonly'),
                Item('last_step', style='readonly'),
                Item('number_of_steps', style='readonly'),
#                 HGroup(
#                        Item('step_selected', editor=EnumEditor(name='step_list'),
#                             springy=True),
#                        Item('step_selected', style='text', show_label=False,
#                             springy=True),
#                        ),
                resizable=True
                )


if __name__ == '__main__':
    if platform.system() == 'Linux':
        aramis_dir = os.path.join(r'/media/data/_linux_data/aachen/ARAMIS_data_IMB/01_ARAMIS_Daten/')
    elif platform.system() == 'Windows':
        aramis_dir = os.path.join(r'E:\_linux_data\aachen\ARAMIS_data_IMB\01_ARAMIS_Daten')
    data_dir = os.path.join(aramis_dir, 'Probe-1-Ausschnitt-Xf15a1-Yf5a4')

    aramis_info = AramisInfo(data_dir=data_dir)
    aramis_info.configure_traits()


