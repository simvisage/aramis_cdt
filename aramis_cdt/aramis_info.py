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

from etsproxy.traits.api import HasTraits, Int, Directory, Str, List, \
                                on_trait_change, Dict

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
    '''Directory of data files (*.txt) exported by Aramis Software.
    '''

    npy_dir = Directory
    @on_trait_change('data_dir')
    def _npy_dir_update(self):
        self.npy_dir = os.path.join(self.data_dir, 'npy/')

#     file_list = List(Str)
#     '''List of names of *.txt data files contained in the directory.
#     '''

    step_list = List(Int)
    '''List of available steps.
    '''
    @on_trait_change('data_dir')
    def _file_step_list_update(self):
        '''Update values of file_list and step_list. Sorted by the step number.
        '''
        if os.path.exists(self.npy_dir):
            file_list = [v for v in os.listdir(self.npy_dir) if os.path.splitext(v)[1] == ".npy"]
        else:
            file_list = [v for v in os.listdir(self.data_dir) if os.path.splitext(v)[1] == ".txt"]
        step_list = []
        pat = r'displ.*-(?P<step>\d+).*'
        for f in file_list:
            m = re.match(pat, f)
            if m:
                step_list.append(int(m.groupdict()['step']))
        # sort filenames by the step number
        # file_arr = np.array(file_list, dtype=str)
        # idx = np.argsort(step_list)
        step_list.sort()

        # self.file_list = file_arr[idx].tolist()
        self.step_list = step_list

    number_of_steps = Int
    '''Number of time steps
    '''
    @on_trait_change('step_list')
    def _number_of_steps_update(self):
        self.number_of_steps = len(self.step_list)

    step_idx_list = List(Int)
    '''List of step indexes
    '''
    @on_trait_change('number_of_steps')
    def _step_idx_list_update(self):
        self.step_idx_list = np.arange(self.number_of_steps, dtype=int).tolist()

    specimen_name = Str
    '''Specimen name obtained from the folder name
    '''
    @on_trait_change('data_dir')
    def _specimen_name_update(self):
        self.specimen_name = os.path.split(self.data_dir)[-1]

    undeformed_coords_filename = Str('undeformed_coords-Stage-0-0')

    displacements_basename = Str('displ-Stage-0-')

    facet_params_dict = Dict
    '''Facet parameters obtained by decompilation of the specimen_name
    '''
    @on_trait_change('specimen_name')
    def _facet_params_dict_update(self):
        pat = r'.*Xf(?P<fsz_x>\d+)s(?P<fst_x>\d+)-Yf(?P<fsz_y>\d+)s(?P<fst_y>\d+).*'
        m = re.match(pat, self.specimen_name)
        self.facet_params_dict = m.groupdict()

    n_px_facet_size_x = Int
    '''Size of a facet in x-direction defined in numbers of pixels
    '''
    @on_trait_change('facet_params_dict')
    def _n_px_facet_size_x_update(self):
        self.n_px_facet_size_x = int(self.facet_params_dict['fsz_x'])

    n_px_facet_step_x = Int
    '''Distance between the mid-points of two facets in x-direction
    '''
    @on_trait_change('facet_params_dict')
    def _n_px_facet_step_x_update(self):
        self.n_px_facet_step_x = int(self.facet_params_dict['fst_x'])

    n_px_facet_size_y = Int
    '''Size of a facet in y-direction defined in numbers of pixels
    '''
    @on_trait_change('facet_params_dict')
    def _n_px_facet_size_y_update(self):
        self.n_px_facet_size_y = int(self.facet_params_dict['fsz_y'])

    n_px_facet_step_y = Int
    '''Distance between the mid-points of two facets in y-direction
    '''
    @on_trait_change('facet_params_dict')
    def _n_px_facet_step_y_update(self):
        self.n_px_facet_step_y = int(self.facet_params_dict['fst_y'])

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

    view = View(
                Item('data_dir'),
                Item('specimen_name', style='readonly'),
                Item('first_step', style='readonly'),
                Item('last_step', style='readonly'),
                Item('number_of_steps', style='readonly'),
                Item('n_px_facet_size_x', style='readonly'),
                Item('n_px_facet_size_y', style='readonly'),
                Item('n_px_facet_step_x', style='readonly'),
                Item('n_px_facet_step_y', style='readonly'),
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
        data_dir = r'/media/data/_linux_data/aachen/Aramis_07_2013/TTb-4c-2cm-0-TU-V2_bs4-Xf19s15-Yf19s15'
    elif platform.system() == 'Windows':
        data_dir = r'E:\_linux_data\aachen/Aramis_07_2013/TTb-4c-2cm-0-TU-V2_bs4-Xf19s15-Yf19s15'

    AI = AramisInfo(data_dir=data_dir)
    print AI.data_dir
    print AI.specimen_name
    print AI.facet_params_dict
    print AI.step_list
    AI.configure_traits()


