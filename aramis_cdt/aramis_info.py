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

from traits.api import \
    HasTraits, Int, Directory, Str, List, \
    on_trait_change, Dict, Property, cached_property

from traitsui.api import \
    View, Item

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
    '''Basic information of Aramis database obtained from directory name and
    from files placed in the data directory.
    '''
    data_dir = Directory(auto_set=False, enter_set=True)
    '''Directory containing experiment data files exported by Aramis Software.
    '''

    npy_dir = Property(Directory, depends_on='data_dir')
    '''Directory containing data files converted to .npy. 
    '''
    @cached_property
    def _get_npy_dir(self):
        return os.path.join(self.data_dir, 'npy/')

    aramis_stage_list = Property(List(Int), depends_on='data_dir')
    '''List of stage numbers obtained from filenames of Aramis files
    '''
    @cached_property
    def _get_aramis_stage_list(self):
        try:
            if os.path.exists(self.npy_dir):
                file_list = [v for v in os.listdir(self.npy_dir) if os.path.splitext(v)[1] == ".npy"]
            else:
                file_list = [v for v in os.listdir(self.data_dir) if os.path.splitext(v)[1] == ".txt"]
            aramis_stage_list = []
            pat = r'displ.*-(?P<step>\d+).*'
            for f in file_list:
                m = re.match(pat, f)
                if m:
                    aramis_stage_list.append(int(m.groupdict()['step']))
            # sort filenames by the step number
            # file_arr = np.array(file_list, dtype=str)
            # idx = np.argsort(aramis_stage_list)
            aramis_stage_list.sort()

            # self.file_list = file_arr[idx].tolist()
            return aramis_stage_list
        except:
            return []

    number_of_steps = Property(Int, depends_on='data_dir')
    '''Number of steps - can differ from max value in aramis_stage_list
    '''
    @cached_property
    def _get_number_of_steps(self):
        return len(self.aramis_stage_list)

    step_list = Property(List(Int), depends_on='data_dir')
    '''List of step numbers
    '''
    @cached_property
    def _get_step_list(self):
        return np.arange(self.number_of_steps, dtype=int).tolist()

    specimen_name = Property(Str, depends_on='data_dir')
    '''Specimen name obtained from the folder name
    '''
    @cached_property
    def _get_specimen_name(self):
        return os.path.split(self.data_dir)[-1]

    undeformed_coords_filename = Str('undeformed_coords-Stage-0-0')
    '''Name of the file containing coordinates in initial state
    '''

    displacements_basename = Str('displ-Stage-0-')
    '''Name of the file containing displacement of facets
    '''

    facet_params_dict = Property(Dict, depends_on='data_dir')
    '''Dictionary containing facet parameters obtained by decompilation of 
    the specimen_name. 
    '''
    @cached_property
    def _get_facet_params_dict(self):
        pat = r'.*Xf(?P<fsz_x>\d+)s(?P<fst_x>\d+)-Yf(?P<fsz_y>\d+)s(?P<fst_y>\d+).*'
        m = re.match(pat, self.specimen_name)
        return m.groupdict()

    n_px_facet_size_x = Property(Int, depends_on='data_dir')
    '''Size of a facet in x-direction [pixel]
    '''
    @cached_property
    def _get_n_px_facet_size_x(self):
        return int(self.facet_params_dict['fsz_x'])

    n_px_facet_step_x = Property(Int, depends_on='data_dir')
    '''Distance between the mid-points of two facets in x-direction [pixel]
    '''
    @cached_property
    def _get_n_px_facet_step_x(self):
        return int(self.facet_params_dict['fst_x'])

    n_px_facet_size_y = Property(Int, depends_on='data_dir')
    '''Size of a facet in y-direction [pixel]
    '''
    @cached_property
    def _get_n_px_facet_size_y(self):
        return int(self.facet_params_dict['fsz_y'])

    n_px_facet_step_y = Property(Int, depends_on='data_dir')
    '''Distance between the mid-points of two facets in y-direction [pixel]
    '''
    @cached_property
    def _get_n_px_facet_step_y(self):
        return int(self.facet_params_dict['fst_y'])



    view = View(
                # Item('data_dir'),
                Item('specimen_name', style='readonly'),
                # Item('number_of_steps', style='readonly'),
                # Item('n_px_facet_size_x', style='readonly'),
                # Item('n_px_facet_size_y', style='readonly'),
                # Item('n_px_facet_step_x', style='readonly'),
                # Item('n_px_facet_step_y', style='readonly'),
                id='aramisCDT.info',
                resizable=True
                )



