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
    '''Class providing tools for preparation (generation) *.npy data files from
    Aramis *.txt files for Aramis Crack Detection Tool (AramisCDT). *.npy files 
    enable faster loading from disk.
    '''

    aramis_info = Instance(AramisInfo)

    #===========================================================================
    # Undeformed state data
    #===========================================================================
    input_array_undeformed = Property(Array, depends_on='aramis_info.data_dir')
    '''Array of values for undeformed state in the first step.
    '''
    @cached_property
    def _get_input_array_undeformed(self):
        '''Load data (undeformed coordinates) for the first step from *.txt and 
        save as *.npy. 
        '''
        fname = self.aramis_info.undeformed_coords_filename
        print 'loading', fname, '...'

        start_t = sysclock()
        dir_npy = self.aramis_info.npy_dir
        if os.path.exists(dir_npy) == False:
            os.mkdir(dir_npy)
        fname_npy = os.path.join(dir_npy, fname + '.npy')
        fname_txt = os.path.join(self.aramis_info.data_dir, fname + '.txt')

        data_arr = np.loadtxt(fname_txt,
                           # skiprows=14,  # not necessary
                           usecols=[0, 1, 2, 3, 4])
        self.x_idx_min_undeformed = int(np.min(data_arr[:, 0]))
        self.y_idx_min_undeformed = int(np.min(data_arr[:, 1]))
        self.x_idx_max_undeformed = int(np.max(data_arr[:, 0]))
        self.y_idx_max_undeformed = int(np.max(data_arr[:, 1]))
        self.n_x_undeformed = int(self.x_idx_max_undeformed - self.x_idx_min_undeformed + 1)
        self.n_y_undeformed = int(self.y_idx_max_undeformed - self.y_idx_min_undeformed + 1)
        data_arr = self._prepare_data_structure(data_arr)

        np.save(fname_npy, data_arr)
        print 'loading time =', sysclock() - start_t
        print 'number of missing facets is', np.sum(np.isnan(data_arr).astype(int))
        return data_arr

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

    ad_channels_lst = List
    '''List of tuples (undeformed, deformed) obtained from AD channels in 
    aramis file header
    '''

    data_array_undeformed_shape = Property(Tuple, depends_on='input_array_undeformed')
    '''Shape of undeformed data array.
    '''
    @cached_property
    def _get_data_array_undeformed_shape(self):
        return (3, self.n_y_undeformed, self.n_x_undeformed)

    x_idx_undeformed = Property(Int, depends_on='input_array_undeformed')
    '''Indices in the first column of the undeformed state starting with zero.
    '''
    @cached_property
    def _get_x_idx_undeformed(self):
        return (np.arange(self.n_x_undeformed)[np.newaxis, :] *
                np.ones(self.n_y_undeformed)[:, np.newaxis]).astype(int)

    y_idx_undeformed = Property(Int, depends_on='input_array_undeformed')
    '''Indices in the first column of the undeformed state starting with zero.
    '''
    @cached_property
    def _get_y_idx_undeformed(self):
        return (np.arange(self.n_y_undeformed)[np.newaxis, :] *
                np.ones(self.n_x_undeformed)[:, np.newaxis]).T

    ad_channels_arr = Property(Array)
    def _get_ad_channels_arr(self):
        return np.array(self.ad_channels_lst, dtype=float)

    generate_npy = Button
    '''Generate npy files from Aramis *.txt data 
    '''
    def _generate_npy_fired(self):
        self.ad_channels_lst = []
        for step_idx in self.aramis_info.step_idx_list:
            self._load_step_data(step_idx)
            self.__decompile_ad_channels(step_idx)
        np.save(os.path.join(self.aramis_info.npy_dir, 'ad_channels.npy'),
                self.ad_channels_arr)

    #===========================================================================
    # Data preparation methods
    #===========================================================================
    def _load_step_data(self, step_idx):
        '''Load data for the specified step from *.npy file. If file *.npy does 
        not exist the data is load from *.txt and saved as *.npy. 
        (improve speed of loading)
        '''
        fname = '%s%d' % (self.aramis_info.displacements_basename,
                         self.aramis_info.step_list[step_idx])
        print 'loading', fname, '...'

        start_t = sysclock()
        dir_npy = self.aramis_info.npy_dir
        if os.path.exists(dir_npy) == False:
            os.mkdir(dir_npy)
        fname_npy = os.path.join(dir_npy, fname + '.npy')
        fname_txt = os.path.join(self.aramis_info.data_dir, fname + '.txt')
#         if os.path.exists(fname_npy):
#             data_arr = np.load(fname_npy)
#         else:
        data_arr = np.loadtxt(fname_txt,
                           # skiprows=14,  # not necessary
                           usecols=[0, 1, 2, 3, 4])
        data_arr = self._prepare_data_structure(data_arr)

        np.save(fname_npy, data_arr)
        print 'loading time =', sysclock() - start_t
        print 'number of missing facets is', np.sum(np.isnan(data_arr).astype(int))
        return data_arr

    def _prepare_data_structure(self, input_arr):
        if self.n_x_undeformed == 0:
            self.input_array_undeformed
        data_arr = np.empty((self.n_x_undeformed * self.n_y_undeformed,
                                 input_arr.shape[1] - 2), dtype=float)
        data_arr.fill(np.nan)

        # input indices (columns 1 and 2)
        in_indices = input_arr[:, :2].astype(int)
        in_indices[:, 0] -= self.x_idx_min_undeformed
        in_indices[:, 1] -= self.y_idx_min_undeformed
        in_indices = in_indices.view([('', in_indices.dtype)] * in_indices.shape[1])

        # undeformed state indices
        un_indices = np.hstack((self.x_idx_undeformed.ravel()[:, np.newaxis],
                               self.y_idx_undeformed.ravel()[:, np.newaxis])).astype(int)
        un_indices = un_indices.view([('', un_indices.dtype)] * un_indices.shape[1])

        # data for higher steps have the same order of rows as
        # undeformed one but missing values
        mask = np.in1d(un_indices, in_indices, assume_unique=True)
        data_arr[mask] = input_arr[:, 2:]

        print data_arr.shape, self.data_array_undeformed_shape
        data_arr = data_arr.T.reshape(self.data_array_undeformed_shape)
        return data_arr

    def __decompile_ad_channels(self, step_idx):
        fname = '%s%d' % (self.aramis_info.displacements_basename,
                         self.aramis_info.step_list[step_idx])
        ad_channels = []
        with open(os.path.join(self.aramis_info.data_dir, fname + '.txt')) as infile:
            for i in range(30):
                line = infile.readline()
                m = re.match(r'#\s+AD-(\d+):\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)', line)
                if m:
                    ad_channels.append(m.groups())
        self.ad_channels_lst.append(ad_channels)

    view = View(
                UItem('generate_npy'),
                )


if __name__ == '__main__':
    data_dir = '/media/data/_linux_data/aachen/Aramis_07_2013/TT-12c-6cm-0-TU-SH4-V1-Xf19s15-Yf19s15'
    AI = AramisInfo(data_dir=data_dir)
    AG = AramisNPyGen(aramis_info=AI)
    AG.configure_traits()
