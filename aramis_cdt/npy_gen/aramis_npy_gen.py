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
    HasTraits, Property, cached_property, Int, Array, Instance, Tuple, Button, List, Float

from traitsui.api import View, UItem

import numpy as np
import os
import re

import platform
import time
if platform.system() == 'Linux':
    sysclock = time.time
elif platform.system() == 'Windows':
    sysclock = time.clock

from aramis_cdt.aramis_info import AramisInfo


class AramisNPyGen(HasTraits):
    '''Class providing tools for preparation (generation) *.npy data files from
    Aramis *.txt files for Aramis Crack Detection Tool (AramisCDT). *.npy files 
    enable faster loading from disk.
    '''

    aramis_info = Instance(AramisInfo)

    # TODO: add to UI and multiply force
    force_t_mult_coef = Float(100.0)
    '''Multiplication coefficient to obtain force_t from AD channel value
    '''

    #===========================================================================
    # Undeformed state data
    #===========================================================================
    X = Property(Array, depends_on='aramis_info.data_dir')
    '''Array of values for undeformed state in the first step.
    '''
    @cached_property
    def _get_X(self):
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
        self.x_idx_min_0 = int(np.min(data_arr[:, 0]))
        self.y_idx_min_0 = int(np.min(data_arr[:, 1]))
        self.x_idx_max_0 = int(np.max(data_arr[:, 0]))
        self.y_idx_max_0 = int(np.max(data_arr[:, 1]))
        self.ni = int(self.x_idx_max_0 - self.x_idx_min_0 + 1)
        self.nj = int(self.y_idx_max_0 - self.y_idx_min_0 + 1)
        data_arr = self._prepare_data_structure(data_arr)

        np.save(fname_npy, data_arr)
        print 'loading time =', sysclock() - start_t
        print 'number of missing facets is', np.sum(np.isnan(data_arr).astype(int))
        return data_arr

    x_idx_min_0 = Int
    '''Minimum value of the indices in the first column of the undeformed state.
    '''

    y_idx_min_0 = Int
    '''Minimum value of the indices in the second column of the undeformed state.
    '''

    x_idx_max_0 = Int
    '''Maximum value of the indices in the first column of the undeformed state.
    '''

    y_idx_max_0 = Int
    '''Maximum value of the indices in the second column of the undeformed state.
    '''

    ni = Int
    '''Number of facets in x-direction
    '''

    nj = Int
    '''Number of facets in y-direction
    '''

    ad_channels_lst = List
    '''List of tuples (undeformed, deformed) obtained from AD channels in 
    aramis file header
    '''

    x_0_shape = Property(Tuple, depends_on='X')
    '''Shape of undeformed data array.
    '''
    @cached_property
    def _get_x_0_shape(self):
        return (3, self.nj, self.ni)

    i = Property(Int, depends_on='X')
    '''Indices in the first column of the undeformed state starting with zero.
    '''
    @cached_property
    def _get_i(self):
        return (np.arange(self.ni)[np.newaxis, :] *
                np.ones(self.nj)[:, np.newaxis]).astype(int)

    j = Property(Int, depends_on='X')
    '''Indices in the first column of the undeformed state starting with zero.
    '''
    @cached_property
    def _get_j(self):
        return (np.arange(self.nj)[np.newaxis, :] *
                np.ones(self.ni)[:, np.newaxis]).T

    ad_channels_arr = Property(Array)
    def _get_ad_channels_arr(self):
        data= np.array(self.ad_channels_lst, dtype=float)
        data[:,1] = data[:,1] * 100
        return data

    generate_npy = Button
    '''Generate npy files from Aramis *.txt data 
    '''
    def _generate_npy_fired(self):
        self.ad_channels_lst = []
        for step_idx in self.aramis_info.step_list:
            self._load_step_data(step_idx)
            self.__decompile_ad_channels(step_idx)
        np.save(os.path.join(self.aramis_info.npy_dir, 'ad_channels.npy'),
                self.ad_channels_arr)
        print self.ad_channels_arr

    #===========================================================================
    # Data preparation methods
    #===========================================================================
    def _load_step_data(self, step_idx):
        '''Load data for the specified step from *.npy file. If file *.npy does 
        not exist the data is load from *.txt and saved as *.npy. 
        (improve speed of loading)
        '''
        fname = '%s%d' % (self.aramis_info.displacements_basename,
                         self.aramis_info.aramis_stage_list[step_idx])
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
        if self.ni == 0:
            self.X
        data_arr = np.empty((self.ni * self.nj,
                                 input_arr.shape[1] - 2), dtype=float)
        data_arr.fill(np.nan)

        # input indices (columns 1 and 2)
        in_indices = input_arr[:, :2].astype(int)
        in_indices[:, 0] -= self.x_idx_min_0
        in_indices[:, 1] -= self.y_idx_min_0
        in_indices = in_indices.view([('', in_indices.dtype)] * in_indices.shape[1])

        # undeformed state indices
        un_indices = np.hstack((self.i.ravel()[:, np.newaxis],
                               self.j.ravel()[:, np.newaxis])).astype(int)
        un_indices = un_indices.view([('', un_indices.dtype)] * un_indices.shape[1])

        # data for higher steps have the same order of rows as
        # undeformed one but missing values
        mask = np.in1d(un_indices, in_indices, assume_unique=True)
        data_arr[mask] = input_arr[:, 2:]

        print data_arr.shape, self.x_0_shape
        data_arr = data_arr.T.reshape(self.x_0_shape)
        return data_arr

    def __decompile_ad_channels(self, step_idx):
        fname = '%s%d' % (self.aramis_info.displacements_basename,
                         self.aramis_info.aramis_stage_list[step_idx])
        with open(os.path.join(self.aramis_info.data_dir, fname + '.txt')) as infile:
            for i in range(30):
                line = infile.readline()
                m = re.match(r'#\s+AD-0:\s+[-+]?\d+\.\d+\s+(?P<force>[-+]?\d+\.\d+)', line)
                if m:
                    force = float(m.groups('force')[0])
		else:
		    force = 0
                m = re.match(r'#\s+deformt:\s+(?P<time>[-+]?\d+\.\d+)', line)
                if m:
                    time = float(m.groups('time')[0])
        self.ad_channels_lst.append([time, force])

    view = View(
                UItem('generate_npy'),
                )


if __name__ == '__main__':
    ns=[#'TTb-2C-1cm-0-800SBR-V1_cyc-Aramis2d-Xf15s13-Yf15s13',
#'TTb-2C-1cm-0-800SBR-V1_cyc-Aramis2d-Xf15s1-Yf15s4',
'TTb-2C-1cm-0-800SBR-V2_cyc-Aramis2d-Xf15s13-Yf15s13',
'TTb-2C-1cm-0-800SBR-V2_cyc-Aramis2d-Xf15s1-Yf15s4',
'TTb-2C-1cm-0-800SBR-V3_cyc-Aramis2d-Xf15s13-Yf15s13',
'TTb-2C-1cm-0-800SBR-V3_cyc-Aramis2d-Xf15s1-Yf15s4',
'TTb-2C-1cm-0-800SBR-V4_cyc-Aramis2d-Xf15s13-Yf15s13',
'TTb-2C-1cm-0-800SBR-V4_cyc-Aramis2d-Xf15s1-Yf15s4',
'TTb-2C-1cm-0-800SBR-V5_cyc-Aramis2d-Xf15s13-Yf15s13',
'TTb-2C-1cm-0-800SBR-V5_cyc-Aramis2d-Xf15s1-Yf15s4'
]
    data_dir = '/media/raid/Aachen/simdb_large_txt/'
    for n in ns:
        AI = AramisInfo(data_dir=data_dir+n)
        AG = AramisNPyGen(aramis_info=AI)
    #AG.configure_traits()
        AG.generate_npy = True
    
