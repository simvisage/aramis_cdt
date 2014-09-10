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
    HasTraits, Float, Property, cached_property, Int, Array, Bool, \
    Instance, DelegatesTo, Tuple, Button, List, Str, Event, on_trait_change

from etsproxy.traits.ui.api import View, Item, HGroup, EnumEditor, Group, UItem, RangeEditor

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

from aramis_info import AramisInfo


class AramisRawData(HasTraits):
    r'''Basic array structure containing the measured
    aramis data prepred for further elaboration in subclasses
    load data from *.npy files
    '''
    aramis_info = Instance(AramisInfo, params_changed=True)

    current_step = Int(0, params_changed=True)
    r'''Number of current step for evaluation.
    '''

    transform_data = Bool(False, params_changed=True)
    r'''Switch data transformation before analysis
    '''

    X = Property(Array, depends_on='aramis_info.+params_changed')
    r'''Cordinates in the initial state formated as an array
    with a first index representing the node number, the second indices
    indicate the following: 
    
    .. math::
            X_n = [i,j,x_0,y_0,z_0].
    '''
    @cached_property
    def _get_X(self, verbose=False):
        fname = self.aramis_info.undeformed_coords_filename
        if verbose:
            print '#' * 50
            print 'loading', fname, '...'

        start_t = sysclock()
        dir_npy = self.aramis_info.npy_dir
        fname_npy = os.path.join(dir_npy, fname + '.npy')

        if os.path.exists(fname_npy):
            data_arr = np.load(fname_npy)
        else:
            raise IOError, '%s data does not exist!' % fname_npy
        if verbose:
            print 'loading time =', sysclock() - start_t
            print 'number of missing facets is', np.sum(np.isnan(data_arr).astype(int))
        return data_arr

    current_step_filename = Property(Str, depends_on='+params_changed')
    r'''Filename for the evaluated step
    '''
    @cached_property
    def _get_current_step_filename(self):
        return '%s%d' % (self.aramis_info.displacements_basename,
                         self.aramis_info.aramis_stage_list[self.current_step])

    U = Property(Array, depends_on='aramis_info.+params_changed, +params_changed')
    r'''Array of data exported from Aramis where for each node :math:`n`
    a data record defined as

    .. math::

        U_n = [i, j, u, v, w]

    specifying the
    row and column index of the node within the aramis grid and the displacement
    of the node in the :math:`x,y,z` directions.
    '''
    @cached_property
    def _get_U(self, verbose=False):
        fname = self.current_step_filename
        if verbose:
            print 'loading', fname, '...'

        start_t = sysclock()
        dir_npy = self.aramis_info.npy_dir
        if os.path.exists(dir_npy) == False:
            os.mkdir(dir_npy)
        fname_npy = os.path.join(dir_npy, fname + '.npy')
        data_arr = np.load(fname_npy)
        if verbose:
            print 'loading time =', sysclock() - start_t
            print 'number of missing facets is', np.sum(np.isnan(data_arr).astype(int))
        return data_arr

    ad_channels_arr = Property(Array, depends_on='aramis_info.+params_changed')
    '''Array of additional channel data including computer time and universal time
    for synchronization with external data.
    '''
    @cached_property
    def _get_ad_channels_arr(self):
        ad_channels_file = os.path.join(self.aramis_info.npy_dir, 'ad_channels.npy')
        if os.path.exists(ad_channels_file):
            return np.load(ad_channels_file)
        else:
            print 'File %s does not exists!' % ad_channels_file


class AramisFieldData(AramisRawData):
    '''Field data structure for AramisCDT
    '''

    #===========================================================================
    # Parameters
    #===========================================================================

    transform_data = Bool(False, params_changed=True)
    '''Switch data transformation before analysis
    '''

    #===========================================================================
    # Initial state arrays - coordinates
    #===========================================================================
    data_array_0 = Property(Array, depends_on='aramis_info.+params_changed')
    '''Array of values for initial state in the first step 
    '''
    @cached_property
    def _get_data_array_0(self):
        if self.transform_data:
            return self.X
        else:
            return self.X

    data_array_0_mask = Property(Tuple, depends_on='X')
    '''Default mask in undeformed state
    '''
    @cached_property
    def _get_data_array_0_mask(self):
        return np.isnan(self.X)

    n_x_0 = Property(Int, depends_on='aramis_info.+params_changed')
    '''Number of facets in x-direction
    '''
    @cached_property
    def _get_n_x_0(self):
        return self.X.shape[2]

    n_y_0 = Property(Int, depends_on='aramis_info.+params_changed')
    '''Number of facets in y-direction
    '''
    @cached_property
    def _get_n_y_0(self):
        return self.X.shape[1]

    x_idx_0 = Property(Int, depends_on='X')
    '''Indices in the first column of the undeformed state starting with zero.
    '''
    @cached_property
    def _get_x_idx_0(self):
        return (np.arange(self.n_x_0)[np.newaxis, :] *
                np.ones(self.n_y_0)[:, np.newaxis]).astype(int)

    y_idx_0 = Property(Int, depends_on='X')
    '''Indices in the first column of the undeformed state starting with zero.
    '''
    @cached_property
    def _get_y_idx_0(self):
        return (np.arange(self.n_y_0)[np.newaxis, :] *
                np.ones(self.n_x_0)[:, np.newaxis]).T

    x_arr_0 = Property(Array, depends_on='aramis_info.+params_changed')
    '''Array of x-coordinates in undeformed state
    '''
    @cached_property
    def _get_x_arr_0(self):
        return self.data_array_0[0, :, :]

    y_arr_0 = Property(Array, depends_on='aramis_info.+params_changed')
    '''Array of y-coordinates in undeformed state
    '''
    @cached_property
    def _get_y_arr_0(self):
        return self.data_array_0[1, :, :]

    z_arr_0 = Property(Array, depends_on='aramis_info.+params_changed')
    '''Array of z-coordinates in undeformed state
    '''
    @cached_property
    def _get_z_arr_0(self):
        return self.data_array_0[2, :, :]

    #------------- measuring field parameters in initial state

    data_array_0_shape = Property(Tuple, depends_on='X')
    '''Shape of undeformed data array.
    '''
    @cached_property
    def _get_data_array_0_shape(self):
        if self.aramis_info.data_dir == '':
            return (0, 0, 0)
        else:
            return (3, self.n_y_0, self.n_x_0)

    length_x_0 = Property(Array, depends_on='aramis_info.+params_changed')
    '''Length of the measuring area in x-direction
    '''
    @cached_property
    def _get_length_x_0(self):
        return np.nanmax(self.x_arr_0) - np.nanmin(self.x_arr_0)

    length_y_0 = Property(Array, depends_on='aramis_info.+params_changed')
    '''Length of the measuring area in y-direction
    '''
    @cached_property
    def _get_length_y_0(self):
        return self.y_arr_0.max() - self.y_arr_0.min()

    length_z_0 = Property(Array, depends_on='aramis_info.+params_changed')
    '''Length of the measuring area in z-direction
    '''
    @cached_property
    def _get_length_z_0(self):
        return self.z_arr_0.max() - self.z_arr_0.min()

    x_arr_stats = Property(depends_on='aramis_info.+params_changed')
    '''
    * mu_mm - mean value of facet midpoint distance [mm]
    * std_mm - standard deviation of facet midpoint distance [mm]
    * mu_px_mm - mean value of one pixel size [mm]
    * std_px_mm - standard deviation of one pixel size [mm]
    '''
    @cached_property
    def _get_x_arr_stats(self):
        x_diff = np.diff(self.x_arr_0, axis=1)
        mu_mm = np.mean(x_diff)
        std_mm = np.std(x_diff)
        mu_px_mm = np.mean(x_diff / self.aramis_info.n_px_facet_step_x)
        std_px_mm = np.std(x_diff / self.aramis_info.n_px_facet_step_x)
        return mu_mm, std_mm, mu_px_mm, std_px_mm

    #===========================================================================
    # Displacement arrays
    #===========================================================================
    data_array = Property(Array, depends_on='aramis_info.+params_changed, +params_changed')
    '''Array of displacements for current step
    '''
    @cached_property
    def _get_data_array(self):
        if self.transform_data:
            return self.U
        else:
            return self.U

    ux_arr = Property(Array, depends_on='aramis_info.+params_changed, +params_changed')
    '''Array of displacements in x-direction
    '''
    @cached_property
    def _get_ux_arr(self):
        # missing values replaced by averaged values in y-direction
        ux_arr = self.data_array[0, :, :]
        ux_masked = np.ma.masked_array(ux_arr, mask=np.isnan(ux_arr))
        ux_avg = np.ma.average(ux_masked, axis=0)

        y_idx_zeros, x_idx_zeros = np.where(np.isnan(ux_arr))
        ux_arr[y_idx_zeros, x_idx_zeros] = ux_avg[x_idx_zeros]
        # ux_arr[ux_arr < 0.01] = 0
        return ux_arr

    ux_arr_avg = Property(Array, depends_on='aramis_info.+params_changed, +params_changed')
    '''Average array of displacements in x-direction
    '''
    @cached_property
    def _get_ux_arr_avg(self):
        # missing values replaced with average in y-direction
        ux_arr = self.ux_arr
        ux_avg = np.average(ux_arr, axis=0)
        return ux_avg

    uy_arr = Property(Array, depends_on='aramis_info.+params_changed, +params_changed')
    '''Array of displacements in y-direction
    '''
    @cached_property
    def _get_uy_arr(self):
        return self.data_array[1, :, :]

    uz_arr = Property(Array, depends_on='aramis_info.+params_changed, +params_changed')
    '''Array of displacements in z-direction
    '''
    @cached_property
    def _get_uz_arr(self):
        return self.data_array[2, :, :]

    step_time = Property(Array, depends_on='aramis_info.+params_changed')
    '''Time of the step obtained from Aramis 
    '''
    @cached_property
    def _get_step_time(self):
        return self.ad_channels_arr[:, 0]

    step_max = Property(Int, depends_on='aramis_info.+params_changed')
    @cached_property
    def _get_step_max(self):
        return self.aramis_info.number_of_steps - 1


    view = View(
                Item('current_step',
                     editor=RangeEditor(low=0, high_name='step_max', mode='slider'),
                                        springy=True),
                Item('data_array_0_shape', label='data shape', style='readonly'),
                Item('length_x_0', label='l_x', style='readonly'),
                Item('length_y_0', label='l_y', style='readonly'),
                Item('length_z_0', label='l_z', style='readonly'),
                Item('x_arr_stats', label='(mu_x, std_x, mu_px_x, std_px_x)', style='readonly'),
                # 'transform_data',
                id='aramisCDT.data',
                )

