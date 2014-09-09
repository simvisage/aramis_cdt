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

def get_d(u_arr, integ_radius):
    ir = integ_radius
    du_arr = np.zeros_like(u_arr)
    du_arr[:, ir:-ir] = (u_arr[:, 2 * ir:] - u_arr[:, :-2 * ir])
    return du_arr


class AramisRawData(HasTraits):
    r'''Basic array structure containing the measured
    aramis data prepred for further elaboration in subclasses
    load data from *.npy files
    '''
    aramis_info = Instance(AramisInfo, params_changed=True)

    aramis_info_changed = Event
    @on_trait_change('aramis_info.data_dir')
    def aramis_info_change(self):
        if self.aramis_info.data_dir == '':
            print 'Data directory in UI is not defined!'
        else:
            self.aramis_info_changed = True

    evaluated_step_idx = Int(0, params_changed=True)
    r'''Current index of a stage for evaluation.
    '''

    evaluated_step_idx_filename = Property(Str, depends_on='+params_changed')
    r'''Filename for the evaluated step
    '''
    @cached_property
    def _get_evaluated_step_idx_filename(self):
        return '%s%d' % (self.aramis_info.displacements_basename,
                         self.aramis_info.aramis_stage_list[self.evaluated_step_idx])

    transform_data = Bool(False, params_changed=True)
    r'''Switch data transformation before analysis
    '''

    input_array_undeformed = Property(Array, depends_on='aramis_info_changed')
    r'''Cordinates in the initial state formated as an array
    with a first index representing the node number, the second indices
    indicate the following: :math:`X_n = [i,j,x_0,y_0,z_0]`.
    '''
    @cached_property
    def _get_input_array_undeformed(self, verbose=False):
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

    input_array = Property(Array, depends_on='aramis_info_changed, +params_changed')
    r'''Array of data exported from Aramis where for each node :math:`n`
    a data record defined as

    .. math::

        U_n = [i, j, u, v, w]

    specifying the
    row and column index of the node within the aramis grid and the displacement
    of the node in the :math:`x,y,z` directions.
    '''
    @cached_property
    def _get_input_array(self, verbose=False):
        fname = '%s%d' % (self.aramis_info.displacements_basename,
                         self.aramis_info.aramis_stage_list[self.evaluated_step_idx])
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

    ad_channels_arr = Property(Array, depends_on='aramis_info_changed')
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
    '''Aramis Data Structure
    '''

    #===========================================================================
    # Parameters
    #===========================================================================

    transform_data = Bool(False, params_changed=True)
    '''Switch data transformation before analysis
    '''

    evaluated_step_idx_filename = Property(Str, depends_on='+params_changed')
    '''Filename for the evaluated step
    '''
    @cached_property
    def _get_evaluated_step_idx_filename(self):
        return '%s%d' % (self.aramis_info.displacements_basename,
                         self.aramis_info.aramis_stage_list[self.evaluated_step_idx])

    #===========================================================================
    # Undeformed state variables
    #===========================================================================
    data_array_undeformed = Property(Array, depends_on='aramis_info_changed')
    '''Array of values for undeformed state in the first step.
    '''
    @cached_property
    def _get_data_array_undeformed(self):
        '''Load data for the first step from *.npy file. If file *.npy does
        not exist the data is load from *.txt and saved as *.npy.
        (improve speed of loading)
        '''
        if self.transform_data:
            return self._transform_coord(self.input_array_undeformed)
        else:
            return self.input_array_undeformed

    data_array_undeformed_shape = Property(Tuple, depends_on='input_array_undeformed')
    '''Shape of undeformed data array.
    '''
    @cached_property
    def _get_data_array_undeformed_shape(self):
        if self.aramis_info.data_dir == '':
            return (0, 0, 0)
        else:
            return (3, self.n_y_undeformed, self.n_x_undeformed)

    data_array_undeformed_mask = Property(Tuple, depends_on='input_array_undeformed')
    '''Default mask in undeformed state
    '''
    @cached_property
    def _get_data_array_undeformed_mask(self):
        return np.isnan(self.input_array_undeformed)

    n_x_undeformed = Property(Int, depends_on='aramis_info_changed')
    '''Number of facets in x-direction
    '''
    @cached_property
    def _get_n_x_undeformed(self):
        return self.input_array_undeformed.shape[2]

    n_y_undeformed = Property(Int, depends_on='aramis_info_changed')
    '''Number of facets in y-direction
    '''
    @cached_property
    def _get_n_y_undeformed(self):
        return self.input_array_undeformed.shape[1]

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

    x_arr_undeformed = Property(Array, depends_on='aramis_info_changed')
    '''Array of x-coordinates in undeformed state
    '''
    @cached_property
    def _get_x_arr_undeformed(self):
        return self.data_array_undeformed[0, :, :]

    y_arr_undeformed = Property(Array, depends_on='aramis_info_changed')
    '''Array of y-coordinates in undeformed state
    '''
    @cached_property
    def _get_y_arr_undeformed(self):
        return self.data_array_undeformed[1, :, :]

    z_arr_undeformed = Property(Array, depends_on='aramis_info_changed')
    '''Array of z-coordinates in undeformed state
    '''
    @cached_property
    def _get_z_arr_undeformed(self):
        return self.data_array_undeformed[2, :, :]

    length_x_undeformed = Property(Array, depends_on='aramis_info_changed')
    '''Length of the specimen in x-direction
    '''
    @cached_property
    def _get_length_x_undeformed(self):
        return np.nanmax(self.x_arr_undeformed) - np.nanmin(self.x_arr_undeformed)

    length_y_undeformed = Property(Array, depends_on='aramis_info_changed')
    '''Length of the specimen in y-direction
    '''
    @cached_property
    def _get_length_y_undeformed(self):
        return self.y_arr_undeformed.max() - self.y_arr_undeformed.min()

    length_z_undeformed = Property(Array, depends_on='aramis_info_changed')
    '''Length of the specimen in z-direction
    '''
    @cached_property
    def _get_length_z_undeformed(self):
        return self.z_arr_undeformed.max() - self.z_arr_undeformed.min()

    #===========================================================================
    # Data array
    #===========================================================================
    data_array = Property(Array, depends_on='aramis_info_changed, +params_changed')
    '''Array of Aramis exported data
    [index_x, index_y, displ_x, displ_y, displ_z]
    '''
    @cached_property
    def _get_data_array(self):
        if self.transform_data:
            return self.input_array
        else:
            return self.input_array

    #===========================================================================
    # Displacement arrays
    #===========================================================================
    ux_arr = Property(Array, depends_on='aramis_info_changed, +params_changed')
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

    ux_arr_avg = Property(Array, depends_on='aramis_info_changed, +params_changed')
    '''Average array of displacements in x-direction
    '''
    @cached_property
    def _get_ux_arr_avg(self):
        # missing values replaced with average in y-direction
        ux_arr = self.ux_arr
        ux_avg = np.average(ux_arr, axis=0)
        return ux_avg

    uy_arr = Property(Array, depends_on='aramis_info_changed, +params_changed')
    '''Array of displacements in y-direction
    '''
    @cached_property
    def _get_uy_arr(self):
        return self.data_array[1, :, :]

    uz_arr = Property(Array, depends_on='aramis_info_changed, +params_changed')
    '''Array of displacements in z-direction
    '''
    @cached_property
    def _get_uz_arr(self):
        return self.data_array[2, :, :]

    step_time = Property(Array, depends_on='aramis_info_changed')
    @cached_property
    def _get_step_time(self):
        return self.ad_channels_arr[:, 0]

    force = Property(Array, depends_on='aramis_info_changed')
    '''Force obtained from AD channel value
    '''
    @cached_property
    def _get_force(self):
        return np.array([10, 20])
        # return self.ad_channels_arr[:, 1]

    # TODO: load from exp db
    area = Float(100.0 * 20.0)

    stress = Property(Array, depends_on='aramis_info_changed')
    @cached_property
    def _get_stress(self):
        return self.force / self.area

    step_idx_max = Property(Int, depends_on='aramis_info_changed')
    @cached_property
    def _get_step_idx_max(self):
        if hasattr(self.aramis_info, 'number_of_steps'):
            return self.aramis_info.number_of_steps - 1
        else:
            return 1


    view = View(
                Item('evaluated_step_idx',
                     editor=RangeEditor(low=0, high_name='step_idx_max', mode='slider'),
                                        springy=True),
                Item('data_array_undeformed_shape', label='data shape', style='readonly'),
                # 'transform_data',
                id='aramisCDT.data',
                )

