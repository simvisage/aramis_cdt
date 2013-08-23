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
    Instance, DelegatesTo, Tuple, Button, List, Str

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
    '''Aramis Data Structure - load data from *.npy files 
    '''
    aramis_info = Instance(AramisInfo)

    evaluated_step_idx = Int(0, params_changed=True)
    '''Time step used for the evaluation.
    '''

    input_array_undeformed = Property(Array, depends_on='aramis_info.data_dir')
    '''Array of values for undeformed state in the first step.
    '''
    @cached_property
    def _get_input_array_undeformed(self):
        '''Load data for the first step from *.npy file. If file *.npy does 
        not exist the data is load from *.txt and saved as *.npy. 
        (improve speed of loading)
        '''
        fname = self.aramis_info.undeformed_coords_filename
        print 'loading', fname, '...'

        start_t = sysclock()
        dir_npy = self.aramis_info.npy_dir
        fname_npy = os.path.join(dir_npy, fname + '.npy')

        if os.path.exists(fname_npy):
            data_arr = np.load(fname_npy)
        else:
            print '*.npy data does not exist!'
        print 'loading time =', sysclock() - start_t
        print 'number of missing facets is', np.sum(np.isnan(data_arr).astype(int))
        return data_arr

    input_array = Property(Array, depends_on='aramis_info.data_dir, +params_changed')
    '''Array of Aramis exported data 
    [index_x, index_y, displ_x, displ_y, displ_z]
    '''
    @cached_property
    def _get_input_array(self):
        fname = '%s%d' % (self.aramis_info.displacements_basename,
                         self.aramis_info.step_list[self.evaluated_step_idx])
        print 'loading', fname, '...'

        start_t = sysclock()
        dir_npy = self.aramis_info.npy_dir
        if os.path.exists(dir_npy) == False:
            os.mkdir(dir_npy)
        fname_npy = os.path.join(dir_npy, fname + '.npy')
        data_arr = np.load(fname_npy)
        print 'loading time =', sysclock() - start_t
        print 'number of missing facets is', np.sum(np.isnan(data_arr).astype(int))
        return data_arr


class AramisData(AramisRawData):
    '''Aramis Data Structure
    '''

    step_list = DelegatesTo('aramis_info')

    #===========================================================================
    # Parameters
    #===========================================================================

    # TODO: evaluate from l-d diag else -1 : np.argwhere(x==max).max()
    crack_detect_idx = Int(-1, params_changed=True)
    '''Index of the step used to determine the crack pattern
    '''

    # integration radius for the non-local average of the measured strain
    # defined as integer value of the number of facets (or elements)
    #
    # the value should correspond to
#    def _integ_radius_default(self):
#        return ceil( float( self.n_px_f / self.n_px_a )
    integ_radius = Int(11, params_changed=True)

    transform_data = Bool(False, params_changed=True)
    '''Switch data transformation before analysis
    '''

    evaluated_step_idx_filename = Property(Str, depends_on='+params_changed')
    '''Filename for the evaluated step 
    '''
    def _get_evaluated_step_idx_filename(self):
        return '%s%d' % (self.aramis_info.displacements_basename,
                         self.aramis_info.step_list[self.evaluated_step_idx])

    #===========================================================================
    # Undeformed state variables
    #===========================================================================
    data_array_undeformed = Property(Array, depends_on='aramis_info.data_dir')
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
        return (3, self.n_y_undeformed, self.n_x_undeformed)

    data_array_undeformed_mask = Property(Tuple, depends_on='input_array_undeformed')
    '''Default mask in undeformed state
    '''
    @cached_property
    def _get_data_array_undeformed_mask(self):
        return np.isnan(self.input_array_undeformed)

    n_x_undeformed = Property(Int, depends_on='aramis_info.data_dir')
    '''Number of facets in x-direction
    '''
    def _get_n_x_undeformed(self):
        return self.input_array_undeformed.shape[2]

    n_y_undeformed = Property(Int, depends_on='aramis_info.data_dir')
    '''Number of facets in y-direction
    '''
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

    x_arr_undeformed = Property(Array, depends_on='aramis_info.data_dir')
    '''Array of x-coordinates in undeformed state
    '''
    @cached_property
    def _get_x_arr_undeformed(self):
        return self.data_array_undeformed[0, :, :]

    y_arr_undeformed = Property(Array, depends_on='aramis_info.data_dir')
    '''Array of y-coordinates in undeformed state
    '''
    @cached_property
    def _get_y_arr_undeformed(self):
        return self.data_array_undeformed[1, :, :]

    z_arr_undeformed = Property(Array, depends_on='aramis_info.data_dir')
    '''Array of z-coordinates in undeformed state
    '''
    @cached_property
    def _get_z_arr_undeformed(self):
        return self.data_array_undeformed[2, :, :]

    length_x_undeformed = Property(Array, depends_on='aramis_info.data_dir')
    '''Length of the specimen in x-direction
    '''
    @cached_property
    def _get_length_x_undeformed(self):
        return np.nanmax(self.x_arr_undeformed) - np.nanmin(self.x_arr_undeformed)

    length_y_undeformed = Property(Array, depends_on='aramis_info.data_dir')
    '''Length of the specimen in y-direction
    '''
    @cached_property
    def _get_length_y_undeformed(self):
        return self.y_arr_undeformed.max() - self.y_arr_undeformed.min()

    length_z_undeformed = Property(Array, depends_on='aramis_info.data_dir')
    '''Length of the specimen in z-direction
    '''
    @cached_property
    def _get_length_z_undeformed(self):
        return self.z_arr_undeformed.max() - self.z_arr_undeformed.min()

    #===========================================================================
    # Data array
    #===========================================================================
    data_array = Property(Array, depends_on='aramis_info.data_dir, +params_changed')
    '''Array of Aramis exported data 
    [index_x, index_y, displ_x, displ_y, displ_z]
    '''
    @cached_property
    def _get_data_array(self):
        if self.transform_data:
            return self._transform_displ(self.input_array)
        else:
            return self.input_array

    def _transform_coord(self, input_array):
        data_arr = input_array
        mask = ~self.data_array_undeformed_mask[0, :, :]
        shape = data_arr.shape

        # rotate counterclockwise about the axis z
        linreg = stats.linregress(self.input_array_undeformed[0, :, :][mask].ravel(),
                                  self.input_array_undeformed[1, :, :][mask].ravel())
        alpha = np.arctan(-linreg[0])
        rot_z = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                        [np.sin(alpha), np.cos(alpha), 0],
                        [0, 0, 1]])

        # rotate counterclockwise about the axis y
        linreg = stats.linregress(self.input_array_undeformed[0, :, :][mask].ravel(),
                                  self.input_array_undeformed[2, :, :][mask].ravel())
        alpha = np.arctan(-linreg[0])
        rot_y = np.array([[np.cos(alpha), 0, -np.sin(alpha)],
                          [0, 1, 0],
                          [np.sin(alpha), 0, np.cos(alpha)]])

        # rotate counterclockwise about the axis x
        linreg = stats.linregress(self.input_array_undeformed[1, :, :][mask].ravel(),
                                  self.input_array_undeformed[2, :, :][mask].ravel())
        alpha = np.arctan(-linreg[0])
        rot_x = np.array([[1, 0, 0],
                        [0, np.cos(alpha), -np.sin(alpha)],
                        [0, np.sin(alpha), np.cos(alpha)]])
        shape = data_arr.shape

        # rotate coordinates
        tmp = np.dot(rot_z, data_arr.ravel().reshape(3, shape[1] * shape[2]))
        tmp = np.dot(rot_y, tmp)
        data_arr = np.dot(rot_x, tmp).reshape(3, shape[1], shape[2])

        # translation
        data_arr -= np.nanmin(np.nanmin(data_arr, axis=1),
                              axis=1)[:, np.newaxis, np.newaxis]

        return data_arr

    def _transform_displ(self, input_array):
        data_arr = input_array
        mask = ~self.data_array_undeformed_mask[0, :, :]
        shape = data_arr.shape

        # rotate counterclockwise about the axis z
        linreg = stats.linregress(self.input_array_undeformed[0, :, :][mask].ravel(),
                                  self.input_array_undeformed[1, :, :][mask].ravel())
        alpha = np.arctan(-linreg[0])
        rot_z = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                        [np.sin(alpha), np.cos(alpha), 0],
                        [0, 0, 1]])

        # rotate counterclockwise about the axis y
        linreg = stats.linregress(self.input_array_undeformed[0, :, :][mask].ravel(),
                                  self.input_array_undeformed[2, :, :][mask].ravel())
        alpha = np.arctan(-linreg[0])
        rot_y = np.array([[np.cos(alpha), 0, -np.sin(alpha)],
                          [0, 1, 0],
                          [np.sin(alpha), 0, np.cos(alpha)]])

        # rotate counterclockwise about the axis x
        linreg = stats.linregress(self.input_array_undeformed[1, :, :][mask].ravel(),
                                  self.input_array_undeformed[2, :, :][mask].ravel())
        alpha = np.arctan(-linreg[0])
        rot_x = np.array([[1, 0, 0],
                        [0, np.cos(alpha), -np.sin(alpha)],
                        [0, np.sin(alpha), np.cos(alpha)]])
        shape = data_arr.shape

        # rotate displacements
        tmp = np.dot(rot_z, data_arr.ravel().reshape(3, shape[1] * shape[2]))
        tmp = np.dot(rot_y, tmp)
        data_arr = np.dot(rot_x, tmp).reshape(3, shape[1], shape[2])

        return data_arr

    #===========================================================================
    # Displacement arrays
    #===========================================================================
    ux_arr = Property(Array, depends_on='aramis_info.data_dir, +params_changed')
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

    ux_arr_avg = Property(Array, depends_on='aramis_info.data_dir, +params_changed')
    '''Average array of displacements in x-direction
    '''
    @cached_property
    def _get_ux_arr_avg(self):
        # missing values replaced with average in y-direction
        ux_arr = self.ux_arr
        ux_avg = np.average(ux_arr, axis=0)
        return ux_avg

    uy_arr = Property(Array, depends_on='aramis_info.data_dir, +params_changed')
    '''Array of displacements in y-direction
    '''
    @cached_property
    def _get_uy_arr(self):
        return self.data_array[1, :, :]

    uz_arr = Property(Array, depends_on='aramis_info.data_dir, +params_changed')
    '''Array of displacements in z-direction
    '''
    @cached_property
    def _get_uz_arr(self):
        return self.data_array[2, :, :]

    d_ux_threshold = Float(0.0)
    '''The first derivative of displacement in x-direction threshold
    '''

    d_ux_arr = Property(Array, depends_on='aramis_info.data_dir, +params_changed')
    '''The first derivative of displacement in x-direction
    '''
    @cached_property
    def _get_d_ux_arr(self):
        d_arr = get_d(self.ux_arr, self.integ_radius)
        # cutoff the negative strains - noise
        d_arr[ d_arr < 0.0 ] = 0.0
        # eliminate displacement jumps smaller then specified tolerance
        d_arr[ d_arr < self.d_ux_threshold ] = 0.0
        return d_arr

    dd_ux_arr = Property(Array, depends_on='aramis_info.data_dir, +params_changed')
    @cached_property
    def _get_dd_ux_arr(self):
        return get_d(self.d_ux_arr, self.integ_radius)

    dd_ux_arr_avg = Property(Array, depends_on='aramis_info.data_dir, +params_changed')
    @cached_property
    def _get_dd_ux_arr_avg(self):
        return np.average(self.dd_ux_arr, axis=0)

    ddd_ux_arr = Property(Array, depends_on='aramis_info.data_dir, +params_changed')
    @cached_property
    def _get_ddd_ux_arr(self):
        return get_d(self.dd_ux_arr, self.integ_radius)

    ddd_ux_arr_avg = Property(Array, depends_on='aramis_info.data_dir, +params_changed')
    @cached_property
    def _get_ddd_ux_arr_avg(self):
        return np.average(self.ddd_ux_arr, axis=0)

    d_ux_arr_avg = Property(Array, depends_on='aramis_info.data_dir, +params_changed')
    '''Average of d_ux in y-direction
    '''
    @cached_property
    def _get_d_ux_arr_avg(self):
        d_u = self.d_ux_arr
        d_avg = np.average(d_u, axis=0)
        return d_avg

    d_uy_arr = Property(Array, depends_on='aramis_info.data_dir, +params_changed')
    '''The first derivative of displacement in x-direction
    '''
    @cached_property
    def _get_d_uy_arr(self):
        d_arr = get_d(self.uy_arr, self.integ_radius)
        # cutoff the negative strains - noise
        d_arr[ d_arr < 0.0 ] = 0.0
        # eliminate displacement jumps smaller then specified tolerance
        d_arr[ d_arr < self.d_ux_threshold ] = 0.0
        return d_arr

    d_uz_arr = Property(Array, depends_on='aramis_info.data_dir, +params_changed')
    '''The first derivative of displacement in x-direction
    '''
    @cached_property
    def _get_d_uz_arr(self):
        d_arr = get_d(self.uz_arr, self.integ_radius)
        # cutoff the negative strains - noise
        d_arr[ d_arr < 0.0 ] = 0.0
        # eliminate displacement jumps smaller then specified tolerance
        d_arr[ d_arr < self.d_ux_threshold ] = 0.0
        return d_arr

    #===========================================================================
    # Crack detection
    #===========================================================================
    dd_ux_threshold = Float(0.0)
    '''The second derivative of displacement in x-direction threshold
    '''

    ddd_ux_threshold = Float(-0.01)
    '''The third derivative of displacement in x-direction threshold
    '''

    dd_ux_avg_threshold = Float(0.0)
    '''Average of he second derivative of displacement in x-direction threshold
    '''

    ddd_ux_avg_threshold = Float(-0.01)
    '''Average the third derivative of displacement in x-direction threshold
    '''

    crack_filter = Property(Array, depends_on='aramis_info.data_dir, +params_changed')
    @cached_property
    def _get_crack_filter(self):
        dd_ux_arr = self.dd_ux_arr
        ddd_ux_arr = self.ddd_ux_arr
        return ((dd_ux_arr[:, 1:] * dd_ux_arr[:, :-1] < self.dd_ux_threshold) *
                ((ddd_ux_arr[:, 1:] + ddd_ux_arr[:, :-1]) / 2.0 < self.ddd_ux_threshold))

    number_of_subcracks = Property(Int, depends_on='aramis_info.data_dir, +params_changed')
    '''Number of sub-cracks
    '''
    def _get_number_of_subcracks(self):
        return np.sum(self.crack_filter)

    crack_filter_avg = Property(Array, depends_on='aramis_info.data_dir, +params_changed')
    @cached_property
    def _get_crack_filter_avg(self):
        dd_ux_arr_avg = self.dd_ux_arr_avg
        ddd_ux_arr_avg = self.ddd_ux_arr_avg
        crack_filter_avg = ((dd_ux_arr_avg[1:] * dd_ux_arr_avg[:-1] < self.dd_ux_avg_threshold) *
                           ((ddd_ux_arr_avg[1:] + ddd_ux_arr_avg[:-1]) / 2.0 < self.ddd_ux_avg_threshold))
        print "number of cracks determined by 'crack_filter_avg': ", np.sum(crack_filter_avg)
        return crack_filter_avg

    number_of_cracks_avg = Property(Int, depends_on='aramis_info.data_dir, +params_changed')
    '''Number of cracks using averaging
    '''
    def _get_number_of_cracks_avg(self):
        return np.sum(self.crack_filter_avg)

    crack_spacing_avg = Property(Array, depends_on='aramis_info.data_dir, +params_changed')
    @cached_property
    def _get_crack_spacing_avg(self):
        n_cr_avg = np.sum(self.crack_filter_avg)
        s_cr_avg = self.length_x_undeformed / n_cr_avg
        print "average crack spacing [mm]: %.1f" % (s_cr_avg)
        return s_cr_avg

    crack_arr = Property(Array, depends_on='aramis_info.data_dir, +params_changed')
    @cached_property
    def _get_crack_arr(self):
        return self.d_ux_arr[np.where(self.crack_filter)]

    crack_arr_mean = Property(Float, depends_on='aramis_info.data_dir, +params_changed')
    def _get_crack_arr_mean(self):
        return self.crack_arr.mean()

    crack_arr_std = Property(Float, depends_on='aramis_info.data_dir, +params_changed')
    def _get_crack_arr_std(self):
        return self.crack_arr.std()

    crack_avg_arr = Property(Array, depends_on='aramis_info.data_dir, +params_changed')
    @cached_property
    def _get_crack_avg_arr(self):
        return self.d_ux_arr_avg[np.where(self.crack_filter_avg)]

    crack_field_arr = Property(Array, depends_on='aramis_info.data_dir, +params_changed')
    @cached_property
    def _get_crack_field_arr(self):
        cf_w = np.zeros_like(self.d_ux_arr)
        cf_w[np.where(self.crack_filter)] = self.crack_arr
        return cf_w

    #===========================================================================
    # Crack detection in time
    #===========================================================================
    number_of_cracks_t = Array
    '''Number of cracks in time
    '''

    number_of_missing_facets_t = List
    '''Number of missing facets in time (missing, unidentified, not satisfying
    conditions, destroyed by crack)
    '''

    control_strain_t = Array()
    '''Control strain measured in the middle of y
    '''

    init_step_avg_lst = List()
    '''List of steps (list length is equal to number of cracks at the 
    crack_detect_idx) when the crack initiate
    '''
    init_step_lst = List()

    crack_width_avg_t = Array
    crack_stress_t = Array

    force_t_mult_coef = Float(100.0)
    '''Multiplication coefficient to obtain force_t from AD channel value
    '''

    # TODO: load from exp db
    force_t_to_stress = Float(100.0 * 20.0)

    force_t = Property(Array)
    '''Force evaluated from AD channel value (default channel for force_t is 0)
    '''
    def _get_force_t(self):
        return (self.ad_channels_arr[:, 0, 2] -
                self.ad_channels_arr[:, 0, 1]) * self.force_t_mult_coef

    stress_t = Property(Array)
    '''Concrete stress
    '''
    def _get_stress_t(self):
        return self.force_t * 1000 / self.force_t_to_stress

    ad_channels_arr = Property(Array)
    def _get_ad_channels_arr(self):
        ad_channels_file = os.path.join(self.aramis_info.npy_dir, 'ad_channels.npy')
        if os.path.exists(ad_channels_file):
            return np.load(ad_channels_file)
        else:
            print 'File %s does not exists!' % ad_channels_file

    number_of_cracks_t_analyse = Bool(True)

    def __number_of_cracks_t(self):
        self.number_of_cracks_t = np.append(self.number_of_cracks_t,
                                           self.number_of_cracks_avg)

    def __decompile_ad_channels(self):
        fname = self.evaluated_step_idx_filename
        ad_channels = []
        with open(os.path.join(self.aramis_info.data_dir, fname + '.txt')) as infile:
            for i in range(30):
                line = infile.readline()
                m = re.match(r'#\s+AD-(\d+):\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)', line)
                if m:
                    ad_channels.append(m.groups())
        self.ad_channels_lst.append(ad_channels)

    crack_detect_mask_avg = Property(Array, depends_on='crack_detect_idx')
    '''Mask of cracks identified in crack_detect_idx and used for backward 
    identification of the crack initialization.
    '''
    @cached_property
    def _get_crack_detect_mask_avg(self):
        self.evaluated_step_idx = self.crack_detect_idx
        return self.crack_filter_avg

    crack_detect_mask = Property(Array, depends_on='crack_detect_idx')
    '''Mask of cracks identified in crack_detect_idx and used for backward 
    identification of the crack initialization.
    '''
    @cached_property
    def _get_crack_detect_mask(self):
        self.evaluated_step_idx = self.crack_detect_idx
        return self.crack_filter

    run_t = Button('Run in time')
    '''Run analysis of all steps in time
    '''
    def _run_t_fired(self):
        start_step_idx = self.evaluated_step_idx
        self.number_of_cracks_t = np.array([])
        self.number_of_missing_facets_t = []
        self.control_strain_t = np.array([])

        for step_idx in self.aramis_info.step_idx_list:
            self.evaluated_step_idx = step_idx
            if self.number_of_cracks_t_analyse:
                self.__number_of_cracks_t()
            self.control_strain_t = np.append(self.control_strain_t,
                                            (self.ux_arr[20, -10] - self.ux_arr[20, 10]) /
                                            (self.x_arr_undeformed[20, -10] - self.x_arr_undeformed[20, 10]))
            self.number_of_missing_facets_t.append(np.sum(np.isnan(self.data_array).astype(int)))

        self.evaluated_step_idx = start_step_idx

    run_back = Button('Run back in time')
    '''Run analysis of all steps in time
    '''
    def _run_back_fired(self):
        # todo: better
        x = self.x_arr_undeformed[20, :]

        # import matplotlib.pyplot as plt
        # plt.rc('font', size=25)
        step = self.crack_detect_idx
        crack_step_avg = np.zeros_like(self.crack_detect_mask_avg, dtype=int)
        crack_step_avg[self.crack_detect_mask_avg] = step
        crack_step = np.zeros_like(self.crack_detect_mask, dtype=int)
        crack_step[self.crack_detect_mask] = step
        step -= 1
        self.crack_width_avg_t = np.zeros((np.sum(self.crack_detect_mask_avg),
                                       len(self.aramis_info.step_list)))

        self.crack_width_t = np.zeros((np.sum(self.crack_detect_mask),
                                       len(self.aramis_info.step_list)))

        self.crack_stress_t = np.zeros((np.sum(self.crack_detect_mask_avg),
                                       len(self.aramis_info.step_list)))
        # plt.figure()
        while step:
            self.evaluated_step_idx = step
            mask = self.crack_filter_avg * self.crack_detect_mask_avg
            crack_step_avg[mask] = step
            mask = self.crack_filter * self.crack_detect_mask
            crack_step[mask] = step
            # if number of cracks = 0 break
            # plt.plot(x, self.d_ux_arr_avg, color='grey')
#             if step == 178:
# #                 print '[asdfasfas[', self.d_ux_arr_avg
#                 plt.plot(x, self.d_ux_arr_avg, 'k-', linewidth=3, zorder=10000)
            # if np.any(mask) == False:
                # print mask
            #    break
            if step == 0:
                break
            self.crack_width_avg_t[:, step] = self.d_ux_arr_avg[crack_step_avg > 0]
            self.crack_width_t[:, step] = self.d_ux_arr[crack_step > 0]
            # y = self.ad_channels_arr[:, :, 2] - self.ad_channels_arr[:, :, 1]
            # self.crack_stress_t[:, step] = y[:, 0][step] * 1e5 / (140 * 60)
            step -= 1
        # y_max_lim = plt.gca().get_ylim()[-1]
#         plt.vlines(x[:-1], [0], self.crack_detect_mask_avg * y_max_lim,
#                    color='magenta', linewidth=1, zorder=10)
#         plt.xlim(0, 527)
#         plt.ylim(0, 0.25)
#         print 'initializationinit_steps time/step of the crack' , crack_step_avg[crack_step_avg > 0]
#
        init_steps_avg = crack_step_avg[crack_step_avg > 0]
        init_steps = crack_step[crack_step > 0]

        self.init_step_avg_lst = init_steps_avg.tolist()
        self.init_step_lst = init_steps.tolist()
#         init_steps_avg.sort()
#         x = np.hstack((0, np.repeat(init_steps_avg, 2), self.aramis_info.last_step))
#         y = np.repeat(np.arange(init_steps_avg.size + 1), 2)
#         print x, y
#
#         plt.figure()
#         plt.title('number of cracks vs. step')
#         plt.plot(x, y)
#
#         plt.figure()
#         x = np.hstack((0, np.repeat(init_steps_avg, 2), self.aramis_info.last_step))
#         x = self.control_strain_t[x]
#         y = np.repeat(np.arange(init_steps_avg.size + 1), 2)
#         plt.title('number of cracks vs. control strain')
#         plt.plot(x, y)

        # plt.show()

    step_idx_max = Property(Int, depends_on='aramis_info.data_dir')
    @cached_property
    def _get_step_idx_max(self):
        return self.aramis_info.number_of_steps - 1


    view = View(
                Item('evaluated_step_idx',
                     editor=RangeEditor(low=0, high_name='step_idx_max', mode='slider'),
                                        springy=True),
                Item('data_array_undeformed_shape', label='data shape', style='readonly'),
                Item('crack_detect_idx'),
                Item('integ_radius'),
                Item('d_ux_threshold'),
                Item('dd_ux_threshold'),
                Item('ddd_ux_threshold'),
                Item('dd_ux_avg_threshold'),
                Item('ddd_ux_avg_threshold'),
                'transform_data',
                Group(
                      'number_of_cracks_t_analyse',
                      UItem('run_t'),
                      UItem('run_back'),
                      ),
                )


if __name__ == '__main__':
    if platform.system() == 'Linux':
        data_dir = r'/media/data/_linux_data/aachen/Aramis_07_2013/TTb-4c-2cm-0-TU-V2_bs4-Xf19s15-Yf19s15'
    elif platform.system() == 'Windows':
        data_dir = r'E:\_linux_data\aachen/Aramis_07_2013/TTb-4c-2cm-0-TU-V2_bs4-Xf19s15-Yf19s15'

    data_dir = '/media/data/_linux_data/aachen/Aramis_07_2013/TT-12c-6cm-0-TU-SH4-V1-Xf19s15-Yf19s15'
    AI = AramisInfo(data_dir=data_dir)
    AC = AramisCDT(aramis_info=AI,
                  integ_radius=1,
                  evaluated_step_idx=129,
                  crack_detect_idx=129,
                  transform_data=True)


    print AC.ux_arr
    AC.configure_traits()
