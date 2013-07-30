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

from etsproxy.traits.api import \
    HasTraits, Float, Property, cached_property, Int, Array, Bool, \
    Instance, DelegatesTo, Tuple, Button, List

from etsproxy.traits.ui.api import View, Item, HGroup, EnumEditor, Group, UItem

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

class AramisInit(HasTraits):
    '''Data structure for the first step (undeformed state)
    '''
    pass


class AramisStep(HasTraits):
    '''Data structure for the evaluated step
    '''
    pass


class AramisCDT(HasTraits):
    '''Crack Detection Tool for detection of cracks, etc. from Aramis data.
    '''

    aramis_info = Instance(AramisInfo)

    step_list = DelegatesTo('aramis_info')

    #===========================================================================
    # Parameters
    #===========================================================================
    evaluated_step = Int(0, params_changed=True)
    '''Time step used for the evaluation.
    '''

    n_px_facet_size_x = Int(19, params_changed=True)
    '''Size of a facet in x-direction defined in numbers of pixels
    '''

    n_px_facet_distance_x = Int(15, params_changed=True)
    '''Distance between the mid-points of two facets in x-direction
    '''

    n_px_facet_size_y = Int(19, params_changed=True)
    '''Size of a facet in y-direction defined in numbers of pixels
    '''

    n_px_facet_distance_y = Int(15, params_changed=True)
    '''Distance between the mid-points of two facets in y-direction
    '''

    w_detect_step = Int(-1, params_changed=True)
    '''Time step used to determine the crack pattern
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

    #===========================================================================
    # Undeformed state parameters
    #===========================================================================

    input_array_init = Property(Array, depends_on='aramis_info.data_dir')
    '''Array of values for undeformed state in the first step.
    '''
    @cached_property
    def _get_input_array_init(self):
        '''Load data for the first step from *.npy file. If file *.npy does 
        not exist the data is load from *.txt and saved as *.npy. 
        (improve speed of loading)
        '''
        fname = '%s%d' % (self.aramis_info.basename, self.aramis_info.first_step)
        print 'loading', fname, '...'

        fname_txt = os.path.join(self.aramis_info.data_dir, fname + '.txt')
        input_arr = np.loadtxt(fname_txt,
                           skiprows=14,  # not necessary
                           usecols=[0, 1, 2, 3, 4, 8, 9, 10])
        return input_arr

    data_array_init = Property(Array, depends_on='aramis_info.data_dir')
    '''Array of values for undeformed state in the first step.
    '''
    @cached_property
    def _get_data_array_init(self):
        '''Load data for the first step from *.npy file. If file *.npy does 
        not exist the data is load from *.txt and saved as *.npy. 
        (improve speed of loading)
        '''
        return self._prepare_data_structure(self.input_array_init)

    data_array_init_shape = Property(Tuple, depends_on='aramis_info.data_dir')
    '''Shape of undeformed data array.
    '''
    @cached_property
    def _get_data_array_init_shape(self):
        return ((self.input_array_init.shape[1] - 2),
                (self.n_y_init),
                (self.n_x_init))

    data_array_init_mask = Property(Tuple, depends_on='aramis_info.data_dir')
    '''Default mask in undeformed state
    '''
    @cached_property
    def _get_data_array_init_mask(self):
        return np.isnan(self.data_array_init)

    x_idx_min_init = Property(Int, depends_on='aramis_info.data_dir')
    '''Minimum value of the indices in the first column of the undeformed state.
    '''
    @cached_property
    def _get_x_idx_min_init(self):
        return int(np.min(self.input_array_init[:, 0]))

    y_idx_min_init = Property(Int, depends_on='aramis_info.data_dir')
    '''Minimum value of the indices in the second column of the undeformed state.
    '''
    @cached_property
    def _get_y_idx_min_init(self):
        return int(np.min(self.input_array_init[:, 1]))

    x_idx_max_init = Property(Int, depends_on='aramis_info.data_dir')
    '''Maximum value of the indices in the first column of the undeformed state.
    '''
    @cached_property
    def _get_x_idx_max_init(self):
        return int(np.max(self.input_array_init[:, 0]))

    y_idx_max_init = Property(Int, depends_on='aramis_info.data_dir')
    '''Maximum value of the indices in the second column of the undeformed state.
    '''
    @cached_property
    def _get_y_idx_max_init(self):
        return np.max(self.input_array_init[:, 1])

    n_x_init = Property(Int, depends_on='aramis_info.data_dir')
    '''Namber of facets in x-direction
    '''
    @cached_property
    def _get_n_x_init(self):
        return int(self.x_idx_max_init - self.x_idx_min_init + 1)

    n_y_init = Property(Int, depends_on='aramis_info.data_dir')
    '''Number of facets in y-direction
    '''
    @cached_property
    def _get_n_y_init(self):
        return int(self.y_idx_max_init - self.y_idx_min_init + 1)

    x_idx_init = Property(Int, depends_on='aramis_info.data_dir')
    '''Indices in the first column of the undeformed state starting with zero.
    '''
    @cached_property
    def _get_x_idx_init(self):
        return (np.arange(self.n_x_init)[np.newaxis, :] *
                np.ones(self.n_y_init)[:, np.newaxis]).astype(int)

    y_idx_init = Property(Int, depends_on='aramis_info.data_dir')
    '''Indices in the first column of the undeformed state starting with zero.
    '''
    @cached_property
    def _get_y_idx_init(self):
        return (np.arange(self.n_y_init)[np.newaxis, :] *
                np.ones(self.n_x_init)[:, np.newaxis]).T

    x_arr_init = Property(Array, depends_on='aramis_info.data_dir')
    '''Array of x-coordinates in undeformed state
    '''
    @cached_property
    def _get_x_arr_init(self):
        return self.data_array_init[0, :, :]

    y_arr_init = Property(Array, depends_on='aramis_info.data_dir')
    '''Array of y-coordinates in undeformed state
    '''
    @cached_property
    def _get_y_arr_init(self):
        return self.data_array_init[1, :, :]

    z_arr_init = Property(Array, depends_on='aramis_info.data_dir')
    '''Array of z-coordinates in undeformed state
    '''
    @cached_property
    def _get_z_arr_init(self):
        return self.data_array_init[2, :, :]

    length_x_init = Property(Array, depends_on='aramis_info.data_dir')
    '''Length of the specimen in x-direction
    '''
    @cached_property
    def _get_length_x_init(self):
        return np.nanmax(self.x_arr_init) - np.nanmin(self.x_arr_init)

    length_y_init = Property(Array, depends_on='aramis_info.data_dir')
    '''Length of the specimen in y-direction
    '''
    @cached_property
    def _get_length_y_init(self):
        return self.y_arr_init.max() - self.y_arr_init.min()

    length_z_init = Property(Array, depends_on='aramis_info.data_dir')
    '''Length of the specimen in z-direction
    '''
    @cached_property
    def _get_length_z_init(self):
        return self.z_arr_init.max() - self.z_arr_init.min()

    #===========================================================================
    # Data array
    #===========================================================================
    input_array = Property(Array, depends_on='aramis_info.data_dir, +params_changed')
    '''Array of Aramis exported data 
    [index_x, index_y, x_init, y_init, z_init, displ_x, displ_y, displ_z]
    '''
    @cached_property
    def _get_input_array(self):
        return self._load_step_data(self.evaluated_step)

    data_array = Property(Array, depends_on='aramis_info.data_dir, +params_changed')
    '''Array of Aramis exported data 
    [index_x, index_y, x_init, y_init, z_init, displ_x, displ_y, displ_z]
    '''
    @cached_property
    def _get_data_array(self):
        if self.transform_data:

            data_arr = self.input_array
            mask = ~self.data_array_init_mask[0, :, :]
            shape = data_arr.shape

            # rotate counterclockwise about the axis z
            linreg = stats.linregress(self.data_array_init[0, :, :][mask].ravel(),
                                      self.data_array_init[1, :, :][mask].ravel())
            alpha = np.arctan(-linreg[0])
            rot_z = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                            [np.sin(alpha), np.cos(alpha), 0],
                            [0, 0, 1]])

            # rotate counterclockwise about the axis y
            linreg = stats.linregress(self.data_array_init[0, :, :][mask].ravel(),
                                      self.data_array_init[2, :, :][mask].ravel())
            alpha = np.arctan(-linreg[0])
            rot_y = np.array([[np.cos(alpha), 0, -np.sin(alpha)],
                              [0, 1, 0],
                              [np.sin(alpha), 0, np.cos(alpha)]])

            # rotate counterclockwise about the axis x
            linreg = stats.linregress(self.data_array_init[1, :, :][mask].ravel(),
                                      self.data_array_init[2, :, :][mask].ravel())
            alpha = np.arctan(-linreg[0])
            rot_x = np.array([[1, 0, 0],
                            [0, np.cos(alpha), -np.sin(alpha)],
                            [0, np.sin(alpha), np.cos(alpha)]])
            shape = data_arr.shape

            # rotate coordinates
            tmp = np.dot(rot_z, data_arr[:3, :, :].ravel().reshape(3, shape[1] * shape[2]))
            tmp = np.dot(rot_y, tmp)
            data_arr[:3, :, :] = np.dot(rot_x, tmp).reshape(3, shape[1], shape[2])

            # rotate displacements
            tmp = np.dot(rot_z, data_arr[3:, :, :].ravel().reshape(3, shape[1] * shape[2]))
            tmp = np.dot(rot_y, tmp)
            data_arr[3:, :, :] = np.dot(rot_x, tmp).reshape(3, shape[1], shape[2])

            # translation
            data_arr[:3, :, :] -= np.nanmin(np.nanmin(data_arr[:3, :, :], axis=1),
                                            axis=1)[:, np.newaxis, np.newaxis]

            return data_arr
        else:
            return self.input_array

    #===========================================================================
    # Data preparation methods
    #===========================================================================
    def _load_step_data(self, step):
        '''Load data for the specified step from *.npy file. If file *.npy does 
        not exist the data is load from *.txt and saved as *.npy. 
        (improve speed of loading)
        '''
        fname = '%s%d' % (self.aramis_info.basename, step)
        print 'loading', fname, '...'

        start_t = sysclock()
        dir_npy = os.path.join(self.aramis_info.data_dir, 'npy/')
        if os.path.exists(dir_npy) == False:
            os.mkdir(dir_npy)
        fname_npy = os.path.join(dir_npy, fname + '.npy')
        fname_txt = os.path.join(self.aramis_info.data_dir, fname + '.txt')
        if os.path.exists(fname_npy):
            data_arr = np.load(fname_npy)
        else:
            data_arr = np.loadtxt(fname_txt,
                               skiprows=14,  # not necessary
                               usecols=[0, 1, 2, 3, 4, 8, 9, 10])
            data_arr = self._prepare_data_structure(data_arr)

            np.save(fname_npy, data_arr)
        print 'loading time =', sysclock() - start_t
        print 'number of missing facets is', np.sum(np.isnan(data_arr).astype(int))
        return data_arr

    def _prepare_data_structure(self, input_arr):
        data_arr = np.empty((self.n_x_init * self.n_y_init,
                                 self.input_array_init.shape[1] - 2), dtype=float)
        data_arr.fill(np.nan)

        # input indices (columns 1 and 2)
        in_indices = input_arr[:, :2].astype(int)
        in_indices[:, 0] -= self.x_idx_min_init
        in_indices[:, 1] -= self.y_idx_min_init
        in_indices = in_indices.view([('', in_indices.dtype)] * in_indices.shape[1])

        # undeformed state indices
        un_indices = np.hstack((self.x_idx_init.ravel()[:, np.newaxis],
                               self.y_idx_init.ravel()[:, np.newaxis])).astype(int)
        un_indices = un_indices.view([('', un_indices.dtype)] * un_indices.shape[1])

        # data for higher steps have the same order of rows as
        # undeformed one but missing values
        mask = np.in1d(un_indices, in_indices, assume_unique=True)
        data_arr[mask] = input_arr[:, 2:]

        data_arr = data_arr.T.reshape(self.data_array_init_shape)
        return data_arr

    #===========================================================================
    # Geometry arrays
    #===========================================================================
    x_arr = Property(Array, depends_on='aramis_info.data_dir, +params_changed')
    '''Array of x-coordinates
    '''
    @cached_property
    def _get_x_arr(self):
        return self.data_array[0, :, :]

    y_arr = Property(Array, depends_on='aramis_info.data_dir, +params_changed')
    '''Array of y-coordinates
    '''
    @cached_property
    def _get_y_arr(self):
        return self.data_array[1, :, :]

    z_arr = Property(Array, depends_on='aramis_info.data_dir, +params_changed')
    '''Array of z-coordinates
    '''
    @cached_property
    def _get_z_arr(self):
        return self.data_array[2, :, :]

    #===========================================================================
    # Displacement arrays
    #===========================================================================
    ux_arr = Property(Array, depends_on='aramis_info.data_dir, +params_changed')
    '''Array of displacement in x-direction
    '''
    @cached_property
    def _get_ux_arr(self):
        # missing values replaced with average in y-direction
        ux_arr = self.data_array[3, :, :]
        ux_masked = np.ma.masked_array(ux_arr, mask=np.isnan(ux_arr))
        ux_avg = np.ma.average(ux_masked, axis=0)

        y_idx_zeros, x_idx_zeros = np.where(np.isnan(ux_arr))
        ux_arr[y_idx_zeros, x_idx_zeros] = ux_avg[x_idx_zeros]
        return ux_arr

    ux_arr_avg = Property(Array, depends_on='aramis_info.data_dir, +params_changed')
    '''Array of displacement in x-direction
    '''
    @cached_property
    def _get_ux_arr_avg(self):
        # missing values replaced with average in y-direction
        ux_arr = self.ux_arr
        ux_avg = np.average(ux_arr, axis=0)
        return ux_avg

    uy_arr = Property(Array, depends_on='aramis_info.data_dir, +params_changed')
    '''Array of displacement in y-direction
    '''
    @cached_property
    def _get_uy_arr(self):
        return self.data_array[4, :, :]

    uz_arr = Property(Array, depends_on='aramis_info.data_dir, +params_changed')
    '''Array of displacement in z-direction
    '''
    @cached_property
    def _get_uz_arr(self):
        return self.data_array[5, :, :]

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
        s_cr_avg = self.length_x_init / n_cr_avg
        print "average crack spacing [mm]: %.1f" % (s_cr_avg)
        return s_cr_avg

    crack_arr = Property(Array, depends_on='aramis_info.data_dir, +params_changed')
    @cached_property
    def _get_crack_arr(self):
        return self.d_ux_arr[np.where(self.crack_filter)]

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

    ad_channels_lst = List
    '''List of tuples (undeformed, deformed) obtained from AD channels in 
    aramis file header
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
    w_detect_step) when the crack initiate
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
        return np.array(self.ad_channels_lst, dtype=float)

    ad_channels_read = Bool(True)

    number_of_cracks_t_analyse = Bool(True)

    def __number_of_cracks_t(self):
        self.number_of_cracks_t = np.append(self.number_of_cracks_t,
                                           self.number_of_cracks_avg)

    def __decompile_ad_channels(self):
        fname = '%s%d.txt' % (self.aramis_info.basename, self.evaluated_step)
        ad_channels = []
        with open(os.path.join(self.aramis_info.data_dir, fname)) as infile:
            for i in range(30):
                line = infile.readline()
                m = re.match(r'#\s+AD-(\d+):\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)', line)
                if m:
                    ad_channels.append(m.groups())
        self.ad_channels_lst.append(ad_channels)

    run_t = Button('Run in time')
    '''Run analysis of all steps in time
    '''
    def _run_t_fired(self):
        self.number_of_cracks_t = np.array([])
        self.ad_channels_lst = []
        self.number_of_missing_facets_t = []
        self.control_strain_t = np.array([])
        print self.ux_arr.shape
        for step, step_file in zip(self.aramis_info.step_list, self.aramis_info.file_list):
            self.evaluated_step = step
            if self.ad_channels_read:
                self.__decompile_ad_channels()
            if self.number_of_cracks_t_analyse:
                self.__number_of_cracks_t()
            self.control_strain_t = np.append(self.control_strain_t,
                                            (self.ux_arr[20, -10] - self.ux_arr[20, 10]) /
                                            (self.x_arr[20, -10] - self.x_arr[20, 10]))
            self.number_of_missing_facets_t.append(np.sum(np.isnan(self.data_array).astype(int)))
        print 'control_strain_t', self.control_strain_t
#         import matplotlib.pyplot as plt
#         plt.figure()
#         plt.plot(self.control_strain_t)
#         plt.show()


    w_detect_mask_avg = Property(Array, depends_on='w_detect_step')
    '''Mask of cracks identified in w_detect_step and used for backward 
    identification of the crack initialization.
    '''
    @cached_property
    def _get_w_detect_mask_avg(self):
        self.evaluated_step = self.step_list[self.w_detect_step]
        return self.crack_filter_avg

    w_detect_mask = Property(Array, depends_on='w_detect_step')
    '''Mask of cracks identified in w_detect_step and used for backward 
    identification of the crack initialization.
    '''
    @cached_property
    def _get_w_detect_mask(self):
        self.evaluated_step = self.step_list[self.w_detect_step]
        return self.crack_filter

    run_back = Button('Run back in time')
    '''Run analysis of all steps in time
    '''
    def _run_back_fired(self):
        # todo: better
        x = self.x_arr[20, :]
        print x
        # import matplotlib.pyplot as plt
        # plt.rc('font', size=25)
        step = self.step_list[self.w_detect_step]
        crack_step_avg = np.zeros_like(self.w_detect_mask_avg, dtype=int)
        crack_step_avg[self.w_detect_mask_avg] = step
        crack_step = np.zeros_like(self.w_detect_mask, dtype=int)
        crack_step[self.w_detect_mask] = step
        step -= 1
        self.crack_width_avg_t = np.zeros((np.sum(self.w_detect_mask_avg),
                                       len(self.aramis_info.step_list)))

        self.crack_width_t = np.zeros((np.sum(self.w_detect_mask),
                                       len(self.aramis_info.step_list)))
        print 'fasdf', self.crack_width_t.shape
        self.crack_stress_t = np.zeros((np.sum(self.w_detect_mask_avg),
                                       len(self.aramis_info.step_list)))
        # plt.figure()
        while step:
            self.evaluated_step = step
            mask = self.crack_filter_avg * self.w_detect_mask_avg
            crack_step_avg[mask] = step
            mask = self.crack_filter * self.w_detect_mask
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
#         plt.vlines(x[:-1], [0], self.w_detect_mask_avg * y_max_lim,
#                    color='magenta', linewidth=1, zorder=10)
#         plt.xlim(0, 527)
#         plt.ylim(0, 0.25)
#         print 'initializationinit_steps time/step of the crack' , crack_step_avg[crack_step_avg > 0]
#
        init_steps_avg = crack_step_avg[crack_step_avg > 0]
        init_steps = crack_step[crack_step > 0]
#         # crack width
#         plt.figure()
#         # plt.title('crack width vs. control strain')
#         plt.plot(self.control_strain_t * 100, self.crack_width_avg_t.T)
#         plt.xlim(0, 0.71)
#         plt.ylim(0, 0.25)
#         plt.xlabel('control strain [%]')
#         plt.ylabel('crack width [mm]')
#         plt.tight_layout()
#         print self.crack_filter_avg
#         print self.crack_filter_avg.max()
#
#         for i, s in enumerate(init_steps_avg):
#             self.crack_width_avg_t[i, :s] = 0
#         plt.figure()
#         # plt.title('crack width vs. control strain')
#         plt.plot(self.control_strain_t * 100, self.crack_width_avg_t.T)
#         plt.xlim(0, 0.71)
#         plt.ylim(0, 0.25)
#         plt.xlabel('control strain [%]')
#         plt.ylabel('crack width [mm]')
#         plt.tight_layout()
#         print self.crack_filter_avg
#         print self.crack_filter_avg.max()
#
#         plt.figure()
#         plt.title('nominal stress vs crack width')
#         plt.plot(self.crack_width_avg_t.T, self.crack_stress_t.T)
#
#         #
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


    view = View(
                HGroup(
                        Item('evaluated_step',
                             editor=EnumEditor(name='step_list'),
                             springy=True),
                        Item('evaluated_step', style='text', show_label=False,
                             springy=True),
                       ),
                Item('data_array_init_shape', label='data shape', style='readonly'),
                Item('n_px_facet_size_x'),
                Item('n_px_facet_distance_x'),
                Item('n_px_facet_size_y'),
                Item('n_px_facet_distance_y'),
                Item('w_detect_step'),
                Item('integ_radius'),
                Item('d_ux_threshold'),
                Item('dd_ux_threshold'),
                Item('ddd_ux_threshold'),
                Item('dd_ux_avg_threshold'),
                Item('ddd_ux_avg_threshold'),
                'transform_data',
                Group(
                      'number_of_cracks_t_analyse',
                      'ad_channels_read',
                      UItem('run_t'),
                      UItem('run_back'),
                      ),
                )


if __name__ == '__main__':
    if platform.system() == 'Linux':
        aramis_dir = os.path.join(r'/media/data/_linux_data/aachen/ARAMIS_data_IMB/01_ARAMIS_Daten/')
    elif platform.system() == 'Windows':
        aramis_dir = os.path.join(r'E:\_linux_data\aachen\ARAMIS_data_IMB\01_ARAMIS_Daten')
    data_dir = os.path.join(aramis_dir, 'Probe-1-Ausschnitt-Xf15a1-Yf5a4')
    data_dir = '/media/data/_linux_data/aachen/IMB_orig/Probe-2/Probe-2'
    # data_dir = '/media/data/_linux_data/aachen/IMB_orig/Probe-1-final/Probe-1/'
    AI = AramisInfo(data_dir=data_dir)
    ACT = AramisCDT(aramis_info=AI,
                   evaluated_step=400,
                   integ_radius=1,
                   transform_data=True,
                   w_detect_step=472)  # 443

    print ACT.data_array

    print ACT.crack_filter_avg

    ACT._run_t_fired()
    ACT._run_back_fired()

#     view = View(
#                 UItem('aramis_info.data_dir@'),
#                 HGroup(
#                         Item('evaluated_step',
#                              editor=EnumEditor(name='step_list'),
#                              springy=True),
#                         Item('evaluated_step', style='text', show_label=False,
#                              springy=True),
#                        ),
#                 )
