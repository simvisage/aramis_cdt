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

    n_px_facette_size_x = Int(19, params_changed=True)
    '''Size of a facet in x-direction defined in numbers of pixels
    '''

    n_px_facette_distance_x = Int(15, params_changed=True)
    '''Distance between the mid-points of two facets in x-direction
    '''

    n_px_facette_size_y = Int(19, params_changed=True)
    '''Size of a facet in y-direction defined in numbers of pixels
    '''

    n_px_facette_distance_y = Int(15, params_changed=True)
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

    #===========================================================================
    # Undeformed state parameters
    #===========================================================================
    # TODO: length of header change depending on number of AD-channels (or other additional informations?)
    # if the header line starts with # than this is not necessary
    header_length = Int

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
                np.ones(self.n_y_init)[:, np.newaxis])

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
        return self.data_array_init[2, :, :]

    y_arr_init = Property(Array, depends_on='aramis_info.data_dir')
    '''Array of y-coordinates in undeformed state
    '''
    @cached_property
    def _get_y_arr_init(self):
        return self.data_array_init[3, :, :]

    z_arr_init = Property(Array, depends_on='aramis_info.data_dir')
    '''Array of z-coordinates in undeformed state
    '''
    @cached_property
    def _get_z_arr_init(self):
        return self.data_array_init[4, :, :]

    length_x_init = Property(Array, depends_on='aramis_info.data_dir')
    '''Length of the specimen in x-direction
    '''
    @cached_property
    def _get_length_x_init(self):
        return self.x_arr_init.max() - self.x_arr_init.min()

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

    transform_data = Bool(False, params_changed=True)

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
            shape = data_arr.shape

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

    d_ux_tol = Float(0.00)

    d_ux_arr = Property(Array, depends_on='aramis_info.data_dir, +params_changed')
    '''The first derivative of displacement in x-direction
    '''
    @cached_property
    def _get_d_ux_arr(self):
        d_arr = get_d(self.ux_arr, self.integ_radius)
        # cutoff the negative strains - noise
        d_arr[ d_arr < 0.0 ] = 0.0
        # eliminate displacement jumps smaller then specified tolerance
        d_arr[ d_arr < self.d_ux_tol ] = 0.0
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
        d_arr[ d_arr < self.d_ux_tol ] = 0.0
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
        d_arr[ d_arr < self.d_ux_tol ] = 0.0
        return d_arr

    #===========================================================================
    # Crack detection
    #===========================================================================
    crack_filter = Property(Array, depends_on='aramis_info.data_dir, +params_changed')
    @cached_property
    def _get_crack_filter(self):
        dd_ux_arr = self.dd_ux_arr
        ddd_ux_arr = self.ddd_ux_arr
        return ((dd_ux_arr[:, 1:] * dd_ux_arr[:, :-1] < 0.0) *
                ((ddd_ux_arr[:, 1:] + ddd_ux_arr[:, :-1]) / 2.0 < -0.005))

    crack_filter_avg = Property(Array, depends_on='aramis_info.data_dir, +params_changed')
    @cached_property
    def _get_crack_filter_avg(self):
        dd_ux_arr_avg = self.dd_ux_arr_avg
        ddd_ux_arr_avg = self.ddd_ux_arr_avg
        crack_filter_avg = ((dd_ux_arr_avg[1:] * dd_ux_arr_avg[:-1] < 0.0) *
                           ((ddd_ux_arr_avg[1:] + ddd_ux_arr_avg[:-1]) / 2.0 < -0.01))
        print "number of cracks determined by 'crack_filter_avg': ", np.sum(crack_filter_avg)
        return crack_filter_avg

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

    ad_channels_arr = Property(Array)
    def _get_ad_channels_arr(self):
        return np.array(self.ad_channels_lst, dtype=float)

    ad_channels_read = Bool(True)

    number_of_cracks_t_analyse = Bool(True)

    def __number_of_cracks_t(self):
        self.number_of_cracks_t = np.append(self.number_of_cracks_t,
                                           np.sum(self.crack_filter_avg))

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
        for step, step_file in zip(self.aramis_info.step_list, self.aramis_info.file_list):
            self.evaluated_step = step
            if self.ad_channels_read:
                self.__decompile_ad_channels()
            if self.number_of_cracks_t_analyse:
                self.__number_of_cracks_t()

    view = View(
                HGroup(
                        Item('evaluated_step',
                             editor=EnumEditor(name='step_list'),
                             springy=True),
                        Item('evaluated_step', style='text', show_label=False,
                             springy=True),
                       ),
                Item('data_array_init_shape', label='data shape', style='readonly'),
                Item('n_px_facette_size_x'),
                Item('n_px_facette_distance_x'),
                Item('n_px_facette_size_y'),
                Item('n_px_facette_distance_y'),
                Item('w_detect_step'),
                Item('integ_radius'),
                'transform_data',
                Group(
                      'number_of_cracks_t_analyse',
                      'ad_channels_read',
                      UItem('run_t'),
                      ),
                )


if __name__ == '__main__':
    if platform.system() == 'Linux':
        aramis_dir = os.path.join(r'/media/data/_linux_data/aachen/ARAMIS_data_IMB/01_ARAMIS_Daten/')
    elif platform.system() == 'Windows':
        aramis_dir = os.path.join(r'E:\_linux_data\aachen\ARAMIS_data_IMB\01_ARAMIS_Daten')
    data_dir = os.path.join(aramis_dir, 'Probe-1-Ausschnitt-Xf15a1-Yf5a4')
    AI = AramisInfo(data_dir='/media/data/_linux_data/aachen/IMB_orig/Probe-2/Probe-2')
    ACT = AramisCDT(aramis_info=AI,
                   evaluated_step=400)

    print ACT.data_array
    ACT.transform_data = True
    print ACT.data_array

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
