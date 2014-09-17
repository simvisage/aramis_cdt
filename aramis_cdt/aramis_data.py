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

from etsproxy.traits.ui.api import View, Item, HGroup, EnumEditor, Group, UItem, RangeEditor, TextEditor, HTMLEditor

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

def get_d(u_arr, r_arr, integ_radius):
    '''Get the derivatives
    
    Args:
        u_arr: variable to differentiate
        
        r_arr: spatial coordinates
        
        integ_radius: radius at which the normalization is performed
    '''
    ir = integ_radius
    du_arr = np.zeros_like(u_arr)
    du_arr[:, ir:-ir] = (u_arr[:, 2 * ir:] - u_arr[:, :-2 * ir]) / (r_arr[:, 2 * ir:] - r_arr[:, :-2 * ir])
    return du_arr

class InfoViewer(HasTraits):
    text = Str()

    view = View(Item('text', editor=HTMLEditor(format_text=True), style='readonly', show_label=False),
                resizable=True,
                width=0.3,
                height=0.2)

class AramisRawData(HasTraits):
    r'''Basic array structure containing the measured
    aramis data prepred for further elaboration in subclasses
    load data from *.npy files
    '''
    aramis_info = Instance(AramisInfo, params_changed=True)

    current_step = Int(0, params_changed=True, auto_set=False)
    r'''Number of current step for evaluation. Can be set manually or calculated 
    from current_time (the closest step number is selected).
    '''

    current_time = Float(0, params_changed=True, auto_set=False)
    r'''Current time for evaluation. Can be set manually or calculated 
    from current_step.
    '''

    # integration radius for the non-local average of the measured strain
    # defined as integer value of the number of facets (or elements)
    #
    # the value should correspond to
#    def _integ_radius_default(self):
#        return ceil( float( self.n_px_f / self.n_px_a )
    integ_radius = Int(2, params_changed=True)
    '''Integration radius for non-local average of the measured strain defined
    as integer value of number of facets.
    '''

    X = Property(Array, depends_on='aramis_info.+params_changed')
    r'''Cordinates in the initial state formated as an 3D array
    
    .. image:: figs/X.png
            :width: 200px
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

    .. image:: figs/U.png
            :width: 200px
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

    transform_data = Bool(True, params_changed=True)
    '''Switch data transformation before analysis
    '''

    #===========================================================================
    #
    #===========================================================================
    ni = Property(Int, depends_on='aramis_info.+params_changed')
    '''Number of facets in x-direction
    '''
    @cached_property
    def _get_ni(self):
        return self.X.shape[2]

    nj = Property(Int, depends_on='aramis_info.+params_changed')
    '''Number of facets in y-direction
    '''
    @cached_property
    def _get_nj(self):
        return self.X.shape[1]

    i = Property(Array(Int), depends_on='X')
    '''Indices of the facet field starting with zero in x-direction
    '''
    @cached_property
    def _get_i(self):
        return (np.arange(self.ni)[np.newaxis, :] *
                np.ones(self.nj)[:, np.newaxis]).astype(int)

    j = Property(Array(Int), depends_on='X')
    '''Indices of the facet field starting with zero in y-direction
    '''
    @cached_property
    def _get_j(self):
        return (np.arange(self.nj)[np.newaxis, :] *
                np.ones(self.ni)[:, np.newaxis]).T

    #===========================================================================
    # Slider
    #===========================================================================
    i_min = Property(Int, depends_on='X')
    '''Minimum of i
    '''
    @cached_property
    def _get_i_min(self):
        return np.min(self.i).astype(int)

    i_max = Property(Int, depends_on='X')
    '''Maximum of i
    '''
    @cached_property
    def _get_i_max(self):
        return np.max(self.i).astype(int)

    i_min_right = Property(Int, depends_on='X')
    '''Minimum of i
    '''
    @cached_property
    def _get_i_min_right(self):
        return np.min(self.i).astype(int) + 1

    i_max_right = Property(Int, depends_on='X')
    '''Maximum of i
    '''
    @cached_property
    def _get_i_max_right(self):
        return np.max(self.i).astype(int) + 1

    j_min = Property(Int, depends_on='X')
    '''Minimum of j
    '''
    @cached_property
    def _get_j_min(self):
        return np.min(self.j).astype(int)

    j_max = Property(Int, depends_on='X')
    '''Maximum of j
    '''
    @cached_property
    def _get_j_max(self):
        return np.max(self.j).astype(int)

    j_min_bottom = Property(Int, depends_on='X')
    '''Minimum of j
    '''
    @cached_property
    def _get_j_min_bottom(self):
        return np.min(self.j).astype(int) + 1

    j_max_bottom = Property(Int, depends_on='X')
    '''Maximum of j
    '''
    @cached_property
    def _get_j_max_bottom(self):
        return np.max(self.j).astype(int) + 1

    left_i = Int(params_changed=True)
    '''Position (index) of left slider to limit analyzed area from the left.
    '''
    def _left_i_default(self):
        return self.i_min

    right_i = Int(params_changed=True)
    '''Position (index) of right slider to limit analyzed area from the right.
    '''
    def _right_i_default(self):
        return self.i_max

    def _left_i_changed(self):
        if self.left_i >= self.right_i:
            self.right_i = self.left_i + 1

    def _right_i_changed(self):
        if self.right_i <= self.left_i:
            self.left_i = self.right_i - 1

    bottom_j = Int(params_changed=True)
    '''Position (index) of bottom slider to limit analyzed area from the bottom.
    '''
    def _bottom_j_default(self):
        return self.j_max

    top_j = Int(params_changed=True)
    '''Position (index) of top slider to limit analyzed area from the top.
    '''
    def _top_j_default(self):
        return self.j_min

    def _top_j_changed(self):
        if self.top_j >= self.bottom_j:
            self.bottom_j = self.top_j + 1

    def _bottom_j_changed(self):
        if self.bottom_j <= self.top_j:
            self.top_j = self.bottom_j - 1

    i_cut = Property(depends_on='+params_changed')
    '''i values cropped by left and right slider (left_i, right_i)
    '''
    def _get_i_cut(self):
        return self.i[self.top_j:self.bottom_j, self.left_i:self.right_i]

    j_cut = Property(depends_on='+params_changed')
    '''j values cropped by bottom and top slider (bottom_j, top_j)
    '''
    def _get_j_cut(self):
        return self.j[self.top_j:self.bottom_j, self.left_i:self.right_i]


    #===========================================================================
    # Initial state arrays - coordinates
    #===========================================================================
    x_0 = Property(Array, depends_on='aramis_info.+params_changed, +params_changed')
    '''Array of values for initial state in the first step 
    '''
    @cached_property
    def _get_x_0(self):
        X = self.X[:, self.top_j:self.bottom_j, self.left_i:self.right_i]
        if self.transform_data:
            # move to 0,0
            X[0, :, :] = X[0, :, :] - np.min(X[0, :, :])
            X[1, :, :] = X[1, :, :] - np.min(X[1, :, :])
            return X
        else:
            return X

    x_0_mask = Property(Tuple, depends_on='+params_changed')
    '''Default mask in initial state
    '''
    @cached_property
    def _get_x_0_mask(self):
        return np.isnan(self.x_0)

    x_arr_0 = Property(Array, depends_on='aramis_info.+params_changed, +params_changed')
    '''Array of x-coordinates in undeformed state
    '''
    @cached_property
    def _get_x_arr_0(self):
        return self.x_0[0, :, :]

    y_arr_0 = Property(Array, depends_on='aramis_info.+params_changed, +params_changed')
    '''Array of y-coordinates in undeformed state
    '''
    @cached_property
    def _get_y_arr_0(self):
        return self.x_0[1, :, :]

    z_arr_0 = Property(Array, depends_on='aramis_info.+params_changed, +params_changed')
    '''Array of z-coordinates in undeformed state
    '''
    @cached_property
    def _get_z_arr_0(self):
        return self.x_0[2, :, :]

    #------------- measuring field parameters in initial state

    x_0_shape = Property(Tuple, depends_on='X')
    '''Shape of undeformed data array.
    '''
    @cached_property
    def _get_x_0_shape(self):
        if self.aramis_info.data_dir == '':
            return (0, 0, 0)
        else:
            return (3, self.nj, self.ni)

    lx_0 = Property(Float, depends_on='aramis_info.+params_changed, +params_changed')
    '''Length of the measuring area in x-direction
    '''
    @cached_property
    def _get_lx_0(self):
        return np.nanmax(self.x_arr_0) - np.nanmin(self.x_arr_0)

    ly_0 = Property(Float, depends_on='aramis_info.+params_changed, +params_changed')
    '''Length of the measuring area in y-direction
    '''
    @cached_property
    def _get_ly_0(self):
        return self.y_arr_0.max() - self.y_arr_0.min()

    lz_0 = Property(Float, depends_on='aramis_info.+params_changed, +params_changed')
    '''Length of the measuring area in z-direction
    '''
    @cached_property
    def _get_lz_0(self):
        return self.z_arr_0.max() - self.z_arr_0.min()

    x_0_stats = Property(depends_on='aramis_info.+params_changed, +params_changed')
    '''
    * mu_mm - mean value of facet midpoint distance [mm]
    * std_mm - standard deviation of facet midpoint distance [mm]
    * mu_px_mm - mean value of one pixel size [mm]
    * std_px_mm - standard deviation of one pixel size [mm]
    '''
    @cached_property
    def _get_x_0_stats(self):
        x_diff = np.diff(self.x_arr_0, axis=1)
        mu_mm = np.mean(x_diff)
        std_mm = np.std(x_diff)
        mu_px_mm = np.mean(x_diff / self.aramis_info.n_px_facet_step_x)
        std_px_mm = np.std(x_diff / self.aramis_info.n_px_facet_step_x)
        return mu_mm, std_mm, mu_px_mm, std_px_mm

    stats_str = Property(Str, depends_on='aramis_info.+params_changed, +params_changed')
    @cached_property
    def _get_stats_str(self):
        t = '''<b>Statistics:</b>
        <table>
        <tr><td>L_x:</td> <td> %.3g mm </td></tr>
        <tr><td>L_y:</td> <td> %.3g mm </td></tr>
        <tr><td>L_z:</td> <td> %.3g mm </td></tr>
        <tr><td>Facet distance (mean):</td> <td> %.3g mm </td></tr>
        <tr><td>Facet distance (std):</td> <td> %.3g mm </td></tr>
        <tr><td>Pixel size (mean):</td> <td> %.3g mm </td></tr>
        <tr><td>Pixel size (std):</td> <td> %.3g mm </td></tr>
        <tr><td>Data shape:</td> <td> %s </td></tr>
        </table>
        ''' % (self.lx_0,
               self.ly_0,
               self.lz_0,
               self.x_0_stats[0],
               self.x_0_stats[1],
               self.x_0_stats[2],
               self.x_0_stats[3],
               str(self.x_0_shape))
        return t

    show_stats = Button
    def _show_stats_fired(self):
        InfoViewer(text=self.stats_str).configure_traits()

    #===========================================================================
    # Displacement arrays
    #===========================================================================
    u = Property(Array, depends_on='aramis_info.+params_changed, +params_changed')
    '''Array of displacements for current step
    '''
    @cached_property
    def _get_u(self):
        U = self.U[:, self.top_j:self.bottom_j, self.left_i:self.right_i]
        if self.transform_data:
            return U
        else:
            return U

    ux_arr = Property(Array, depends_on='aramis_info.+params_changed, +params_changed')
    '''Array of displacements in x-direction
    '''
    @cached_property
    def _get_ux_arr(self):
        # missing values replaced by averaged values in y-direction
        ux_arr = self.u[0, :, :]
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
        return self.u[1, :, :]

    uz_arr = Property(Array, depends_on='aramis_info.+params_changed, +params_changed')
    '''Array of displacements in z-direction
    '''
    @cached_property
    def _get_uz_arr(self):
        return self.u[2, :, :]

    #===========================================================================
    #
    #===========================================================================

    d_ux = Property(Array, depends_on='+params_changed')
    '''Strain 1D in x-direction
    '''
    @cached_property
    def _get_d_ux(self):
        return get_d(self.ux_arr, self.x_arr_0, self.integ_radius)

    dd_ux = Property(Array, depends_on='+params_changed')
    @cached_property
    def _get_dd_ux(self):
        return get_d(self.d_ux, self.x_arr_0, self.integ_radius)

    dd_ux_avg = Property(Array, depends_on='+params_changed')
    @cached_property
    def _get_dd_ux_avg(self):
        return np.average(self.dd_ux, axis=0)

    ddd_ux = Property(Array, depends_on='+params_changed')
    @cached_property
    def _get_ddd_ux(self):
        return get_d(self.dd_ux, self.x_arr_0, self.integ_radius)

    ddd_ux_avg = Property(Array, depends_on='+params_changed')
    @cached_property
    def _get_ddd_ux_avg(self):
        return np.average(self.ddd_ux, axis=0)

    d_ux_avg = Property(Array, depends_on='+params_changed')
    '''Average of d_ux in y-direction
    '''
    @cached_property
    def _get_d_ux_avg(self):
        d_u = self.d_ux
        d_avg = np.average(d_u, axis=0)
        return d_avg

    d_uy = Property(Array, depends_on='+params_changed')
    '''Strain 1D in y-direction
    '''
    @cached_property
    def _get_d_uy(self):
        return get_d(self.uy_arr, self.y_arr_0, self.integ_radius)

    d_uz = Property(Array, depends_on='+params_changed')
    '''Strain 1D in z-direction
    '''
    @cached_property
    def _get_d_uz(self):
        return get_d(self.uz_arr, self.z_arr_0, self.integ_radius)


    step_times = Property(Array, depends_on='aramis_info.+params_changed')
    '''Capture time of each step
    '''
    @cached_property
    def _get_step_times(self):
        return self.ad_channels_arr[:, 0]

    step_times_min = Property(Float, depends_on='aramis_info.+params_changed')
    '''Minimum value of capture time
    '''
    def _get_step_times_min(self):
        return np.min(self.step_times)

    step_times_max = Property(Float, depends_on='aramis_info.+params_changed')
    '''Maximum value of capture time
    '''
    def _get_step_times_max(self):
        return np.max(self.step_times)

    def _current_time_changed(self):
        self.on_trait_change(self._current_time_changed, 'current_step', remove=True)
        self.current_step = np.abs(self.step_times - self.current_time).argmin()
        self.on_trait_change(self._current_time_changed, 'current_step')

    def _current_step_changed(self):
        self.on_trait_change(self._current_time_changed, 'current_time', remove=True)
        self.current_time = self.step_times[self.current_step]
        self.on_trait_change(self._current_time_changed, 'current_time')

    step_max = Property(Int, depends_on='aramis_info.+params_changed')
    '''Maximum step number
    '''
    @cached_property
    def _get_step_max(self):
        return self.aramis_info.number_of_steps - 1


    view = View(
                Item('current_step',
                     editor=RangeEditor(low=0, high_name='step_max', mode='slider', label_width=35),
                                        springy=True),
                Item('current_time',
                     editor=RangeEditor(low_name='step_times_min', high_name='step_times_max', mode='slider', format='%.1f', label_width=35),
                                        springy=True),
                Item('integ_radius'),
                HGroup(Group(Item('left_i',
                                   editor=RangeEditor(low_name='i_min',
                                                         high_name='i_max',
                                                         format='%d',
                                                         label_width=28,
                                                         mode='slider')),
                            Item('right_i',
                               editor=RangeEditor(low_name='i_min_right',
                                                     high_name='i_max_right',
                                                     format='%d',
                                                     label_width=28,
                                                     mode='slider')),
                             ),
                       Group(Item('top_j',
                               editor=RangeEditor(low_name='j_min',
                                                     high_name='j_max',
                                                     format='%d',
                                                     label_width=28,
                                                     mode='slider')),
                             Item('bottom_j',
                                   editor=RangeEditor(low_name='j_min_bottom',
                                                         high_name='j_max_bottom',
                                                         format='%d',
                                                         label_width=28,
                                                         mode='slider')),
                             ),
                       ),
                'show_stats',
                id='aramisCDT.data',
                )

