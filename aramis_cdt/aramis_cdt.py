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
    Instance, DelegatesTo, Button, List, Event, on_trait_change

from etsproxy.traits.ui.api import View, Item, Group, UItem

import numpy as np
import os

import platform
import time
if platform.system() == 'Linux':
    sysclock = time.time
elif platform.system() == 'Windows':
    sysclock = time.clock

from aramis_info import AramisInfo
from aramis_data import AramisData

def get_d(u_arr, a_arr, integ_radius):
    ir = integ_radius
    du_arr = np.zeros_like(u_arr)
    du_arr[:, ir:-ir] = (u_arr[:, 2 * ir:] - u_arr[:, :-2 * ir]) / (a_arr[:, 2 * ir:] - a_arr[:, :-2 * ir])
    return du_arr


class AramisCDT(HasTraits):
    '''Crack Detection Tool for detection of cracks, etc. from Aramis data.
    '''

    aramis_info = Instance(AramisInfo)

    aramis_info_changed = Event
    @on_trait_change('aramis_info.data_dir')
    def aramis_info_change(self):
        if self.aramis_info.data_dir == '':
            print 'Data directory in UI is not defined!'
        else:
            self.aramis_info_changed = True

    aramis_data = Instance(AramisData)

    step_list = DelegatesTo('aramis_info')

    # integration radius for the non-local average of the measured strain
    # defined as integer value of the number of facets (or elements)
    #
    # the value should correspond to
#    def _integ_radius_default(self):
#        return ceil( float( self.n_px_f / self.n_px_a )
    integ_radius = Int(2, params_changed=True)

    crack_detect_idx = Int(params_changed=True)
    '''Index of the step used to determine the initial crack pattern
    '''

    #===========================================================================
    # Crack detection
    #===========================================================================
    d_ux_threshold = Float(0.0, params_changed=True)
    '''The first derivative of displacement in x-direction threshold
    '''

    d_ux_arr = Property(Array, depends_on='aramis_info_changed, aramis_data.+params_changed, +params_changed')
    '''The first derivative of displacement in x-direction
    '''
    @cached_property
    def _get_d_ux_arr(self):
        d_arr = get_d(self.aramis_data.ux_arr, self.aramis_data.x_arr_undeformed, self.integ_radius)
        # cutoff the negative strains - noise
        # d_arr[ d_arr < 0.0 ] = 0.0
        # eliminate displacement jumps smaller then specified tolerance
        # d_arr[ d_arr < self.d_ux_threshold ] = 0.0
        return d_arr

    dd_ux_arr = Property(Array, depends_on='aramis_info_changed, aramis_data.+params_changed, +params_changed')
    @cached_property
    def _get_dd_ux_arr(self):
        return get_d(self.d_ux_arr, self.aramis_data.x_arr_undeformed, self.integ_radius)

    dd_ux_arr_avg = Property(Array, depends_on='aramis_info_changed, aramis_data.+params_changed, +params_changed')
    @cached_property
    def _get_dd_ux_arr_avg(self):
        return np.average(self.dd_ux_arr, axis=0)

    ddd_ux_arr = Property(Array, depends_on='aramis_info_changed, aramis_data.+params_changed, +params_changed')
    @cached_property
    def _get_ddd_ux_arr(self):
        return get_d(self.dd_ux_arr, self.aramis_data.x_arr_undeformed, self.integ_radius)

    ddd_ux_arr_avg = Property(Array, depends_on='aramis_info_changed, aramis_data.+params_changed, +params_changed')
    @cached_property
    def _get_ddd_ux_arr_avg(self):
        return np.average(self.ddd_ux_arr, axis=0)

    d_ux_arr_avg = Property(Array, depends_on='aramis_info_changed, aramis_data.+params_changed, +params_changed')
    '''Average of d_ux in y-direction
    '''
    @cached_property
    def _get_d_ux_arr_avg(self):
        d_u = self.d_ux_arr
        d_avg = np.average(d_u, axis=0)
        return d_avg

    d_uy_arr = Property(Array, depends_on='aramis_info_changed, aramis_data.+params_changed, +params_changed')
    '''The first derivative of displacement in x-direction
    '''
    @cached_property
    def _get_d_uy_arr(self):
        d_arr = get_d(self.aramis_data.uy_arr, self.aramis_data.y_arr_undeformed, self.integ_radius)
        # cutoff the negative strains - noise
        # d_arr[ d_arr < 0.0 ] = 0.0
        # eliminate displacement jumps smaller then specified tolerance
        # d_arr[ d_arr < self.d_ux_threshold ] = 0.0
        return d_arr

    d_uz_arr = Property(Array, depends_on='aramis_info_changed, aramis_data.+params_changed, +params_changed')
    '''The first derivative of displacement in x-direction
    '''
    @cached_property
    def _get_d_uz_arr(self):
        d_arr = get_d(self.aramis_data.uz_arr, self.aramis_data.z_arr_undeformed, self.integ_radius)
        # cutoff the negative strains - noise
        # d_arr[ d_arr < 0.0 ] = 0.0
        # eliminate displacement jumps smaller then specified tolerance
        # d_arr[ d_arr < self.d_ux_threshold ] = 0.0
        return d_arr

    dd_ux_threshold = Float(0.0, params_changed=True)
    '''The second derivative of displacement in x-direction threshold
    '''

    ddd_ux_threshold = Float(-0.005, params_changed=True)
    '''The third derivative of displacement in x-direction threshold
    '''

    dd_ux_avg_threshold = Float(0.0, params_changed=True)
    '''Average of he second derivative of displacement in x-direction threshold
    '''

    ddd_ux_avg_threshold = Float(-0.0005, params_changed=True)
    '''Average the third derivative of displacement in x-direction threshold
    '''

    crack_filter = Property(Array, depends_on='aramis_info_changed, aramis_data.+params_changed, +params_changed')
    @cached_property
    def _get_crack_filter(self):
        dd_ux_arr = self.dd_ux_arr
        ddd_ux_arr = self.ddd_ux_arr
        crack_filter = ((dd_ux_arr[:, 1:] * dd_ux_arr[:, :-1] < self.dd_ux_threshold) *
                        ((ddd_ux_arr[:, 1:] + ddd_ux_arr[:, :-1]) / 2.0 < self.ddd_ux_threshold))
        # print "number of cracks determined by 'crack_filter': ", np.sum(crack_filter, axis=1)
        return crack_filter

    number_of_subcracks = Property(Int, depends_on='aramis_info_changed, aramis_data.+params_changed, +params_changed')
    '''Number of sub-cracks
    '''
    def _get_number_of_subcracks(self):
        return np.sum(self.crack_filter)

    crack_filter_avg = Property(Array, depends_on='aramis_info_changed, aramis_data.+params_changed, +params_changed')
    @cached_property
    def _get_crack_filter_avg(self):
        dd_ux_arr_avg = self.dd_ux_arr_avg
        ddd_ux_arr_avg = self.ddd_ux_arr_avg
        crack_filter_avg = ((dd_ux_arr_avg[1:] * dd_ux_arr_avg[:-1] < self.dd_ux_avg_threshold) *
                           ((ddd_ux_arr_avg[1:] + ddd_ux_arr_avg[:-1]) / 2.0 < self.ddd_ux_avg_threshold))
        print 'crack detection step', self.crack_detect_idx
        print "number of cracks determined by 'crack_filter_avg': ", np.sum(crack_filter_avg)
        return crack_filter_avg

    number_of_cracks_avg = Property(Int, depends_on='aramis_info_changed, aramis_data.+params_changed, +params_changed')
    '''Number of cracks using averaging
    '''
    def _get_number_of_cracks_avg(self):
        return np.sum(self.crack_filter_avg)

    crack_spacing_avg = Property(Array, depends_on='aramis_info_changed, aramis_data.+params_changed, +params_changed')
    @cached_property
    def _get_crack_spacing_avg(self):
        n_cr_avg = np.sum(self.crack_filter_avg)
        s_cr_avg = self.aramis_data.length_x_undeformed / n_cr_avg
        # print "average crack spacing [mm]: %.1f" % (s_cr_avg)
        return s_cr_avg

    crack_arr = Property(Array, depends_on='aramis_info_changed, aramis_data.+params_changed, +params_changed')
    @cached_property
    def _get_crack_arr(self):
        return self.d_ux_arr[np.where(self.crack_filter)]

    crack_arr_mean = Property(Float, depends_on='aramis_info_changed, aramis_data.+params_changed, +params_changed')
    def _get_crack_arr_mean(self):
        return self.crack_arr.mean()

    crack_arr_std = Property(Float, depends_on='aramis_info_changed, aramis_data.+params_changed, +params_changed')
    def _get_crack_arr_std(self):
        return self.crack_arr.std()

    crack_avg_arr = Property(Array, depends_on='aramis_info_changed, aramis_data.+params_changed, +params_changed')
    @cached_property
    def _get_crack_avg_arr(self):
        return self.d_ux_arr_avg[np.where(self.crack_filter_avg)]

    crack_field_arr = Property(Array, depends_on='aramis_info_changed, aramis_data.+params_changed, +params_changed')
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

    number_of_cracks_t_analyse = Bool(True)

    def __number_of_cracks_t(self):
        self.number_of_cracks_t = np.append(self.number_of_cracks_t,
                                           self.number_of_cracks_avg)

    crack_detect_mask_avg = Property(Array, depends_on='crack_detect_idx')
    '''Mask of cracks identified in crack_detect_idx and used for backward
    identification of the crack initialization.
    '''
    @cached_property
    def _get_crack_detect_mask_avg(self):
        self.aramis_data.evaluated_step_idx = self.crack_detect_idx
        return self.crack_filter_avg

    crack_detect_mask = Property(Array, depends_on='crack_detect_idx')
    '''Mask of cracks identified in crack_detect_idx and used for backward
    identification of the crack initialization.
    '''
    @cached_property
    def _get_crack_detect_mask(self):
        self.aramis_data.evaluated_step_idx = self.crack_detect_idx
        return self.crack_filter

    run_t = Button('Run in time')
    '''Run analysis of all steps in time
    '''
    def _run_t_fired(self):
        start_step_idx = self.aramis_data.evaluated_step_idx
        self.number_of_cracks_t = np.array([])
        self.number_of_missing_facets_t = []
        self.control_strain_t = np.array([])

        for step_idx in self.aramis_info.step_idx_list:
            self.aramis_data.evaluated_step_idx = step_idx
            if self.number_of_cracks_t_analyse:
                self.__number_of_cracks_t()
            self.control_strain_t = np.append(self.control_strain_t,
                                            (self.aramis_data.ux_arr[20, -10] - self.aramis_data.ux_arr[20, 10]) /
                                            (self.aramis_data.x_arr_undeformed[20, -10] - self.aramis_data.x_arr_undeformed[20, 10]))
            self.number_of_missing_facets_t.append(np.sum(np.isnan(self.aramis_data.data_array).astype(int)))

        self.aramis_data.evaluated_step_idx = start_step_idx

    run_back = Button('Run back in time')
    '''Run analysis of all steps in time
    '''
    def _run_back_fired(self):
        # todo: better
        x = self.aramis_data.x_arr_undeformed[20, :]

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
            self.aramis_data.evaluated_step_idx = step
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


    view = View(
                Item('crack_detect_idx'),
                Item('d_ux_threshold'),
                Item('dd_ux_threshold'),
                Item('ddd_ux_threshold'),
                Item('dd_ux_avg_threshold'),
                Item('ddd_ux_avg_threshold'),
                Item('integ_radius'),
                Group(
                      'number_of_cracks_t_analyse',
                      UItem('run_t'),
                      UItem('run_back'),
                      ),
                id='aramisCDT.cdt',
                )


if __name__ == '__main__':
    from os.path import expanduser
    home = expanduser("~")

#     data_dir = os.path.join(home, '.simdb_cache', 'aramis', 'TTb-4c-2cm-0-TU-V1_bs4-Xf19s15-Yf19s15')
#
#     AI = AramisInfo(data_dir=data_dir)
#     AD = AramisData(aramis_info=AI)
#     AC = AramisCDT(aramis_info=AI,
#                    aramis_data=AD)

    ac = AramisCDT()
    print ac.__doc__
#    AC.configure_traits()
