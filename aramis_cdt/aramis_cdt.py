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
from aramis_data import AramisFieldData

class AramisCDT(HasTraits):
    '''Crack Detection Tool for detection of cracks, etc. from Aramis data.
    '''
    aramis_info = Instance(AramisInfo, params_changed=True)

    aramis_data = Instance(AramisFieldData, params_changed=True)

    number_of_steps = DelegatesTo('aramis_info')

    crack_detection_step = Int(params_changed=True)
    '''Index of the step used to determine the crack pattern
    '''

    #===========================================================================
    # Thresholds
    #===========================================================================
    d_ux_threshold = Float(0.0, params_changed=True)
    '''The first derivative of displacement in x-direction threshold
    '''

    dd_ux_threshold = Float(0.0, params_changed=True)
    '''The second derivative of displacement in x-direction threshold
    '''

    ddd_ux_threshold = Float(-1e-4, params_changed=True)
    '''The third derivative of displacement in x-direction threshold
    '''

    d_ux_avg_threshold = Float(0.0, params_changed=True)
    '''The first derivative of displacement in x-direction threshold
    '''

    dd_ux_avg_threshold = Float(0.0, params_changed=True)
    '''Average of he second derivative of displacement in x-direction threshold
    '''

    ddd_ux_avg_threshold = Float(-1e-4, params_changed=True)
    '''Average of the third derivative of displacement in x-direction threshold
    '''

    #===========================================================================
    # Crack detection
    #===========================================================================
    crack_filter = Property(Array, depends_on='aramis_data.+params_changed, +params_changed')
    @cached_property
    def _get_crack_filter(self):
        dd_ux = self.aramis_data.dd_ux
        ddd_ux = self.aramis_data.ddd_ux
        crack_filter = ((dd_ux[:, 1:] * dd_ux[:, :-1] < self.dd_ux_threshold) *
                        ((ddd_ux[:, 1:] + ddd_ux[:, :-1]) / 2.0 < self.ddd_ux_threshold))
        # print "number of cracks determined by 'crack_filter': ", np.sum(crack_filter, axis=1)
        return crack_filter

    number_of_subcracks = Property(Int, depends_on='aramis_data.+params_changed, +params_changed')
    '''Number of sub-cracks
    '''
    def _get_number_of_subcracks(self):
        return np.sum(self.crack_filter)

    crack_filter_avg = Property(Array, depends_on='aramis_data.+params_changed, +params_changed')
    @cached_property
    def _get_crack_filter_avg(self):
        dd_ux_avg = self.aramis_data.dd_ux_avg
        ddd_ux_avg = self.aramis_data.ddd_ux_avg
        crack_filter_avg = ((dd_ux_avg[1:] * dd_ux_avg[:-1] < self.dd_ux_avg_threshold) *
                           ((ddd_ux_avg[1:] + ddd_ux_avg[:-1]) / 2.0 < self.ddd_ux_avg_threshold))
        print 'crack detection step', self.crack_detection_step
        print "number of cracks determined by 'crack_filter_avg': ", np.sum(crack_filter_avg)
        return crack_filter_avg

    number_of_cracks_avg = Property(Int, depends_on='aramis_data.+params_changed, +params_changed')
    '''Number of cracks using averaging
    '''
    def _get_number_of_cracks_avg(self):
        return np.sum(self.crack_filter_avg)

    crack_spacing_avg = Property(Array, depends_on='aramis_data.+params_changed, +params_changed')
    @cached_property
    def _get_crack_spacing_avg(self):
        n_cr_avg = np.sum(self.crack_filter_avg)
        if n_cr_avg > 0:
            s_cr_avg = self.aramis_data.lx_0 / n_cr_avg
            # print "average crack spacing [mm]: %.1f" % (s_cr_avg)
            return s_cr_avg

    crack_arr = Property(Array, depends_on='aramis_data.+params_changed, +params_changed')
    @cached_property
    def _get_crack_arr(self):
        from aramis_data import get_d2
        return (get_d2(self.aramis_data.ux_arr, self.aramis_data.integ_radius))[np.where(self.crack_filter)]
        return self.aramis_data.d_ux[np.where(self.crack_filter)]

    crack_arr_mean = Property(Float, depends_on='aramis_data.+params_changed, +params_changed')
    def _get_crack_arr_mean(self):
        return self.crack_arr.mean()

    crack_arr_std = Property(Float, depends_on='aramis_data.+params_changed, +params_changed')
    def _get_crack_arr_std(self):
        return self.crack_arr.std()

    crack_avg_arr = Property(Array, depends_on='aramis_data.+params_changed, +params_changed')
    @cached_property
    def _get_crack_avg_arr(self):
        return self.aramis_data.d_ux_avg[np.where(self.crack_filter_avg)]

    crack_field_arr = Property(Array, depends_on='aramis_data.+params_changed, +params_changed')
    @cached_property
    def _get_crack_field_arr(self):
        cf_w = np.zeros_like(self.aramis_data.d_ux)
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
    crack_detection_step) when the crack initiate
    '''
    init_step_lst = List()

    crack_width_avg_t = Array
    crack_stress_t = Array

    number_of_cracks_t_analyse = Bool(True)

    def __number_of_cracks_t(self):
        self.number_of_cracks_t = np.append(self.number_of_cracks_t,
                                           self.number_of_cracks_avg)

    crack_detect_mask_avg = Property(Array, depends_on='crack_detection_step')
    '''Mask of cracks identified in crack_detection_step and used for backward
    identification of the crack initialization.
    '''
    @cached_property
    def _get_crack_detect_mask_avg(self):
        self.aramis_data.current_step = self.crack_detection_step
        return self.crack_filter_avg

    crack_detect_mask = Property(Array, depends_on='crack_detection_step')
    '''Mask of cracks identified in crack_detection_step and used for backward
    identification of the crack initialization.
    '''
    @cached_property
    def _get_crack_detect_mask(self):
        self.aramis_data.current_step = self.crack_detection_step
        return self.crack_filter

    run_t = Button('Run in time')
    '''Run analysis of all steps in time
    '''
    def _run_t_fired(self):
        start_step_idx = self.aramis_data.current_step
        self.number_of_cracks_t = np.array([])
        self.number_of_missing_facets_t = []
        self.control_strain_t = np.array([])

        for step_idx in self.aramis_info.step_list:
            self.aramis_data.current_step = step_idx
            if self.number_of_cracks_t_analyse:
                self.__number_of_cracks_t()
            self.control_strain_t = np.append(self.control_strain_t,
                                            (self.aramis_data.ux_arr[20, -10] - self.aramis_data.ux_arr[20, 10]) /
                                            (self.aramis_data.x_arr_0[20, -10] - self.aramis_data.x_arr_0[20, 10]))
            self.number_of_missing_facets_t.append(np.sum(np.isnan(self.aramis_data.data_array).astype(int)))

        self.aramis_data.current_step = start_step_idx

    run_back = Button('Run back in time')
    '''Run analysis of all steps in time
    '''
    def _run_back_fired(self):
        step = self.crack_detection_step
        crack_step_avg = np.zeros_like(self.crack_detect_mask_avg, dtype=int)
        crack_step_avg[self.crack_detect_mask_avg] = step

        m = self.crack_detect_mask_avg.copy()
        step -= 1

        while step:
            if step == 0:
                break
            print step
            self.aramis_data.current_step = step
            d_ux_avg_mask = (self.aramis_data.d_ux_avg[1:] + self.aramis_data.d_ux_avg[:-1]) * 0.5 > self.d_ux_avg_threshold
            mask = d_ux_avg_mask * m
            crack_step_avg[mask] = step

            if np.sum(mask) == 0:
                break

            step -= 1

        init_steps_avg = crack_step_avg[crack_step_avg > 0]

        self.init_step_avg_lst = init_steps_avg.tolist()
        print self.init_step_avg_lst
        position_index = np.argwhere(self.crack_detect_mask_avg).flatten()
        print 'position index', position_index
        position = self.aramis_data.x_arr_0[0, :][self.crack_detect_mask_avg]
        print 'position', position
        time_of_init = self.aramis_data.step_times[self.init_step_avg_lst]
        print 'time of initiation', time_of_init
        force = self.aramis_data.ad_channels_arr[:, 1][self.init_step_avg_lst]
        print 'force', force

        data_to_save = np.vstack((position_index, position, time_of_init, force)).T

        np.savetxt('%s.txt' % self.aramis_info.specimen_name, data_to_save, delimiter=';',
                   header='position_index; position; time_of_init; force')

        '''
        # todo: better
        # x = self.aramis_data.x_arr_0[20, :]

        # import matplotlib.pyplot as plt
        # plt.rc('font', size=25)
        step = self.crack_detection_step
        crack_step_avg = np.zeros_like(self.crack_detect_mask_avg, dtype=int)
        crack_step_avg[self.crack_detect_mask_avg] = step
        crack_step = np.zeros_like(self.crack_detect_mask, dtype=int)
        crack_step[self.crack_detect_mask] = step
        step -= 1
        self.crack_width_avg_t = np.zeros((np.sum(self.crack_detect_mask_avg),
                                       self.number_of_steps))

        self.crack_width_t = np.zeros((np.sum(self.crack_detect_mask),
                                       self.number_of_steps))

        self.crack_stress_t = np.zeros((np.sum(self.crack_detect_mask_avg),
                                       self.number_of_steps))
        # plt.figure()
        m = self.crack_detect_mask_avg.copy()
#         idx1 = np.argwhere(m == True) - 1
#         idx1 = np.delete(idx1, np.argwhere(idx1 < 0))
#         m[idx1] = True
#         idx2 = np.argwhere(m == True) + 1
#         idx2 = np.delete(idx1, np.argwhere(idx2 >= m.shape[0]))
#         m[idx2] = True
        while step:
            print 'step', step
            self.aramis_data.current_step = step
            mask = self.crack_filter_avg  # * m
            crack_step_avg[mask] = step
            mask = self.crack_filter * self.crack_detect_mask
            crack_step[mask] = step
            # if number of cracks = 0 break
            # plt.plot(x, self.aramis_data.d_ux_avg, color='grey')
#             if step == 178:
# #                 print '[asdfasfas[', self.aramis_data.d_ux_avg
#                 plt.plot(x, self.aramis_data.d_ux_avg, 'k-', linewidth=3, zorder=10000)
            # if np.any(mask) == False:
                # print mask
            #    break
            if step == 0:
                break
            # self.crack_width_avg_t[:, step] = self.aramis_data.d_ux_avg[crack_step_avg > 0]
            # self.crack_width_t[:, step] = self.aramis_data.d_ux[crack_step > 0]
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
        print 'finished'

#         for i in idx1:
#             if crack_step_avg[i] < crack_step_avg[i + 1]:
#                 crack_step_avg[i] = crack_step_avg[i + 1]
#             crack_step_avg[i] = 0
#         for i in idx2:
#             if crack_step_avg[i] < crack_step_avg[i - 1]:
#                 crack_step_avg[i] = crack_step_avg[i - 1]
#             crack_step_avg[i] = 0
        init_steps_avg = crack_step_avg[crack_step_avg > 0]
        init_steps = crack_step[crack_step > 0]

        self.init_step_avg_lst = init_steps_avg.tolist()
        self.init_step_lst = init_steps.tolist()
        print self.init_step_avg_lst
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
        '''

    view = View(
                Item('crack_detection_step'),
                Item('d_ux_threshold'),
                Item('dd_ux_threshold'),
                Item('ddd_ux_threshold'),
                Item('d_ux_avg_threshold'),
                Item('dd_ux_avg_threshold'),
                Item('ddd_ux_avg_threshold'),
                Group(
                      'number_of_cracks_t_analyse',
                      UItem('run_t'),
                      UItem('run_back'),
                      ),
                id='aramisCDT.cdt',
                )
