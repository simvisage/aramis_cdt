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
    HasTraits, Instance, Button, Bool, Directory

from etsproxy.traits.ui.api import UItem, View, Item, RangeEditor

import platform
import time
if platform.system() == 'Linux':
    sysclock = time.time
elif platform.system() == 'Windows':
    sysclock = time.clock

import os
import tempfile
import numpy as np

from aramis_cdt import AramisCDT
from aramis_info import AramisInfo

import matplotlib
# matplotlib.use('Qt4Agg')
matplotlib.use('WxAgg')
import matplotlib.pyplot as plt


class AramisPlot2D(HasTraits):
    '''This class manages 2D views for AramisCDT variables
    '''

    aramis_info = Instance(AramisInfo)

    aramis_cdt = Instance(AramisCDT)

    save_plot = Bool(False)

    show_plot = Bool(True)

    save_dir = Directory
    def _save_dir_default(self):
        return os.getcwd()

    temp_dir = Directory
    def _temp_dir_default(self):
        return tempfile.mkdtemp()

    plot2d = Button
    def _plot2d_fired(self):
        aramis_cdt = self.aramis_cdt
        plt.figure()
        plt.subplot(2, 2, 1)

        plt.plot(aramis_cdt.x_idx_undeformed.T, aramis_cdt.d_ux_arr.T, color='black')
        plt.plot(aramis_cdt.x_idx_undeformed[0, :], aramis_cdt.d_ux_arr_avg, color='red', linewidth=2)
        y_max_lim = plt.gca().get_ylim()[-1]
        plt.vlines(aramis_cdt.x_idx_undeformed[0, :-1], [0], aramis_cdt.crack_filter_avg * y_max_lim,
                 color='magenta', linewidth=1, zorder=10)

        plt.subplot(2, 2, 2)
        plt.plot(aramis_cdt.x_idx_undeformed.T, aramis_cdt.ux_arr.T, color='green')
        plt.plot(aramis_cdt.x_idx_undeformed[0, :], aramis_cdt.ux_arr_avg, color='red', linewidth=2)

        plt.subplot(2, 2, 3)
        plt.plot(aramis_cdt.x_idx_undeformed[0, :], aramis_cdt.dd_ux_arr_avg, color='black')
        plt.plot(aramis_cdt.x_idx_undeformed[0, :], aramis_cdt.ddd_ux_arr_avg, color='blue')
        y_max_lim = plt.gca().get_ylim()[-1]
        plt.vlines(aramis_cdt.x_idx_undeformed[0, :-1], [0], aramis_cdt.crack_filter_avg * y_max_lim,
                 color='magenta', linewidth=1, zorder=10)

        plt.subplot(2, 2, 4)

        plt.suptitle(aramis_cdt.aramis_info.specimen_name + ' - %d' % aramis_cdt.evaluated_step_idx)

        aramis_cdt.crack_spacing_avg

        if self.save_plot:
            plt.savefig(os.path.join(self.save_dir, 'plot2d.png'))

        if self.show_plot:
            plt.show()


    plot_crack_hist = Button
    def _plot_crack_hist_fired(self):

        aramis_cdt = self.aramis_cdt
        plt.figure()

        plt.hist(aramis_cdt.crack_arr, bins=40, normed=True)

        plt.xlabel('crack_width [mm]')
        plt.ylabel('frequency [-]')

        plt.twinx()
        plt.hist(aramis_cdt.crack_arr, normed=True,
                 histtype='step', color='black',
                 cumulative=True, bins=40, linewidth=2)
        plt.ylabel('probability [-]')

        plt.title(aramis_cdt.aramis_info.specimen_name + ' - %d' % aramis_cdt.evaluated_step_idx)

        if self.save_plot:
            plt.savefig(os.path.join(self.save_dir, 'crack_hist.png'))

        if self.show_plot:
            plt.show()

    plot_number_of_cracks_t = Button
    def _plot_number_of_cracks_t_fired(self):

        aramis_cdt = self.aramis_cdt
        plt.figure()

        plt.plot(aramis_cdt.control_strain_t, aramis_cdt.number_of_cracks_t)
        # self.aramis_info.step_idx_list

        plt.xlabel('control strain')  # 'step number'
        plt.ylabel('number of cracks')
        # plt.title('cracks in time')

        if self.show_plot:
            plt.show()

    plot_force_step = Button
    def _plot_force_step_fired(self):

        aramis_cdt = self.aramis_cdt
        plt.figure()

        plt.title('')
        plt.plot(self.aramis_info.step_idx_list, aramis_cdt.force_t)

        plt.xlabel('step')
        plt.ylabel('force_t [kN]')

        if self.save_plot:
            plt.savefig(os.path.join(self.save_dir, 'force_time.png'))

        if self.show_plot:
            plt.show()

    plot_stress_strain = Button
    def _plot_stress_strain_fired(self):

        aramis_cdt = self.aramis_cdt
        plt.figure()

        plt.title('')
        plt.plot(aramis_cdt.control_strain_t, aramis_cdt.stress_t)

        plt.xlabel('control strain [-]')
        plt.ylabel('nominal stress [MPa]')

        if self.save_plot:
            plt.savefig(os.path.join(self.save_dir, 'stress_strain.png'))

        if self.show_plot:
            plt.show()

    plot_w_strain = Button
    def _plot_w_strain_fired(self):

        aramis_cdt = self.aramis_cdt
        plt.figure()

        plt.title('')
        plt.plot(aramis_cdt.control_strain_t,
                 aramis_cdt.crack_width_avg_t.T)

        plt.xlabel('control strain [-]')
        plt.ylabel('crack width [mm]')

        if self.save_plot:
            plt.savefig(os.path.join(self.save_dir, 'w_strain.png'))

        if self.show_plot:
            plt.show()

    plot_subw_strain = Button
    def _plot_subw_strain_fired(self):

        aramis_cdt = self.aramis_cdt
        plt.figure()

        plt.title('')
        plt.plot(aramis_cdt.control_strain_t,
                 aramis_cdt.crack_width_t.T)

        plt.xlabel('control strain [-]')
        plt.ylabel('subcrack width [mm]')

        if self.save_plot:
            plt.savefig(os.path.join(self.save_dir, 'subw_strain.png'))

        if self.show_plot:
            plt.show()

    plot_stress_strain_init = Button
    def _plot_stress_strain_init_fired(self):

        aramis_cdt = self.aramis_cdt
        plt.figure()

        plt.title('')
        plt.plot(aramis_cdt.control_strain_t, aramis_cdt.stress_t)
        plt.plot(aramis_cdt.control_strain_t[aramis_cdt.init_step_avg_lst],
                 aramis_cdt.stress_t[aramis_cdt.init_step_avg_lst],
                 'ro', linewidth=4)
        plt.plot(aramis_cdt.control_strain_t[aramis_cdt.init_step_lst],
                 aramis_cdt.stress_t[aramis_cdt.init_step_lst],
                 'kx', linewidth=4)

        plt.xlabel('control strain [-]')
        plt.ylabel('nominal stress [MPa]')

        if self.save_plot:
            plt.savefig(os.path.join(self.save_dir, 'stress_strain_init.png'))

        if self.show_plot:
            plt.show()

    plot_crack_init = Button
    def _plot_crack_init_fired(self):

        aramis_cdt = self.aramis_cdt

        plt.figure()
        plt.title('')
        plt.vlines(np.arange(len(aramis_cdt.init_step_avg_lst)) + 0.5, [0], aramis_cdt.init_step_avg_lst)
        plt.xlabel('crack number')
        plt.ylabel('initiation step')

        if self.save_plot:
            plt.savefig(os.path.join(self.save_dir, 'crack_init.png'))

        if self.show_plot:
            plt.show()

    plot_ad_channels = Button
    def _plot_ad_channels_fired(self):

        aramis_cdt = self.aramis_cdt
        plt.figure()

        plt.title('AD channel values')
        y = aramis_cdt.ad_channels_arr[:, :, 2] - aramis_cdt.ad_channels_arr[:, :, 1]
        plt.plot(self.aramis_info.step_idx_list, y)

        if self.show_plot:
            plt.show()

    plot_number_of_missing_facets = Button
    def _plot_number_of_missing_facets_fired(self):

        aramis_cdt = self.aramis_cdt
        plt.figure()

        y = aramis_cdt.number_of_missing_facets_t
        plt.plot(self.aramis_info.step_idx_list, y)

        plt.xlabel('step')
        plt.ylabel('number of missing facets')
        plt.title('missing facets in time')

        if self.save_plot:
            plt.savefig(os.path.join(self.save_dir, 'number_of_missing_facets.png'))

        if self.show_plot:
            plt.show()

    plot_strain_crack_avg = Button
    def _plot_strain_crack_avg_fired(self):

        aramis_cdt = self.aramis_cdt

        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        ax.set_title(aramis_cdt.aramis_info.specimen_name + ' - %d' % aramis_cdt.evaluated_step_idx)

        plot3d_var = getattr(aramis_cdt, 'd_ux_arr')

        mask = np.logical_or(np.isnan(aramis_cdt.x_arr_undeformed),
                             aramis_cdt.data_array_undeformed_mask[0, :, :])
        mask = None
#         plt.scatter(aramis_cdt.x_arr_undeformed[mask],
#                    aramis_cdt.y_arr_undeformed[mask], c=plot3d_var[mask], cmap=my_cmap_lin,
#                    marker='s')

        print plot3d_var[mask].shape
        # contour the data, plotting dots at the nonuniform data points.
#         CS = plt.contour(aramis_cdt.x_arr_undeformed[mask][0, :, :],
#                          aramis_cdt.y_arr_undeformed[mask][0, :, :],
#                          plot3d_var[mask][0, :, :], 25, linewidths=.5, colors='k')
        # plotting filled contour
        CS = ax.contourf(aramis_cdt.x_arr_undeformed,
                         aramis_cdt.y_arr_undeformed,
                         plot3d_var, 256, cmap=plt.get_cmap('jet'))

        ax.vlines(aramis_cdt.x_arr_undeformed[0, :][aramis_cdt.crack_filter_avg],
                   [0], np.nanmax(aramis_cdt.y_arr_undeformed),
                   color='white', zorder=10, linewidth=2)

        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')

        fig.colorbar(CS, orientation='horizontal')

        if self.save_plot:
            fig.savefig(os.path.join(self.save_dir, 'strain_crack_avg.png'))

        if self.show_plot:
            fig.show()

    plot_crack_filter_crack_avg = Button
    def _plot_crack_filter_crack_avg_fired(self):
        aramis_cdt = self.aramis_cdt

        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        ax.set_title(aramis_cdt.aramis_info.specimen_name + ' - %d' % aramis_cdt.evaluated_step_idx)

        plot3d_var = getattr(aramis_cdt, 'd_ux_arr')

        mask = np.logical_or(np.isnan(aramis_cdt.x_arr_undeformed),
                             aramis_cdt.data_array_undeformed_mask[0, :, :])
        mask = None
#         plt.scatter(aramis_cdt.x_arr_undeformed[mask],
#                    aramis_cdt.y_arr_undeformed[mask], c=plot3d_var[mask], cmap=my_cmap_lin,
#                    marker='s')

        print plot3d_var[mask].shape
        # contour the gridded data, plotting dots at the nonuniform data points.
#         CS = plt.contour(aramis_cdt.x_arr_undeformed[mask][0, :, :],
#                          aramis_cdt.y_arr_undeformed[mask][0, :, :],
#                          plot3d_var[mask][0, :, :], 25, linewidths=.5, colors='k')
        # plotting filled contour
        CS = ax.contourf(aramis_cdt.x_arr_undeformed,
                         aramis_cdt.y_arr_undeformed,
                         plot3d_var, 256, cmap=plt.get_cmap('jet'))

        ax.plot(aramis_cdt.x_arr_undeformed[aramis_cdt.crack_filter],
                 aramis_cdt.y_arr_undeformed[aramis_cdt.crack_filter],
                 'k.')

        ax.vlines(aramis_cdt.x_arr_undeformed[0, :][aramis_cdt.crack_filter_avg],
                   [0], np.nanmax(aramis_cdt.y_arr_undeformed[mask]),
                   color='white', zorder=100, linewidth=2)

        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')

        # plt.colorbar()

        # plt.axis('equal')

        if self.save_plot:
            fig.savefig(os.path.join(self.save_dir, 'crack_filter_crack_avg.png'))

        if self.show_plot:
            fig.show()

    test_figure = Instance(plt.Figure)

    plot_test = Button
    def _plot_test_fired(self):
        aramis_cdt = self.aramis_cdt
        if self.test_figure == None:
            self.test_figure = plt.figure(figsize=(12, 6), dpi=100)
            self.test_figure.canvas.mpl_connect('close_event', self.handle_close)
        fig = self.test_figure
        fig.clf()

        fig.suptitle(aramis_cdt.aramis_info.specimen_name + ' - %d' % aramis_cdt.evaluated_step_idx, y=1)

        ax_diag = plt.subplot2grid((2, 3), (0, 0))
        ax_hist = plt.subplot2grid((2, 3), (1, 0))
        ax_area = plt.subplot2grid((2, 3), (0, 1), rowspan=2, colspan=2,
                                   adjustable='box', aspect='equal')

        ax_diag.plot(aramis_cdt.control_strain_t, aramis_cdt.stress_t)
        ax_diag.plot(aramis_cdt.control_strain_t[aramis_cdt.evaluated_step_idx],
                 aramis_cdt.stress_t[aramis_cdt.evaluated_step_idx], 'ro')

        ax_diag.set_xlabel('control strain [-]')
        ax_diag.set_ylabel('nominal stress [MPa]')
        ax_diag.set_xlim(0, aramis_cdt.control_strain_t.max())
        ax_diag.set_ylim(0, aramis_cdt.stress_t.max())

        if aramis_cdt.crack_arr.size != 0:
            ax_hist.hist(aramis_cdt.crack_arr, bins=40, normed=True)

        ax_hist.set_xlabel('crack_width [mm]')
        ax_hist.set_ylabel('frequency [-]')


        ax_hist_2 = ax_hist.twinx()

        if aramis_cdt.crack_arr.size != 0:
            ax_hist_2.hist(aramis_cdt.crack_arr, normed=True,
                     histtype='step', color='black',
                     cumulative=True, bins=40, linewidth=2)
        ax_hist_2.set_ylabel('probability [-]')
        ax_hist_2.set_ylim(0, 1)
        ax_hist.set_xlim(0, 0.14)
        ax_hist.set_ylim(0, 100)

        plot3d_var = getattr(aramis_cdt, 'd_ux_arr')

        CS = ax_area.contourf(aramis_cdt.x_arr_undeformed,
                         aramis_cdt.y_arr_undeformed,
                         plot3d_var, 2, cmap=plt.get_cmap('binary'))
        ax_area.plot(aramis_cdt.x_arr_undeformed, aramis_cdt.y_arr_undeformed, 'ko')

        ax_area.plot(aramis_cdt.x_arr_undeformed[aramis_cdt.crack_filter],
                 aramis_cdt.y_arr_undeformed[aramis_cdt.crack_filter], linestyle='None',
                 marker='.', color='white')

#         ax_area.vlines(aramis_cdt.x_arr_undeformed[10, :-1][aramis_cdt.crack_filter_avg],
#                    [0], np.nanmax(aramis_cdt.y_arr_undeformed),
#                    color='red', zorder=100, linewidth=1)

        ax_area.set_xlabel('x [mm]')
        ax_area.set_ylabel('y [mm]')

        # ax_area.set_colorbar(CS, orientation='horizontal')
        fig.tight_layout()

#         cbar_ax = fig.add_axes([0.45, 0.15, 0.5, 0.03])
#         fig.colorbar(CS, cax=cbar_ax, orientation='horizontal')

        fig.canvas.draw()

        if self.save_plot:
            fig.savefig(os.path.join(self.save_dir, '%s%04d.png'
                                     % (aramis_cdt.aramis_info.specimen_name, aramis_cdt.evaluated_step_idx)))

        if self.show_plot:
            fig.show()

    def handle_close(self, evt):
        print 'Closed Figure!'
        self.test_figure = None

    create_animation = Button
    def _create_animation_fired(self):
        aramis_cdt = self.aramis_cdt
        save_plot, show_plot = self.save_plot, self.show_plot
        self.save_plot = False
        self.show_plot = False
        start_step_idx = self.aramis_cdt.evaluated_step_idx
        fname_pattern = '%s%04d'
        for step_idx in self.aramis_info.step_idx_list:
            self.aramis_cdt.evaluated_step_idx = step_idx
            self.plot_test = True
            self.test_figure.savefig(os.path.join(self.temp_dir, fname_pattern
                                     % (aramis_cdt.aramis_info.specimen_name, aramis_cdt.evaluated_step_idx)))
        try:
            os.system('ffmpeg -framerate 3 -i %s.png -vcodec ffv1 -sameq %s.avi' %
                      (os.path.join(self.temp_dir, fname_pattern.replace('%s', aramis_cdt.aramis_info.specimen_name)),
                       os.path.join(self.save_dir, aramis_cdt.aramis_info.specimen_name)))
            os.system('convert -verbose -delay 25 %s* %s.gif' %
                      (os.path.join(self.temp_dir, aramis_cdt.aramis_info.specimen_name),
                       os.path.join(self.save_dir, aramis_cdt.aramis_info.specimen_name)))
        except:
            print 'It\'s not possible to create animation'

        self.save_plot = save_plot
        self.show_plot = show_plot
        self.aramis_cdt.evaluated_step_idx = start_step_idx

    view = View(
                Item('object.aramis_cdt.evaluated_step_idx',
                     editor=RangeEditor(low=0, high_name='object.aramis_cdt.step_idx_max',
                                        auto_set=False, enter_set=True, mode='slider'),
                                        springy=True),
                UItem('plot2d'),
                UItem('plot_crack_hist'),
                UItem('plot_number_of_cracks_t'),
                UItem('plot_ad_channels'),
                UItem('plot_number_of_missing_facets'),
                UItem('plot_stress_strain_init'),
                UItem('plot_stress_strain'),
                UItem('plot_w_strain'),
                UItem('plot_subw_strain'),
                UItem('plot_force_step'),
                UItem('plot_strain_crack_avg'),
                UItem('plot_crack_filter_crack_avg'),
                UItem('plot_crack_init'),
                UItem('plot_test'),
                UItem('create_animation')
                )
