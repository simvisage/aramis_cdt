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
    HasTraits, Float, Bool, Trait, Instance, Button, WeakRef

from etsproxy.traits.ui.api import  UItem, View, Item

import numpy as np
# import etsproxy.mayavi.mlab as m

import platform
import time
if platform.system() == 'Linux':
    sysclock = time.time
elif platform.system() == 'Windows':
    sysclock = time.clock

import os
from mayavi.tools.mlab_scene_model import MlabSceneModel

from aramis_info import AramisInfo
from aramis_data import AramisData
from aramis_cdt import AramisCDT

class AramisView3D(HasTraits):
    '''This class manages 3D views for AramisCDT variables
    '''

    aramis_data = Instance(AramisData)

    aramis_cdt = Instance(AramisCDT)

    scene = Instance(MlabSceneModel)

    plot_title = Bool(True)

    plot3d_var = Trait('07 d_ux_arr [mm]', {'01 x_arr [mm]':['aramis_data', 'x_arr_undeformed'],
                                            '02 y_arr [mm]':['aramis_data', 'y_arr_undeformed'],
                                            '03 z_arr [mm]':['aramis_data', 'z_arr_undeformed'],
                                            '04 ux_arr [mm]':['aramis_data', 'ux_arr'],
                                            '05 uy_arr [mm]':['aramis_data', 'uy_arr'],
                                            '06 uz_arr [mm]':['aramis_data', 'uz_arr'],
                                            '07 d_ux_arr [mm]':['aramis_cdt', 'd_ux_arr'],
                                            '08 crack_filed_arr [mm]': ['aramis_cdt', 'crack_field_arr']})

    plot3d_points_flat = Button
    def _plot3d_points_flat_fired(self):
        '''Plot array of variable using colormap
        '''
        # m.figure(fgcolor=(0, 0, 0), bgcolor=(1, 1, 1) , size=(900, 600))
        # engine = m.get_engine()
        # scene = engine.scenes[0]
        self.scene.mlab.clf()
        m = self.scene.mlab
        m.fgcolor = (0, 0, 0)
        m.bgcolor = (1, 1, 1)
        self.scene.scene.disable_render = True

        plot3d_var = getattr(getattr(self, self.plot3d_var_[0]), self.plot3d_var_[1])

        mask = np.logical_or(np.isnan(self.aramis_data.x_arr_undeformed),
                             self.aramis_data.data_array_undeformed_mask[0, :, :])
        mask = None
        m.points3d(self.aramis_data.x_arr_undeformed[mask],
                   self.aramis_data.y_arr_undeformed[mask],
                   self.aramis_data.z_arr_undeformed[mask],
                   plot3d_var[mask],
                   mode='cube',
                   scale_mode='none', scale_factor=1)
        m.view(0, 0)
        self.scene.scene.parallel_projection = True
        self.scene.scene.disable_render = False

        if self.plot_title:
            m.title('step no. %d' % self.aramis_data.evaluated_step_idx, size=0.3)

        m.scalarbar(orientation='horizontal', title=self.plot3d_var_[1])

        # plot axes
        m.axes()

    glyph_x_length = Float(0.200)
    glyph_y_length = Float(0.200)
    glyph_z_length = Float(0.000)

    glyph_x_length_cr = Float(3.000)
    glyph_y_length_cr = Float(0.120)
    glyph_z_length_cr = Float(0.120)

    plot3d_points = Button
    def _plot3d_points_fired(self):
        '''Plot arrays of variables in 3d relief
        '''
        aramis_cdt = self.aramis_cdt
        # m.figure(fgcolor=(0, 0, 0), bgcolor=(1, 1, 1), size=(900, 600))

        #
        # scene = engine.scenes[0]
        self.scene.mlab.clf()
        m = self.scene.mlab
        m.fgcolor = (0, 0, 0)
        m.bgcolor = (1, 1, 1)
        self.scene.scene.disable_render = True

        #-----------------------------------
        # plot crack width ('crack_field_w')
        #-----------------------------------

        z_arr = np.zeros_like(self.aramis_data.z_arr_undeformed)

        plot3d_var = getattr(getattr(self, self.plot3d_var_[0]), self.plot3d_var_[1])
        m.points3d(z_arr, self.aramis_data.x_arr_undeformed, self.aramis_data.y_arr_undeformed, plot3d_var,
                   mode='cube', colormap="blue-red", scale_mode='scalar')

        # scale glyphs
        #
        glyph = self.scene.engine.scenes[0].children[0].children[0].children[0]
        glyph.glyph.glyph_source.glyph_position = 'tail'
        glyph.glyph.glyph_source.glyph_source.x_length = self.glyph_x_length_cr
        glyph.glyph.glyph_source.glyph_source.y_length = self.glyph_y_length_cr
        glyph.glyph.glyph_source.glyph_source.z_length = self.glyph_z_length_cr

        #-----------------------------------
        # plot displacement jumps ('d_ux_w')
        #-----------------------------------

        plot3d_var = getattr(getattr(self, self.plot3d_var_[0]), self.plot3d_var_[1])
        m.points3d(z_arr, self.aramis_data.x_arr_undeformed, self.aramis_data.y_arr_undeformed, plot3d_var, mode='cube',
                   colormap="blue-red", scale_mode='none')

        glyph1 = self.scene.engine.scenes[0].children[1].children[0].children[0]
#       # switch order of the scale_factor corresponding to the order of the
        glyph1.glyph.glyph_source.glyph_source.x_length = self.glyph_z_length
        glyph1.glyph.glyph_source.glyph_source.y_length = self.glyph_x_length
        glyph1.glyph.glyph_source.glyph_source.z_length = self.glyph_y_length

        # rotate scene
        #
        # scene = engine.scenes[0]
        self.scene.scene.parallel_projection = True
        m.view(0, 90)

        glyph.glyph.glyph_source.glyph_position = 'head'
        glyph.glyph.glyph_source.glyph_position = 'tail'

        module_manager = self.scene.engine.scenes[0].children[1].children[0]
        module_manager.scalar_lut_manager.show_scalar_bar = True
        module_manager.scalar_lut_manager.show_legend = True
        module_manager.scalar_lut_manager.scalar_bar.orientation = 'horizontal'
        module_manager.scalar_lut_manager.scalar_bar.title = self.plot3d_var_[1]
        module_manager.scalar_lut_manager.scalar_bar_representation.position = (0.10, 0.05)
        module_manager.scalar_lut_manager.scalar_bar_representation.position2 = (0.8, 0.15)
        self.scene.scene.disable_render = False

        if self.plot_title:
            m.title('step no. %d' % self.aramis_data.evaluated_step_idx, size=0.3)

        # m.scalarbar(orientation='horizontal', title=self.plot3d_var_[1])

        # plot axes
        #
        m.axes()

    plot3d_cracks = Button
    def _plot3d_cracks_fired(self):
        '''Plot cracks in 3D
        '''
        aramis_cdt = self.aramis_cdt
        # m.figure(fgcolor=(0, 0, 0), bgcolor=(1, 1, 1), size=(900, 600))

        # engine = m.get_engine()
        # scene = engine.scenes[0]
        self.scene.mlab.clf()
        m = self.scene.mlab
        m.fgcolor = (0, 0, 0)
        m.bgcolor = (1, 1, 1)
        self.scene.scene.disable_render = True

        #-----------------------------------
        # plot crack width ('crack_field_w')
        #-----------------------------------

        z_arr = np.zeros_like(self.aramis_data.z_arr_undeformed)

        plot3d_var = aramis_cdt.crack_field_arr
#        plot3d_var = getattr(self, 'crack_field_w')
        m.points3d(z_arr, self.aramis_data.x_arr_undeformed, self.aramis_data.y_arr_undeformed, plot3d_var,
                   mode='cube', colormap="blue-red", scale_mode='scalar')

        # scale glyphs
        #
        glyph = self.scene.engine.scenes[0].children[0].children[0].children[0]
        glyph.glyph.glyph_source.glyph_position = 'tail'
        glyph.glyph.glyph_source.glyph_source.x_length = self.glyph_x_length_cr
        glyph.glyph.glyph_source.glyph_source.y_length = self.glyph_y_length_cr
        glyph.glyph.glyph_source.glyph_source.z_length = self.glyph_z_length_cr

        #-----------------------------------
        # plot displacement jumps ('d_ux_w')
        #-----------------------------------

#        plot3d_var = getattr(self, 'd_ux_w')
        plot3d_var = getattr(aramis_cdt, 'crack_field_arr')
        m.points3d(z_arr, self.aramis_data.x_arr_undeformed, self.aramis_data.y_arr_undeformed, plot3d_var, mode='cube',
                   colormap="blue-red", scale_mode='none')

        glyph1 = self.scene.engine.scenes[0].children[1].children[0].children[0]
#       # switch order of the scale_factor corresponding to the order of the
        glyph1.glyph.glyph_source.glyph_source.x_length = self.glyph_z_length
        glyph1.glyph.glyph_source.glyph_source.y_length = self.glyph_x_length
        glyph1.glyph.glyph_source.glyph_source.z_length = self.glyph_y_length

        # rotate scene
        #
        scene = self.scene.engine.scenes[0]
        scene.scene.parallel_projection = True
        m.view(0, 90)

        glyph.glyph.glyph_source.glyph_position = 'head'
        glyph.glyph.glyph_source.glyph_position = 'tail'

        module_manager = self.scene.engine.scenes[0].children[1].children[0]
        module_manager.scalar_lut_manager.show_scalar_bar = True
        module_manager.scalar_lut_manager.show_legend = True
        module_manager.scalar_lut_manager.scalar_bar.orientation = 'horizontal'
        module_manager.scalar_lut_manager.scalar_bar.title = 'crack_field'
        module_manager.scalar_lut_manager.scalar_bar_representation.position = (0.10, 0.05)
        module_manager.scalar_lut_manager.scalar_bar_representation.position2 = (0.8, 0.15)
        scene.scene.disable_render = False

        if self.plot_title:
            m.title('step no. %d' % self.aramis_data.evaluated_step_idx, size=0.3)

        # m.scalarbar(orientation='horizontal', title='crack_field')

        # plot axes
        #
        # m.axes()

    clean_scene = Button
    def _clean_scene_fired(self):
        self.scene.mlab.clf()


    view = View(
                Item('plot3d_var'),
                UItem('plot3d_points_flat'),
                UItem('plot3d_points'),
                UItem('plot3d_cracks'),
                Item('_'),
                UItem('clean_scene'),
                id='aramisCDT.view3d',
               )


if __name__ == '__main__':
    from os.path import expanduser
    home = expanduser("~")

    data_dir = os.path.join(home, '.simdb_cache', 'aramis', 'TTb-4c-2cm-0-TU-V1_bs4-Xf19s15-Yf19s15')

    AI = AramisInfo(data_dir=data_dir)
    AD = AramisData(aramis_info=AI)
    AC = AramisCDT(aramis_info=AI,
                   aramis_data=AD,
                integ_radius=1
                  )
    AramisView3D(
                 aramis_info=AI,
                 aramis_data=AD,
                 aramis_cdt=AC
                 ).configure_traits()
