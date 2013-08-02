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
    HasTraits, Float, Bool, Trait, Instance, Button

from etsproxy.traits.ui.api import  UItem, View, Item

import numpy as np
import etsproxy.mayavi.mlab as m

import platform
import time
if platform.system() == 'Linux':
    sysclock = time.time
elif platform.system() == 'Windows':
    sysclock = time.clock

from aramis_cdt import AramisCDT

class AramisView3D(HasTraits):
    '''This class manages 3D views for AramisCDT variables
    '''

    aramis_cdt = Instance(AramisCDT)

    plot_title = Bool(True)

    plot3d_var = Trait('07 d_ux_arr [mm]', {'01 x_arr [mm]':'x_arr',
                                            '02 y_arr [mm]':'y_arr',
                                            '03 z_arr [mm]':'z_arr',
                                            '04 ux_arr [mm]':'ux_arr',
                                            '05 uy_arr [mm]':'uy_arr',
                                            '06 uz_arr [mm]':'uz_arr',
                                            '07 d_ux_arr [mm]':'d_ux_arr',
                                            '08 crack_filed_arr [mm]': 'crack_field_arr'})

    plot3d_points_flat = Button
    def _plot3d_points_flat_fired(self):
        '''Plot array of variable using colormap
        '''
        aramis_cdt = self.aramis_cdt

        m.figure(fgcolor=(0, 0, 0), bgcolor=(1, 1, 1) , size=(1600, 900))
        engine = m.get_engine()
        scene = engine.scenes[0]
        scene.scene.disable_render = True

        plot3d_var = getattr(aramis_cdt, self.plot3d_var_)

        mask = np.logical_or(np.isnan(aramis_cdt.x_arr),
                             aramis_cdt.data_array_undeformed_mask[0, :, :])
        mask = None
        m.points3d(aramis_cdt.x_arr[mask],
                   aramis_cdt.y_arr[mask],
                   aramis_cdt.z_arr[mask],
                   plot3d_var[mask],
                   mode='cube',
                   scale_mode='none', scale_factor=1)
        m.view(0, 0)
        scene.scene.parallel_projection = True
        scene.scene.disable_render = False

        if self.plot_title:
            m.title('step no. %d' % aramis_cdt.evaluated_step_idx, size=0.3)

        m.scalarbar(orientation='horizontal', title=self.plot3d_var_)

        # plot axes
        m.axes()

        m.show()

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
        m.figure(fgcolor=(0, 0, 0), bgcolor=(1, 1, 1), size=(900, 600))

        engine = m.get_engine()
        scene = engine.scenes[0]
        scene.scene.disable_render = True

        #-----------------------------------
        # plot crack width ('crack_field_w')
        #-----------------------------------

        z_arr = np.zeros_like(aramis_cdt.z_arr)

        plot3d_var = getattr(aramis_cdt, self.plot3d_var_)
        m.points3d(z_arr, aramis_cdt.x_arr, aramis_cdt.y_arr, plot3d_var,
                   mode='cube', colormap="blue-red", scale_mode='scalar', vmax=.2)

        # scale glyphs
        #
        glyph = engine.scenes[0].children[0].children[0].children[0]
        glyph.glyph.glyph_source.glyph_position = 'tail'
        glyph.glyph.glyph_source.glyph_source.x_length = self.glyph_x_length_cr
        glyph.glyph.glyph_source.glyph_source.y_length = self.glyph_y_length_cr
        glyph.glyph.glyph_source.glyph_source.z_length = self.glyph_z_length_cr

        #-----------------------------------
        # plot displacement jumps ('d_ux_w')
        #-----------------------------------

        plot3d_var = getattr(aramis_cdt, self.plot3d_var_)
        m.points3d(z_arr, aramis_cdt.x_arr, aramis_cdt.y_arr, plot3d_var, mode='cube',
                   colormap="blue-red", scale_mode='none')

        glyph1 = engine.scenes[0].children[1].children[0].children[0]
#       # switch order of the scale_factor corresponding to the order of the
        glyph1.glyph.glyph_source.glyph_source.x_length = self.glyph_z_length
        glyph1.glyph.glyph_source.glyph_source.y_length = self.glyph_x_length
        glyph1.glyph.glyph_source.glyph_source.z_length = self.glyph_y_length

        # rotate scene
        #
        scene = engine.scenes[0]
        scene.scene.parallel_projection = True
        m.view(0, 90)

        glyph.glyph.glyph_source.glyph_position = 'head'
        glyph.glyph.glyph_source.glyph_position = 'tail'

        module_manager = engine.scenes[0].children[0].children[0]
        module_manager.scalar_lut_manager.scalar_bar_representation.minimum_size = np.array([1, 1])
        module_manager.scalar_lut_manager.scalar_bar_representation.position2 = np.array([ 0.20603604, 0.16827586])
        module_manager.scalar_lut_manager.scalar_bar_representation.moving = 0
        module_manager.scalar_lut_manager.scalar_bar_representation.position = np.array([ 0.53971972, 0.19931035])
        module_manager.scalar_lut_manager.scalar_bar_representation.maximum_size = np.array([100000, 100000])
        scene.scene.disable_render = False

        if self.plot_title:
            m.title('step no. %d' % aramis_cdt.evaluated_step_idx, size=0.3)

        m.scalarbar(orientation='horizontal', title=self.plot3d_var_)

        # plot axes
        #
        m.axes()

        m.show()

    plot3d_cracks = Button
    def _plot3d_cracks_fired(self):
        '''Plot cracks in 3D
        '''
        aramis_cdt = self.aramis_cdt
        m.figure(fgcolor=(0, 0, 0), bgcolor=(1, 1, 1), size=(900, 600))

        engine = m.get_engine()
        scene = engine.scenes[0]
        scene.scene.disable_render = True

        #-----------------------------------
        # plot crack width ('crack_field_w')
        #-----------------------------------

        z_arr = np.zeros_like(aramis_cdt.z_arr)

        plot3d_var = aramis_cdt.crack_field_arr
#        plot3d_var = getattr(self, 'crack_field_w')
        m.points3d(z_arr, aramis_cdt.x_arr, aramis_cdt.y_arr, plot3d_var,
                   mode='cube', colormap="blue-red", scale_mode='scalar', vmax=.2)

        # scale glyphs
        #
        glyph = engine.scenes[0].children[0].children[0].children[0]
        glyph.glyph.glyph_source.glyph_position = 'tail'
        glyph.glyph.glyph_source.glyph_source.x_length = self.glyph_x_length_cr
        glyph.glyph.glyph_source.glyph_source.y_length = self.glyph_y_length_cr
        glyph.glyph.glyph_source.glyph_source.z_length = self.glyph_z_length_cr

        #-----------------------------------
        # plot displacement jumps ('d_ux_w')
        #-----------------------------------

#        plot3d_var = getattr(self, 'd_ux_w')
        plot3d_var = getattr(aramis_cdt, 'crack_field_arr')
        m.points3d(z_arr, aramis_cdt.x_arr, aramis_cdt.y_arr, plot3d_var, mode='cube',
                   colormap="blue-red", scale_mode='none')

        glyph1 = engine.scenes[0].children[1].children[0].children[0]
#       # switch order of the scale_factor corresponding to the order of the
        glyph1.glyph.glyph_source.glyph_source.x_length = self.glyph_z_length
        glyph1.glyph.glyph_source.glyph_source.y_length = self.glyph_x_length
        glyph1.glyph.glyph_source.glyph_source.z_length = self.glyph_y_length

        # rotate scene
        #
        scene = engine.scenes[0]
        scene.scene.parallel_projection = True
        m.view(0, 90)

        glyph.glyph.glyph_source.glyph_position = 'head'
        glyph.glyph.glyph_source.glyph_position = 'tail'

        module_manager = engine.scenes[0].children[0].children[0]
        module_manager.scalar_lut_manager.scalar_bar_representation.minimum_size = np.array([1, 1])
        module_manager.scalar_lut_manager.scalar_bar_representation.position2 = np.array([ 0.20603604, 0.16827586])
        module_manager.scalar_lut_manager.scalar_bar_representation.moving = 0
        module_manager.scalar_lut_manager.scalar_bar_representation.position = np.array([ 0.53971972, 0.19931035])
        module_manager.scalar_lut_manager.scalar_bar_representation.maximum_size = np.array([100000, 100000])
        scene.scene.disable_render = False

        if self.plot_title:
            m.title('step no. %d' % aramis_cdt.evaluated_step_idx, size=0.3)

        m.scalarbar(orientation='horizontal', title='crack_field')

        # plot axes
        #
        # m.axes()

        m.show()

    view = View(
                Item('plot3d_var'),
                UItem('plot3d_points_flat'),
                UItem('plot3d_points'),
                UItem('plot3d_cracks'),
               )
