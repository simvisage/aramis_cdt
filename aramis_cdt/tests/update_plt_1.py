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
import numpy as np
import matplotlib
matplotlib.use('WXAgg')
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

class DummyPlot(object):
    def __init__(self):
        self.imsize = (10, 10)
        self.data = np.random.random(self.imsize)

        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(self.data)

        buttonax = self.fig.add_axes([0.45, 0.9, 0.1, 0.075])
        self.button = Button(buttonax, 'Update')
        self.button.on_clicked(self.update)

    def update(self, event):
        self.data += np.random.random(self.imsize) - 0.5
        self.im.set_data(self.data)
        self.im.set_clim([self.data.min(), self.data.max()])
        self.fig.canvas.draw()

    def show(self):
        plt.show()

p = DummyPlot()
p.show()

from traits.api import HasTraits, Button
from traitsui.api import View


class Dplot(HasTraits):

    init_plot = Button
    def _init_plot_fired(self):
        self.imsize = (10, 10)
        self.data = np.random.random(self.imsize)

        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(self.data)

    update_plot = Button
    def _update_plot_fired(self):
        self.data += np.random.random(self.imsize) - 0.5
        self.im.set_data(self.data)
        self.im.set_clim([self.data.min(), self.data.max()])
        self.fig.canvas.draw()

    show_plot = Button
    def _show_plot_fired(self):
        plt.show()

    view = View(
                'init_plot',
                'update_plot',
                'show_plot'
                )

p = Dplot()
p.configure_traits()
