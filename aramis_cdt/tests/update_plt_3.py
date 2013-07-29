import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

class DummyPlot(object):
    def __init__(self):
        self.imsize = (10, 10)
        self.fig, self.ax = plt.subplots()

        self.ax.axis([-0.5, self.imsize[1] - 0.5,
                      self.imsize[0] - 0.5, -0.5])
        self.ax.set_aspect(1.0)
        self.ax.autoscale(False)

        buttonax = self.fig.add_axes([0.45, 0.9, 0.1, 0.075])
        self.button = Button(buttonax, 'Update')
        self.button.on_clicked(self.update)

    def update(self, event):
        self.ax.imshow(np.random.random(self.imsize))
        self.fig.canvas.draw()

    def show(self):
        plt.show()

p = DummyPlot()
p.show()
