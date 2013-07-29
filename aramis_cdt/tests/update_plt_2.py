import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

class DummyPlot(object):
    def __init__(self):
        self.imsize = (10, 10)
        self.data = np.random.random(self.imsize)
        self.fig, self.ax = plt.subplots()

        dummy_data = np.zeros(self.imsize)
        self.im = self.ax.imshow(dummy_data)
        self.im.set_visible(False)

        buttonax = self.fig.add_axes([0.45, 0.9, 0.1, 0.075])
        self.button = Button(buttonax, 'Update')
        self.button.on_clicked(self.update)

    def update(self, event):
        self.im.set_visible(True)
        self.data += np.random.random(self.imsize) - 0.5
        self.im.set_data(self.data)
        self.im.set_clim([self.data.min(), self.data.max()])
        self.fig.canvas.draw()

    def show(self):
        plt.show()

p = DummyPlot()
p.show()
