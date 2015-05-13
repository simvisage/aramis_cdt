"""
Thanks to Tony Yu <tsyu80@gmail.com> for the logo design
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['text.usetex'] = True
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['axes.edgecolor'] = 'gray'


axalpha = 0.05
# figcolor = '#EFEFEF'
figcolor = 'white'
dpi = 80
fig = plt.figure(figsize=(6, 1.1), dpi=dpi)
fig.figurePatch.set_edgecolor(figcolor)
fig.figurePatch.set_facecolor(figcolor)


def add_math_background():
    ax = fig.add_axes([0., 0., 1., 1.])

#     text = []
#     text.append((r"$p\left( t \right) = a_{0} + \sum\limits_{n=1}^{\infty} \left[ a_{n} \cos\left( n \Omega_1 t \right) + b_{n} \sin\left( n \Omega_1 t \right) \right]$",
#                  (0.02, 0.74), 15))
#     text.append((r"$\Omega_1 = \frac{2 \pi}{T_1}$",
#                 (0.8, 0.8), 15))
#     text.append((r"$a_0 = \frac{1}{T_1}\int\limits_{\tau}^{\tau+T_1}p\left( t \right) \mathrm{d}t$",
#                 (0.02, 0.2), 15))
#     text.append((r"$a_n = \frac{2}{T_1}\int\limits_{\tau}^{\tau+T_1}p\left( t \right) \cos\left( n \Omega_1 t \right) \mathrm{d}t$",
#                 (0.25, 0.42), 15))
#     text.append((r"$b_n = \frac{2}{T_1}\int\limits_{\tau}^{\tau+T_1}p\left( t \right) \sin\left( n \Omega_1 t \right) \mathrm{d}t$",
#                 (0.55, 0.2), 15))
#     for eq, (x, y), size in text:
#         ax.text(x, y, eq, ha='left', va='center', color="#11557c", alpha=0.25,
#                 transform=ax.transAxes, fontsize=size)
    ax.set_axis_off()
    return ax

def add_matplotlib_text(ax):
    ax.text(0.01, 0.5, 'AramisCDT', color='#11557c', fontsize=65,
               ha='left', va='center', alpha=1.0, transform=ax.transAxes)

def add_square_wave():
    ax = fig.add_axes([0.04, 0.075, 0.4, 0.85])

    par = [1.27323534e+00, 4.24400565e-01, 2.54626881e-01, 1.81861923e-01, 1.41433208e-01]

    ax.axesPatch.set_alpha(axalpha)
    # ax.set_axisbelow(True)

    x = np.linspace(0, 4 * np.pi, 1000)
    y = 0
    for i, p in zip(range(1, 10, 2), par):
        y += p * np.sin(i * x)
    ax.plot([0, 4 * np.pi], [0, 0], color='grey')
    ax.plot([0, 0, np.pi, np.pi, 2 * np.pi, 2 * np.pi, 3 * np.pi, 3 * np.pi, 4 * np.pi, 4 * np.pi],
            [0, 1, 1, -1, -1, 1, 1, -1, -1, 0], 'k-', linewidth=1.5)
    ax.plot(x, y, 'b-', linewidth=1.5)
    ax.set_axis_off()

if __name__ == '__main__':
    main_axes = add_math_background()
    # add_square_wave()
    add_matplotlib_text(main_axes)
    plt.show()
