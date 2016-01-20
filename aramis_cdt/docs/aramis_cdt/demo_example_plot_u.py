#-------------------------------------------------------------------------
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
#-------------------------------------------------------------------------

from demo_example import x, dic_ui

# begin_plot
ux = dic_ui.aramis_data.ux_arr[10, :]
plt.plot(x, ux)
# end_plot
plt.show()