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
import os

def report_gen(AUI):
    AUI.aramis_view2d.save_plot = True
    AUI.aramis_view2d.show_plot = False

    number_of_cracks_avg = AUI.aramis_cdt.number_of_cracks_avg
    crack_spacing_avg = AUI.aramis_cdt.crack_spacing_avg

    AUI.aramis_view2d.plot_strain_crack_avg = True
    AUI.aramis_view2d.plot_crack_filter_crack_avg = True
    AUI.aramis_view2d.plot_crack_hist = True

    AUI.aramis_cdt.run_t = True
    AUI.aramis_view2d.plot_force_t_step = True
    AUI.aramis_view2d.plot_number_of_missing_facets = True
    AUI.aramis_view2d.plot_stress_strain = True

    AUI.aramis_cdt.run_back = True
    AUI.aramis_view2d.plot_crack_init = True
    AUI.aramis_view2d.plot_stress_strain_init = True

    report_source = '''
======================================
Specimen %s
======================================

    ''' % os.path.split(AUI.aramis_info.data_dir)[-1]

    report_source += '''
AramisCDT settings
^^^^^^^^^^^^^^^^^^  

====================================  ==============
 parameter                             value
====================================  ==============
 number of facets in x                 %d
 number of facets in y                 %d
 facet distance in x                   %d
 facet distance in y                   %d
 integration radius                    %d
 crack detect step                     %d
 transform data                        %s
 d_ux_threshold                        %.6f
 dd_ux_threshold                       %.6f
 dd_ux_avg_threshold                   %.6f
 ddd_ux_threshold                      %.6f
 ddd_ux_avg_threshold                  %.6f
====================================  ==============

Numerical results
^^^^^^^^^^^^^^^^^

====================================  ==============
 max force_t [kN]                      %.3f
 max stress [MPa]                      %.3f
 number of cracks [-]                  %d
 average crack spacing [mm]            %.3f
 first avg crack in step [-]           %d
====================================  ==============

    ''' % (AUI.aramis_cdt.n_px_facet_size_x,
           AUI.aramis_cdt.n_px_facet_size_y,
           AUI.aramis_cdt.n_px_facet_distance_x,
           AUI.aramis_cdt.n_px_facet_distance_y,
           AUI.aramis_cdt.integ_radius,
           AUI.aramis_cdt.w_detect_step,
           AUI.aramis_cdt.transform_data,
           AUI.aramis_cdt.d_ux_threshold,
           AUI.aramis_cdt.dd_ux_threshold,
           AUI.aramis_cdt.dd_ux_avg_threshold,
           AUI.aramis_cdt.ddd_ux_threshold,
           AUI.aramis_cdt.ddd_ux_avg_threshold,
           np.nanmax(AUI.aramis_cdt.force_t),
           np.nanmax(AUI.aramis_cdt.stress_t),
           number_of_cracks_avg,
           crack_spacing_avg,
           np.min(AUI.aramis_cdt.init_step_avg_lst)
           )

    report_source += '''
Check facet field
^^^^^^^^^^^^^^^^^

.. image:: number_of_missing_facets.png
    :width: 500px
    
Crack field at the crack detect step
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+----------------------------------------+----------------------------------------+
| Contour plot of ``d_ux`` overlaid by   | Cracks (black dots),                   |
| average cracks (white lines)           | average cracks (white lines)           |
+----------------------------------------+----------------------------------------+
| .. image:: strain_crack_avg.png        | .. image:: crack_filter_crack_avg.png  |
|     :width: 500px                      |     :width: 500px                      |
+----------------------------------------+----------------------------------------+

.. image:: crack_init.png
    :width: 500px

Histogram of sub-cracks width at the crack detect step
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: crack_hist.png
    :width: 500px

Plots
^^^^^^^^

+--------------------------------+--------------------------------+
| .. image:: force_time.png      | .. image:: stress_strain.png   |
|     :width: 500px              |     :width: 500px              |
+--------------------------------+--------------------------------+

.. image:: stress_strain_init.png
    :width: 500px

    '''

    with open('report.rst', 'w') as f:
        f.write(report_source)

