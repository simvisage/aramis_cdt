===============
Crack detection
=============== 
  
Data representation
-------------------

.. image:: data_structure.png
	:width: 400px
	
The setup of the DIC processor is done using the script:

.. include:: demo_example.py
   :literal:
   :start-after: # begin_read
   :end-before: # end_read

.. todo::
	The api should be simplified. In order to process data, the view class should not be required.
	Ideally, a simple class holding the data and providing property attributes to get the fields
	and data on crack detection should be provided. Independently on that, a viiew (ui) class should
	put it into the interface. Similar concepts are followed in oricreate. 
	
.. todo::
	In the script above I renamed the variables AI, AC etc to lowercase. 
	This is more standard. .
	
.. todo::
	Vaclav, can we rename aramis to dic - (digital image correlation? The processing is not necessarily
	linked with aramis. We are intending to use open source pattern matching software tha can work
	with a standard digital camera. (priority - low)

Available data fields
---------------------	  

The ``dic`` object serve for the access of the cell coordinates by spatial indices :math:`(i,j)`
during the loading history given by the stage index  :math:`s`.

.. math::
	\bm{x}^s_{i,j} \;\; \mathrm{with} \; s = 0 \ldots n_s, \; i = 0 \ldots n_i,  \; j = 0 \ldots n_j

Facet distance for the given integration radius :math:`r` is usually constant:

.. math::
	\Delta \bm{x}^{(2r)}_{i,j} = \bm{x}^0_{i+r,j} - \bm{x}^0_{i-r,j}

Derived fields are provided in form of property traits of the ``dic.aramis_data`` object.
In particular, cell displacements at the stage :math:`s` defined as

.. math::
	\bm{u}^s_{i,j} = \bm{x}^s_{i,j} - \bm{x}^0_{i,j}

can be accessed and plotted using the code

.. include:: demo_example_plot_u.py
   :literal:
   :start-after: # begin_plot
   :end-before: # end_plot

.. plot:: aramis_cdt/demo_example_plot_u.py

Nonlocal strain calculated using the predefined integration radius :math:`r` (see the 
parameter ``integ_radius`` of the ``AramisCDT`` class above) is calculated as

.. math::
	\mathrm{d}_x \bm{u}^s_{i,j} = \frac{\bm{u}^s_{i+r,j} - \bm{u}^s_{i-r,j}}{\Delta \bm{x}^{(2r)}_{i,j} }

In the running example, the evaluation and plotting goes like this:

.. include:: demo_example_plot_du_x.py
   :literal:
   :start-after: # begin_plot
   :end-before: # end_plot

.. plot:: aramis_cdt/demo_example_plot_du_x.py

First derivative of nonlocal strain defined as

.. math::
	\mathrm{d}^2_x \bm{u}^s_{i,j}= \frac{\mathrm{d}_x \bm{u}^s_{i+r,j} - \mathrm{d}_x \bm{u}^s_{i-r,j}}{\Delta \bm{x}^{(2r)}_{i,j} }

can be accessed as

.. include:: demo_example_plot_ddu_x.py
   :literal:
   :start-after: # begin_plot
   :end-before: # end_plot

.. plot:: aramis_cdt/demo_example_plot_ddu_x.py

.. todo:: 
	plot the threshold line for ``delta_2``

Finally, the second derivative of nonlocal strain

.. math::
	\mathrm{d}^3_x \bm{u}^s_{i,j}= \frac{\mathrm{d}^2_x \bm{u}^s_{i+r,j} - \mathrm{d}^2_x \bm{u}^s_{i-r,j}}{\Delta \bm{x}^{(2r)}_{i,j} }

.. include:: demo_example_plot_dddu_x.py
   :literal:
   :start-after: # begin_plot
   :end-before: # end_plot

.. plot:: aramis_cdt/demo_example_plot_dddu_x.py

.. todo::
	Plot the value of used threshold cutting away the noise - ``delta 3``.
	
.. todo::
	include the description of axes including units
	
.. todo:::
	change the size of the figures.

Criteria for crack detection
-----------------------------

The first condition says that the non-local strain, must have a peak of strain at the crack position.. 
This means, that the second displacement derivative must change its sign at crack: 

.. math::
	\mathrm{d}^2_x\bm{u}^s_{i,j} \cdot \mathrm{d}^2_x\bm{u}^s_{i+1,j} \leq \delta_2

However, this condition is not sufficient, since it detects both maximums and minimums of the strain field
:math:`\mathrm{d}_x\bm{u}^s_{i,j}`. 
In order to identify the maximum values, we require the slope of the third derivative 
:math:`\mathrm{d}^2_x\bm{u}^s_{i,j}` to 
be negative, i.e.
	
.. math::
	\mathrm{d}^3_x \bm{u}^s_{i,j} \leq 0

Still, this criterion interprets shallow oscillation of strains around zero :math:`\mathrm{d}_x\bm{u}^s_{i,j}`
due to noise in measurements as crack. Therefore, we introduce a threshold on the slope of the third derivative
:math:`\delta^3\; \mathrm{[-/mm^2]}`. This can be interpreted as minimum required curvature of a strain peak 
to be interpreted as a crack. This curvature is evaluated as for two cells adjacent to the candidate crack position. 

.. math::
	(\mathrm{d}^3_x \bm{u}^s_{i,j} + \mathrm{d}^3_x\bm{u}^s_{i+1,j}) / 2 \leq \delta_3


.. todo::
	Finish the description of the crack detection criteria and check it with the code. Provide the references to
	the places in the code.


Evaluation of crack initiation stage
------------------------------------

Evaluation of the crack width
-----------------------------
