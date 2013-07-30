=================
AramisCDT
================= 
    
.. image:: data_structure.png
	:width: 400px
	

Facet coordinates :math:`\bm{X}^s_i,j`	

Facet displacement

.. math::
	\Delta\bm{X}^s_{i,j} = \bm{X}^s_{i,j} - \bm{X}^0_{i,j}\qquad \mathrm{for}\, s = 0 \ldots n_s, i = 0 \ldots n_i, j = 0 \ldots n_j

strain

.. math::
	\mathrm{d}\bm{X}^s_{i,j} = \bm{X}^s_{i-2r,j} - \bm{X}^s_{i+2r,j}\qquad \mathrm{for}\, s = 0 \ldots n_s, i = 2r \ldots n_i-2r, j = 0 \ldots n_j

.. math::
	\mathrm{d_2}\bm{X}^s_{i,j} = \mathrm{d}\bm{X}^s_{i-2r,j} - \mathrm{d}\bm{X}^s_{i+2r,j}\qquad \mathrm{for}\, s = 0 \ldots n_s, i = 2r \ldots n_i-2r, j = 0 \ldots n_j

.. math::
	\mathrm{d_3}\bm{X}^s_{i,j} = \mathrm{d_2}\bm{X}^s_{i-2r,j} - \mathrm{d_2}\bm{X}^s_{i+2r,j}\qquad \mathrm{for}\, s = 0 \ldots n_s, i = 2r \ldots n_i-2r, j = 0 \ldots n_j

.. math::
	\mathrm{d_2}\bm{X}^s_{i,j} * \mathrm{d_2}\bm{X}^s_{i,j} &< d_2\\
	(\mathrm{d_3}\bm{X}^s_{i,j} * \mathrm{d_3}\bm{X}^s_{i,j}) / 2 &< d_3
	



.. plot:: aramis_cdt/example_01.py



