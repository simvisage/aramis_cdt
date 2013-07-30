.. Aramis Crack Detection Tool documentation master file, created by
   sphinx-quickstart on Sun Jul 21 12:03:21 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Aramis Crack Detection Tool's documentation!
=======================================================

.. default-domain:: py

.. highlight:: python
    :linenothreshold: 5


Tutorial
--------
    
.. toctree::
   :maxdepth: 2

   front
   intro
   aramis/aramis
   aramis_cdt/aramis_cdt
   examples/examples


Todo list
---------
	
	1. rename experiments as in exp_db
	2. change **a** in Xf19a15 to e.g. **d**
	3. implement new data structure (see bellow not implemented yet)
	4. download data from http (prepared) 
	5. average crack number from sub-crack information (number of sub-crack * grid size y (cca 2 mm) / specimen width) and spacing
	6. read info from exp_db
	7. clean the code (temporary plots in aramis_cdt)
	8. default directory for data - ~/.aramis_simdb
	9. decompile facet parameters from file name
	10. merge run in time and run back in time
	11. fix rotation in transform_data

.. todolist::


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

