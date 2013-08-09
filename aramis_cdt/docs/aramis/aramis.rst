
Aramis
============

.. todo::
    describe Aramis
    
Conventions
--------------------

.. todo::
    describe naming and data convention
    
Facet field can be ``rectangular`` or ``quadrangle``. These options are available
in ``Project Mode`` in ``Project >> Project Parameter``.
    
+-----------------------------------------+----------------------------------------------+------------------------------------------+
|  Rectangular facet field                |  Masked rectangular facet field              |   Quadrangle facet field                 |
+-----------------------------------------+----------------------------------------------+------------------------------------------+
| .. image:: facet_field_rectangular.png  | .. image:: facet_field_rectangular_mask.png  |  .. image:: facet_field_quadrangle.png   | 
|    :width: 300px                        |    :width: 300px                             |     :width: 300px                        |
|                                         |                                              |                                          |
+-----------------------------------------+----------------------------------------------+------------------------------------------+

Add (Edit) AD channel values additionally
-----------------------------------------

AD channels can be added or edited after experiment. The following information
are usable for the case when Aramis and testing machine were synchronized on request 
(turned on at the "same" time). The code below prepare data to be added in the 
Aramis project. 

.. toctree::
   prepare_AD_data

In the software Aramis, user can create own macro (Python script). The simplest
way is to record script and modify it. The following script creates variable 
``ad channel 0`` and updates its values to new ones.

.. toctree::
   add_AD_channel_values


Export data for AramisCDT
-------------------------

.. toctree::
   default_export_settings
 
Export undeformed coordinates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Undeformed coordinates should be exported only for stage number 0 as file with  
name ``coord_ProjectName-AutomaticPart.txt`` (e.g. undeformed_coords-Stage-0-0.txt). 

.. toctree::
   coord_export_settings
   
Export displacements
^^^^^^^^^^^^^^^^^^^^

Displacement have to be exported for all stages as file with  
name ``displ_ProjectName-AutomaticPart.txt`` (e.g. displ-Stage-0-i.txt).

.. toctree::
   displ_export_settings

    
    
    