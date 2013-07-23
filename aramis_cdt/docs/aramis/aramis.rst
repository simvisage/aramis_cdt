
Aramis
============

.. todo::
    describe Aramis

Add (edit) AD channel values additionally
-----------------------------------------

AD channels can be added or edited after experiment. The following information
are usable for the case when Aramis and testing machine were synchronized on request 
(turned on at the "same" time). The code below prepare data to be added in the 
Aramis project.

.. literalinclude:: prepare_AD_data.py
    :encoding: latin-1

In the software Aramis, user can create own macro (Python script). The simplest
way is to record script and modify it. The following script creates variable 
``ad channel 0`` and updates its values to new ones.

.. literalinclude:: add_AD_channel_values.py
    :encoding: latin-1



Export data for AramisCDT
-------------------------
    
Default settings for export data from Aramis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: default_export_settings.txt
    :encoding: latin-1
  
Export undeformed coordinates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. todo::
	not implemented yet

Undeformed coordinates should be exported only for stage number 0 as file with  
name ``coord_ProjectName-AutomaticPart.txt`` (e.g. coord_TT-4c-V1-Stage-0-0.txt). 

.. literalinclude:: coord_export_settings.txt
    :encoding: latin-1
   
Export displacements
^^^^^^^^^^^^^^^^^^^^

.. todo::
	not implemented yet

Displacement have to be exported for all stages as file with  
name ``displ_ProjectName-AutomaticPart.txt`` (e.g. displ_TT-4c-V1-Stage-0-i.txt).

.. literalinclude:: displ_export_settings.txt
    :encoding: latin-1

    
    
    