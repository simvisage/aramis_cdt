# -*- coding: utf-8 -*-
# Script: Adding of values into AD channel 0
#
# GOM-Script-Version: 6.3.0-7
#

import gom

# path to the exported file
fileproject_name = r'D:\Users\user\Documents\Aachen_tests_07_2013\TT-4c-V1\AD\AD_data.txt'

# project name to be added values
project_name = 'TT-4c-V1.dap'

# open and read values as list of strings
infile = open(fileproject_name, 'r')
data = infile.readlines()

# remove line break '\n'
for i in range(len(data)):
	data[i] = data[i][:-1]

# create "AD 0" in all stages and set default value to 0
try:
	gom.script.stagedata.create_stage_data (
		data=[gom.app.aramis_projects[project_name].stages['0']],
		target_key='ad channel 0',
		target_display='AD 0',
		value_type=gom.List ('number', ['number', 'time', 'text']),
		default_value='0')
except:
	print 'AD 0 exists!'

# update values of AD 0 by list of values
gom.script.stagedata.set_stage_data (
	data_stagedata=[gom.app.aramis_projects[project_name].stagedata['ad channel 0']],
	values=data)


