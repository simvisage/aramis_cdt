#-------------------------------------------------------------------------------
#
# Copyright (c) 2013
# IMB, RWTH Aachen University,
# ISM, Brno University of Technology
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in the AramisCDT top directory "licence.txt" and may be
# redistributed only under the conditions described in the aforementioned
# license.
#
# Thanks for using Simvisage open source!
#
#-------------------------------------------------------------------------------

from etsproxy.traits.api import \
    HasTraits, Float, Property, cached_property, Int, Array, Bool, Directory, \
    Instance, DelegatesTo, Tuple, Button, List, Str, File, Enum, on_trait_change

from etsproxy.traits.ui.api import View, Item, HGroup, EnumEditor, Group, UItem, RangeEditor

import numpy as np
from scipy import stats
import os
import re
import ConfigParser

from matresdev.db.simdb import SimDB
from sftp_server import SFTPServer
import zipfile

import platform
import time
if platform.system() == 'Linux':
    sysclock = time.time
elif platform.system() == 'Windows':
    sysclock = time.clock

from aramis_info import AramisInfo

class AramisRemote(HasTraits):

    aramis_info = Instance(AramisInfo)

    simdb = Instance(SimDB, ())

    server_username = DelegatesTo('simdb')

    server_host = DelegatesTo('simdb')

    simdb_cache_remote_dir = DelegatesTo('simdb')

    simdb_dir = DelegatesTo('simdb')

    simdb_cache_dir = DelegatesTo('simdb')

    experiment_dir = Directory
    '''Directory of the experiment containing extended data file
    '''

    relative_path = Str
    '''Relative path inside database structure - the path is same for experiment
    in both database structures (remote and local)
    '''
    @on_trait_change('experiment_dir')
    def _relative_path_update(self):
        self.relative_path = self.experiment_dir.replace(self.simdb_dir, '')[1:]

    extended_data_ini = File
    '''Extended data file 'extended_data.ini' contain the information about
    available Aramis files
    '''
    @on_trait_change('experiment_dir')
    def _extended_data_ini_update(self):
        self.extended_data_ini = os.path.join(self.experiment_dir, 'extended_data.cfg')

    aramis_files = List(Str)
    '''Available Aramis files obtained from *.cfg file
    '''
    @on_trait_change('extended_data_ini')
    def _aramis_files_update(self):
        if os.path.exists(self.extended_data_ini):
            config = ConfigParser.ConfigParser()
            config.read(self.extended_data_ini)
            lst = ['']
            lst += config.get('aramis_data', 'aramis_files').split(',\n')
            self.aramis_files = lst
        else:
            self.aramis_files = ['extended file does not exist']

    aramis_file_selected = Str
    '''Selected Aramis file for analysis (download, unzip, analyze)
    '''

    local_dir = Property(File)
    '''Local path to the directory contained selected Aramis data
    '''
    def _get_local_dir(self):
        return os.path.join(self.simdb_cache_dir, self.relative_path, 'aramis')

    server_dir = Property(File)
    '''Remote path to the directory contained selected Aramis data
    '''
    def _get_server_dir(self):
        return os.path.join(self.simdb_cache_remote_dir, self.relative_path, 'aramis')

    zip_filename = Property(Str)
    '''Name of the *.zip file of the selected item
    '''
    def _get_zip_filename(self):
        return self.aramis_file_selected + '.zip'

    download = Button
    '''Prepare local data structure and download the *.zip file from server
    '''
    def _download_fired(self):
        import paramiko
        if not os.path.exists(self.local_dir):
            os.makedirs(self.local_dir)
        try:
            s = SFTPServer(self.server_username, '', self.server_host)
            if hasattr(s, 'sftp'):
                s.download(os.path.join(self.server_dir, self.zip_filename),
                           os.path.join(self.local_dir, self.zip_filename))
                s.sftp.stat(os.path.join(self.server_dir, self.zip_filename))
                s.close()
        except IOError, e:
            print e

    decompress = Button
    '''Decompress downloaded file in local directory
    '''
    def _decompress_fired(self):
        zip_file = os.path.join(self.local_dir, self.zip_filename)
        zf = zipfile.ZipFile(zip_file, 'r')
        zf.extractall(self.local_dir)
        zf.close()
        print 'FILE %s DECOMPRESSED' % self.aramis_file_selected

    set_dir = Button
    def _set_dir_fired(self):
        self.aramis_info.data_dir = os.path.join(self.local_dir, self.aramis_file_selected)

    view = View(
                Item('experiment_dir'),
                Item('server_username', style='readonly'),
                Item('server_host', style='readonly'),
                Item('simdb_cache_remote_dir', style='readonly'),
                Item('aramis_file_selected', editor=EnumEditor(name='aramis_files'),
                     label='Available Aramis files'),
                UItem('download'),
                UItem('decompress'),
                UItem('set_dir')
                )


if __name__ == '__main__':
    AR = AramisRemote(experiment_dir='/home/kelidas/simdb/exdata/tensile_tests/buttstrap_clamping/2013-07-09_TTb-4c-2cm-0-TU_bs4-Aramis3d')
    AR.configure_traits()
