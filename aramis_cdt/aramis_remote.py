#-------------------------------------------------------------------------------
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

    extended_data_cfg_filename = Str('extended_data.cfg')

    reload_experiments = Button

    experiment_lst = Property(List(Str), depends_on='reload_experiments')
    @cached_property
    def _get_experiment_lst(self):
        experiment_lst = []
        for path, subdirs, files in os.walk(self.simdb_dir):
            for name in files:
                if name == self.extended_data_cfg_filename:
                    experiment_lst.append(path.replace(self.simdb_dir, '')[1:])
        return experiment_lst

    experiment_selected = Str
    '''Pointer to the current experiment
    '''
    def _experiment_selected_default(self):
        return self.experiment_lst[0]

    aramis_files = Property(List(Str), depends_on='experiment_selected')
    '''Available Aramis files obtained from *.cfg file
    '''
    @cached_property
    def _get_aramis_files(self):
        path = os.path.join(self.simdb_dir,
                            self.experiment_selected,
                            self.extended_data_cfg_filename)
        if os.path.exists(path):
            config = ConfigParser.ConfigParser()
            config.read(path)
            return config.get('aramis_data', 'aramis_files').split(',\n')
        else:
            return ['extended file does not exist']

    aramis_file_selected = Str
    '''Selected Aramis file for analysis (download, unzip, analyze)
    '''
    def _aramis_file_selected_default(self):
        return self.aramis_files[0]

    @on_trait_change('experiment_selected')
    def _reset_aramis_file_slected(self):
        self.aramis_file_selected = self.aramis_files[0]

    local_dir = Property(File)
    '''Local path to the directory contained selected Aramis data
    '''
    def _get_local_dir(self):
        return os.path.join(self.simdb_cache_dir, self.experiment_selected, 'aramis')

    aramis_file_path = Property(File)
    '''Local path to the directory contained selected Aramis data
    '''
    def _get_aramis_file_path(self):
        return os.path.join(self.local_dir, self.aramis_file_selected)

    server_dir = Property(File)
    '''Remote path to the directory containing selected Aramis data
    '''
    def _get_server_dir(self):
        return os.path.join(self.simdb_cache_remote_dir, self.experiment_selected, 'aramis')

    zip_filename = Property(Str)
    '''Name of the *.zip file of the selected item
    '''
    def _get_zip_filename(self):
        return self.aramis_file_selected + '.zip'

    download = Button
    '''Prepare local data structure and download the *.zip file from server
    '''
    def _download_fired(self):
        print 'server', self.server_dir
        print 'zip', self.zip_filename
        print 'local', self.local_dir
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

    view = View(
                Item('simdb_dir', style='readonly'),
                Item('reload_experiments', show_label=False),
                Item('experiment_selected', editor=EnumEditor(name='experiment_lst')),
                # Item('extended_data_dir'),
                Item('server_username', style='readonly'),
                Item('server_host', style='readonly'),
                Item('simdb_cache_remote_dir', style='readonly'),
                Item('aramis_file_selected', editor=EnumEditor(name='aramis_files'),
                     label='Available Aramis files'),
                UItem('download'),
                UItem('decompress'),
                # UItem('set_dir')
                id='aramisCDT.remote',
                resizable=True
                )


if __name__ == '__main__':
    AR = AramisRemote()
    print 'experiment_list', AR.experiment_lst
    AR.configure_traits()
