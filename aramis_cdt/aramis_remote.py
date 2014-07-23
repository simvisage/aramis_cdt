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

class AramisRemote(HasTraits):
    '''Class managing the interaction with the remote server
    containing the processed aramis data.
    '''

    simdb = Instance(SimDB, ())
    '''Access to simdb used to derive the access to the remote server.
    '''

    server_username = DelegatesTo('simdb')
    '''User name used to access to the data at remote server.
    '''

    server_host = DelegatesTo('simdb')
    '''Host computer of the remote server.
    '''

    simdb_cache_remote_dir = DelegatesTo('simdb')

    simdb_dir = DelegatesTo('simdb')

    simdb_cache_dir = DelegatesTo('simdb')

    extended_data_cfg_filename = Str('extended_data.cfg')
    '''File name of a configuration file specifying the extended data for the
    current test run at the server.
    '''

    reload_experiments = Button
    '''Event triggering the reloading of the test data to the local cache.
    '''

    experiment_lst = Property(List(Str), depends_on='reload_experiments')
    '''List of test runs obtained obtained by scanning the simdb directory tree.
    '''
    @cached_property
    def _get_experiment_lst(self):
        experiment_lst = []
        for path, subdirs, files in os.walk(self.simdb_dir):
            for name in files:
                if name == self.extended_data_cfg_filename:
                    experiment_lst.append(path.replace(self.simdb_dir, '')[1:])
        return experiment_lst

    experiment_selected = Str
    '''Specification of the current experiment from within the experiment_lst.
    '''
    def _experiment_selected_default(self):
        return self.experiment_lst[0]

    aramis_files = Property(List(Str), depends_on='experiment_selected')
    '''Available aramis files obtained from configuration file for the currently selected experiment.
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
    '''Aramis file currently selected for (download, unzip, analyze)
    '''
    def _aramis_file_selected_default(self):
        return self.aramis_files[0]

    @on_trait_change('experiment_selected')
    def _reset_aramis_file_selected(self):
        self.aramis_file_selected = self.aramis_files[0]

    local_dir = Property(File)
    '''Relative path to the directory containing the selected aramis data
    '''
    def _get_local_dir(self):
        return os.path.join(self.simdb_cache_dir, self.experiment_selected, 'aramis')

    aramis_file_path = Property(File)
    '''Absolute path to the directory containing the selected aramis data
    '''
    def _get_aramis_file_path(self):
        return os.path.join(self.local_dir, self.aramis_file_selected)

    server_dir = Property(File)
    '''Remote path to the directory containing selected Aramis data
    '''
    def _get_server_dir(self):
        return os.path.join(self.simdb_cache_remote_dir, self.experiment_selected, 'aramis')

    zip_filename = Property(Str)
    '''Name of the zip file of the selected item
    '''
    def _get_zip_filename(self):
        return self.aramis_file_selected + '.zip'

    download = Button
    '''Prepare local data structure and download the zip file from server.
    '''
    def _download_fired(self):
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
