#-------------------------------------------------------------------------------
#
# Copyright (c) 2009, IMB, RWTH Aachen.
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in simvisage/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.simvisage.com/licenses/BSD.txt
#
# Thanks for using Simvisage open source!
#
# Created on Dec 21, 2011 by: rch

from etsproxy.traits.api import \
    HasTraits, Str, Property, cached_property, \
    Enum

import os.path

from os.path import expanduser

HOME_DIR = expanduser("~")
# build directory
BUILD_DIR = os.path.join(HOME_DIR, '.doc_build', 'aramis_cdt', 'docs')
# output directory for the documentation
DOCS_DIR = os.path.join('..', 'docs',)
# output directory for the example documentation
EX_OUTPUT_DIR = os.path.join(DOCS_DIR, '_component_modules')

class GenDoc(HasTraits):
    '''
    Configuration of the document generation using sphinx.
    '''
    component_modules = []

    build_mode = Enum('local', 'global')

    build_dir = Property(depends_on='build_mode')
    def _get_build_dir(self):
        build_dir = {'local' : '.',
                     'global' : BUILD_DIR }
        return build_dir[self.build_mode]

    html_server = None  # 'root@mordred.imb.rwth-aachen.de:/var/www/docs/oricrete'

    method_dispatcher = {'all' : 'generate_examples' }

    def generate_html(self):
        os.chdir(DOCS_DIR)
        sphings_cmd = 'sphinx-build -b html -E . %s' % self.build_dir
        os.system(sphings_cmd)

    def push_html(self):
        '''
        Push the documentation to the server.
        '''
        rsync_cmd = 'rsync -av --delete %s/ %s' % (self.build_dir, self.html_server)
        os.system(rsync_cmd)

if __name__ == '__main__':

    gd = GenDoc(build_mode='global')
    # gd.generate_examples() # kind = 'sampling_efficiency')
    gd.generate_html()
    # gd.push_html()
