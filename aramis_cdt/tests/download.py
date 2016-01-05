#-------------------------------------------------------------------------
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
#-------------------------------------------------------------------------

import os
import platform
from pyface.api import ProgressDialog
import sys

import numpy as np


def downloadProgress(count, blockSize, totalSize):

    percent = count * blockSize * 100 / float(totalSize)
    sys.stdout.write("\r" + "Downloading... %3f%%" % percent)
    sys.stdout.flush()


if not os.path.exists('mri_data.tar.gz'):
    # Download the data
    import urllib

    site = 'http://www-graphics.stanford.edu/data/voldata/MRbrain.tar.gz'
    (file_, headers) = urllib.urlretrieve(
        site, 'mri_data.tar.gz', downloadProgress)
    print headers


# import zipfile
# zip_file = os.path.join(task_dir, 'mri_data.zip')
# zf = zipfile.ZipFile(zip_file, 'r')
# zf.extractall(task_dir)
# zf.close()
# print 'Results decompressed'

# # Extract the data
# import tarfile
# tar_file = tarfile.open('mri_data.tar.gz')
# try:
#     os.mkdir('mri_data')
# except:
#     pass
# tar_file.extractall('mri_data')
# tar_file.close()
