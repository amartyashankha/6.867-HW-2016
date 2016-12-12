"""
Thierry Bertin-Mahieux (2010) Columbia University tb2332@columbia.edu 
Code to quickly see the content of an HDF5 file.

This is part of the Million Song Dataset project from
LabROSA (Columbia University) and The Echo Nest.


Copyright 2010, Thierry Bertin-Mahieux

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import os
import sys
import hdf5_getters
import numpy as np
import timeit


def die_with_usage():
    """ HELP MENU """
    print 'display_song.py'
    print 'T. Bertin-Mahieux (2010) tb2332@columbia.edu'
    print 'to quickly display all we know about a song'
    print 'usage:'
    print '   python display_song.py [FLAGS] <HDF5 file> <OPT: song idx> <OPT: getter>'
    print 'example:'
    print '   python display_song.py mysong.h5 0 danceability'
    print 'INPUTS'
    print '   <HDF5 file>  - any song / aggregate /summary file'
    print '   <song idx>   - if file contains many songs, specify one'
    print '                  starting at 0 (OPTIONAL)'
    print '   <getter>     - if you want only one field, you can specify it'
    print '                  e.g. "get_artist_name" or "artist_name" (OPTIONAL)'
    print 'FLAGS'
    print '   -summary     - if you use a file that does not have all fields,'
    print '                  use this flag. If not, you might get an error!'
    print '                  Specifically desgin to display summary files'
    sys.exit(0)

from generate_paths import get_all_files

def get_features(hdf5path, feat = None):

    songidx = 0
    onegetter = '' if feat == None else feat
    features = {'path' : hdf5path}

    # sanity check
    if not os.path.isfile(hdf5path):
        print 'ERROR: file',hdf5path,'does not exist.'
        sys.exit(0)
    h5 = hdf5_getters.open_h5_file_read(hdf5path)
    numSongs = hdf5_getters.get_num_songs(h5)
    if songidx >= numSongs:
        print 'ERROR: file contains only',numSongs
        h5.close()
        sys.exit(0)

    # get all getters
    getters = filter(lambda x: x[:4] == 'get_', hdf5_getters.__dict__.keys())
    getters.remove("get_num_songs") # special case
    if onegetter == 'num_songs' or onegetter == 'get_num_songs':
        getters = []
    elif onegetter != '':
        if onegetter[:4] != 'get_':
            onegetter = 'get_' + onegetter
        try:
            getters.index(onegetter)
        except ValueError:
            print 'ERROR: getter requested:',onegetter,'does not exist.'
            h5.close()
            sys.exit(0)
        getters = [onegetter]
    getters = np.sort(getters)

    mx_time = -10.0
    # print them
    for getter in getters:
        try:
            start = timeit.timeit()
            res = hdf5_getters.__getattribute__(getter)(h5,songidx)
            end = timeit.timeit()
        except AttributeError, e:
            if summary:
                continue
            else:
                print e
                print 'forgot -summary flag? specified wrong getter?'
        features[getter[4:]] = res

    h5.close()
    
    return features
    
def get_features_simple(hdf5path, getters, limit=500):


    songidx = 0
    features = []

    h5 = hdf5_getters.open_h5_file_read(hdf5path)
    
    getters.sort()
    getters = map(lambda x: 'get_'+x, getters)

    # print them
    res = hdf5_getters.__getattribute__('get_year')(h5,songidx)
    if res < 1800:
        h5.close()
        return []
    for getter in getters:
        try:
            res = hdf5_getters.__getattribute__(getter)(h5,songidx)
        except AttributeError, e:
            res = None
        if isinstance(res, np.ndarray):
            if len(res) > limit:
                beg = (len(res)-limit)/2
                res = res[beg:beg+limit]
            res = res.tolist()
        features.append(res)

    h5.close()
    
    return features

def get_feature_list():
    files = get_all_files()
    return get_features(files[0]).keys()
