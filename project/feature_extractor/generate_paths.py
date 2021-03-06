import os
import glob
import hdf5_getters
import pickle

def pickle_all_files(basedir,ext='.h5') :
    fles = []
    for root, dirs, files in os.walk(basedir):
        import pdb
        pdb.set_trace()
        files = glob.glob(os.path.join(root,'*'+ext))
        for f in files:
            fles.append(f)
    pickle.dump(fles, open('files.pkl', 'wb'))

def get_all_files():
    files = pickle.load(open('files.pkl', 'rb'))
    return files


def get_all_titles(basedir,ext='.h5') :
    titles = []
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root,'*'+ext))
        for f in files:
            h5 = hdf5_getters.open_h5_file_read(f)
            titles.append( hdf5_getters.get_title(h5) )
            h5.close()
    return titles
