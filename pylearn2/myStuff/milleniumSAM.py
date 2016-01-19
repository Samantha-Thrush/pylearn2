"""
Data from the Millenium dataset
"""
__authors__ = "Sean McLaughlin"
__copyright__ = "lol"
__credits__ = ["Sean McLaughlin"]
__license__ = "lol"
__maintainer__ = "LOL"
__email__ = "mclaughlin6464@gmail.com"

import numpy as np
import cPickle as pickle
import gzip
#from theano.compat.six.moves import xrange
from pylearn2.datasets import dense_design_matrix
from pylearn2.datasets import control
from pylearn2.datasets import cache
from pylearn2.utils import serial
from pylearn2.utils.rng import make_np_rng

#NOTE: I pre-split the data up. It'd be possible to do that here, too, if the interest was in splitting it up in different amounts.

class MILLI_SAM(dense_design_matrix.DenseDesignMatrix):
    """
    The Millenium dataset SAMs results

    Parameters
    ----------
    which_set : str
        'train', 'valid',or 'test'
    start : Where to start slicing. Will start at 0 if None
    stop : Where to stop slicing. Will not slice if None
    """

    def __init__(self, which_set, start=None, stop=None):

        self.args = locals()

        if which_set not in ['train','valid', 'test']:
            if which_set == 'valid':
                raise ValueError(
                'Unrecognized which_set value "%s".' % (which_set,) +
                '". Valid values are ["train","test"].')

        if control.get_load_data():
            path = "${PYLEARN2_DATA_PATH}/"
            if which_set == 'train':
                data_path = path + 'milliTrain.pickle.gz'
            elif which_set == 'valid':
                data_path = path + 'milliValid.pickle.gz'
            else:
                assert which_set == 'test'
                data_path = path + 'milliTest.pickle.gz'

            # Path substitution done here in order to make the lower-level
            # mnist_ubyte.py as stand-alone as possible (for reuse in, e.g.,
            # the Deep Learning Tutorials, or in another package).
            data_path = serial.preprocess(data_path)

            # Locally cache the files before reading them
            #Not sure if it's necessary, but why not?
            datasetCache = cache.datasetCache
            data_path = datasetCache.cache_file(data_path)

            X, y  = pickle.load(gzip.open(data_path))
        else:
            #I don't know when this would be called, or why?
            #It should generate random data of the same dimensions, but I'm not gonna bother doing that.
            #This is the old code for the MNIST images
            if which_set == 'train':
                size = 60000
            elif which_set == 'test':
                size = 10000
            else:
                raise ValueError(
                    'Unrecognized which_set value "%s".' % (which_set,) +
                    '". Valid values are ["train","test"].')
            topo_view = np.random.rand(size, 28, 28)
            y = np.random.randint(0, 10, (size, 1))

        m, r = X.shape
        assert r == 193

        #Shuffle used to be here, which I don't think is terrifically necessary
                                        #X=dimshuffle(X)
        super(MILLI_SAM, self).__init__(X=X, y=y)

        assert not np.any(np.isnan(self.X))

        if start is not None:
            assert start >= 0
            if stop > self.X.shape[0]:
                raise ValueError('stop=' + str(stop) + '>' +
                                 'm=' + str(self.X.shape[0]))
            assert stop > start
            self.X = self.X[start:stop, :]
            if self.X.shape[0] != stop - start:
                raise ValueError("X.shape[0]: %d. start: %d stop: %d"
                                 % (self.X.shape[0], start, stop))
            if len(self.y.shape) > 1:
                self.y = self.y[start:stop, :]
            else:
                self.y = self.y[start:stop]
            assert self.y.shape[0] == stop - start

    def adjust_for_viewer(self, X):
        """
        .. todo::

            WRITEME
        """
        return np.clip(X * 2. - 1., -1., 1.)

    def adjust_to_be_viewed_with(self, X, other, per_example=False):
        """
        .. todo::

            WRITEME
        """
        return self.adjust_for_viewer(X)

    def get_test_set(self):
        """
        .. todo::

            WRITEME
        """
        args = {}
        args.update(self.args)
        del args['self']
        args['which_set'] = 'test'
        args['start'] = None
        args['stop'] = None
        args['fit_preprocessor'] = args['fit_test_preprocessor']
        args['fit_test_preprocessor'] = None
        return MILLI_SAM(**args)
