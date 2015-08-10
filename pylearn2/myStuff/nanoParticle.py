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
import numpy as np

class NANO_PARTICLE(dense_design_matrix.DenseDesignMatrix):
    """
    Running on data from a simulation run by Bobby with a small amount of particles and large number of snapshots

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
            raise ValueError(
            'Unrecognized which_set value "%s".' % (which_set,) +
            '". Valid values are ["train","test", "valid"].')

        if control.get_load_data():
            path = "${PYLEARN2_DATA_PATH}/nanoParticle/"
            #have to load in the whole array from the start anyway
            import h5py
            import numpy as np

            nParticles = 262144
            trainingFrac = .8
            validFrac = .1

            X = np.zeros((100, nParticles*6))
            y = np.zeros((100, nParticles*3))

            def getFilename(i):
                base = path+'snapshot_'
                if i<10:
                    out= base+'00%d.hdf5'%i
                elif i<100:
                    out= base+'0%d.hdf5'%i
                else:
                    out= base+'%d.hdf5'%i
                return serial.preprocess(out)

            for i in xrange(101):
                fname = getFilename(i)
                f = h5py.File(fname, 'r')
                coords = f['PartType1']['Coordinates'][()]
                if i!=0:
                    y[i-1,:] = coords.flatten()
                if i!=100:
                    vels = f['PartType1']['Velocities'][()]
                    data = np.concatenate((coords, vels), axis = 1).flatten()
                    X[i,:] = data
                f.close()

            if which_set == 'train':
                X = X[:, :nParticles*trainingFrac*6]
                y = y[:, :nParticles*trainingFrac*3]
            elif which_set == 'valid':
                X = X[:, nParticles*trainingFrac*6:nParticles*(trainingFrac+validFrac)*6]
                y = y[:, nParticles*trainingFrac*3:nParticles*(trainingFrac+validFrac)*3]
            else:
                assert which_set == 'test'
                X = X[:, nParticles*(trainingFrac+validFrac)*6:]
                y = y[:, nParticles*(trainingFrac+validFrac)*3:]

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
        assert m == 100

        n, s = y.shape
        assert n == 100

        #Shuffle used to be here, which I don't think is terrifically necessary
                                        #X=dimshuffle(X)
        print 'Initiating Superclass'
        super(NANO_PARTICLE, self).__init__(X=X, y=y)
        del X
        del y

        assert not np.any(np.isnan(self.X))

        #Changing to slice off particles, not rows.
        if start is not None:
            assert start >= 0
            if stop > self.X.shape[1]:
                raise ValueError('stop=' + str(stop) + '>' +
                                 'm=' + str(self.X.shape[1]))
            assert stop > start
            self.X = self.X[:, start:stop]
            if self.X.shape[1] != stop - start:
                raise ValueError("X.shape[1]: %d. start: %d stop: %d"
                                 % (self.X.shape[1], start, stop))
            if len(self.y.shape) > 1:
                self.y = self.y[:, start:stop]
            else:
                raise ValueError('y must have >1 dimension.')

            assert self.y.shape[1] == stop - start
        print 'Done'

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