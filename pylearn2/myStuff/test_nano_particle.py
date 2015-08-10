'''
This file makes pred_mat using a saved model
'''

import numpy as np
import sys
import os
import cPickle as pickle
import gzip
import pylearn2

from pylearn2.utils import serial
from theano import tensor as T
from theano import function

from nanoParticle import getTestSet

#os.environ['PYLEARN2_DATA_PATH'] = '/home/mclaughlin6464/GitRepos/pylearn2/data'
os.environ['PYLEARN2_DATA_PATH'] = '/media/Backup'

#This function is from predict_csv in mlp. However, I'm tweaking it to return the prediction!
def predict(model_path, x, predictionType="regression", outputType="float",
            headers=False, first_col_label=False, delimiter=","):
    """
    Predict from a pkl file.

    Parameters
    ----------
    modelFilename : str
        The file name of the model file.
    testFilename : str
        The file name of the file to test/predict.
    outputFilename : str
        The file name of the output file.
    predictionType : str, optional
        Type of prediction (classification/regression).
    outputType : str, optional
        Type of predicted variable (int/float).
    headers : bool, optional
        Indicates whether the first row in the input file is feature labels
    first_col_label : bool, optional
        Indicates whether the first column in the input file is row labels (e.g. row numbers)
    """

    print("loading model...")

    try:
        model = serial.load(model_path)
    except Exception as e:
        print("error loading {}:".format(model_path))
        print(e)
        return False

    print("setting up symbolic expressions...")

    X = model.get_input_space().make_theano_batch()
    Y = model.fprop(X)

    if predictionType == "classification":
        Y = T.argmax(Y, axis=1)

    f = function([X], Y, allow_input_downcast=True)

    print("loading data and predicting...")

    # x is a numpy array
    # x = pickle.load(open(test_path, 'rb'))

    if first_col_label:
        x = x[:,1:]

    y = f(x)
    return y

    #print("writing pred_mat...")

    #variableType = "%d"
    #if outputType != "int":
    #    variableType = "%f"

    #np.savetxt(output_path, y, fmt=variableType)
    #return True

# load the testing set to get the labels
nParticles = 50
test_data, test_labels = getTestSet(nParticles, 95, 100)

path = os.path.join(pylearn2.__path__[0], 'myStuff', sys.argv[1] )
pred_mat = predict(path, test_data)

from scipy.stats import pearsonr
from sklearn.metrics import r2_score

for i in xrange(nParticles):
    R2 = r2_score(test_labels[:,i], pred_mat[:,i])
    pearR, pvalue2 = pearsonr(test_labels[:,i], pred_mat[:,i])
    print 'Particle %d:\tR^2:%.3f\tPearson R:%.3f'%(i,R2, pearR)

from plot_helper import *
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

#sns.palplot(sns.cubehelix_palette(18, start=2, rot=0.1, dark=0, light=0.99))
cold_cmap = sns.cubehelix_palette(18, start=2, rot=0.1, dark=0, light=0.99, as_cmap=True)

sns.set_style("white")
sns.set_style("ticks")

def hexPlot(true_labels, pred_labels):
    g = sns.JointGrid(true_labels,pred_labels)
    g.plot_marginals(sns.distplot, color=".5")
    g.plot_joint(plt.hexbin, bins='log', gridsize=30, cmap=cold_cmap, extent=[0, np.max(true_labels), 0, np.max(true_labels)])
    a=np.linspace(0,max(true_labels),20)
    plt.plot(a,a,'k--')
    plt.xlim([0,np.max(true_labels)])
    plt.ylim([0,np.max(true_labels)])
    cax = g.fig.add_axes([1, 0.20, .01, 0.5])
    cb = plt.colorbar(cax=cax)
    cb.set_label('$\log_{10}(\mathcal{N})$')
    plt.show()

[hexPlot(test_labels[:,i], pred_mat[:,i]) for i in xrange(nParticles)]
