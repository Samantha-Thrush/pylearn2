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

os.environ['PYLEARN2_DATA_PATH'] = '/home/mclaughlin6464/GitRepos/pylearn2/data'

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

test_data, test_labels = pickle.load(gzip.open(os.environ['PYLEARN2_DATA_PATH']+'/milleniumSAMs/'+'milliTest.pickle.gz', 'rb'))
minMaxValues = pickle.load(open(os.environ['PYLEARN2_DATA_PATH']+'/milleniumSAMs/'+'minMaxVals.pkl'))

path = os.path.join(pylearn2.__path__[0], 'myStuff', sys.argv[1] )
pred_mat = predict(path, test_data)

#un-normalize the data
for i in xrange(201):
    minval, maxval = minMaxValues[i]
    if i == 194:#not used
        continue
    elif i<193:#features
        test_data[:,i] = test_data[:,i]*(maxval-minval)+minval
    else:#labels/values
        j = i-195
        test_labels[:,j] = test_labels[:,j]*(maxval-minval)+minval
        pred_mat[:,j] = pred_mat[:,j]*(maxval-minval)+minval

from scipy.stats import pearsonr
from sklearn.metrics import r2_score
labels = ['Stellar Mass', 'Cold Gas Mass', 'Bulge Mass', 'Hot Gas Mass', 'Cooling Radius', 'Blk Hle Mass']

for i in xrange(6):
    R2 = r2_score(test_labels[:,i], pred_mat[:,i])
    pearR, pvalue2 = pearsonr(test_labels[:,i], pred_mat[:,i])
    print '%s:\tR^2:%.3f\tPearson R:%.3f'%(labels[i],R2, pearR)

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np


from plot_helper import *

mse_nn = mse(test_labels, pred_mat)
[genplots_M(np.c_[test_labels[:,i], pred_mat[:,i]], mse_nn[i]) for i in [x for x in xrange(0,6) if x != 4]]
genplots_M(np.c_[test_labels[:,4], pred_mat[:,4]], mse_nn[4], plot_type='R')

#This plot was throwing and error and I couldn't figure out why.
#I won't sweat it for now; can dive in later.
#plot_smhm(test_labels[:,0], pred_mat[:,0], test_data[:,0])
plot_bhbulge(test_labels[:,5], pred_mat[:,5], test_labels[:,2], pred_mat[:,2])
plot_coldgasfrac(test_labels[:,1], pred_mat[:,1], test_labels[:,0], pred_mat[:,0])