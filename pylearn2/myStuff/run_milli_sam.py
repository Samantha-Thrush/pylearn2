from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import pylearn2
os.environ['PYLEARN2_DATA_PATH'] = '/notebooks/ml-sims/data'
path = os.path.join(pylearn2.__path__[0], 'myStuff', 'milli_sam_1.yaml')
with open(path, 'r') as f:
    train = f.read()
hyper_params = {'dim_h0' : 20,
                'max_epochs' : 1000,
                'learning_rate': 0.01,
                'N_wait': 10,
                'save_path' : '.'}
train = train % (hyper_params)
#print train

from pylearn2.config import yaml_parse
train = yaml_parse.load(train)
train.main_loop()
'''
#Now, do the predictions
import numpy as np
import sys
import cPickle, gzip

from pylearn2.utils import serial
from theano import tensor as T
from theano import function

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

    #print("writing predictions...")

    #variableType = "%d"
    #if outputType != "int":
    #    variableType = "%f"

    #np.savetxt(output_path, y, fmt=variableType)
    #return True

# load the testing set to get the labels
path = os.path.join(pylearn2.__path__[0], 'myStuff', 'mlp_2_best.pkl' )
test_data, test_labels = cPickle.load(gzip.open('milliTest.pickle.gz', 'rb'))

os.environ['PYLEARN2_DATA_PATH'] = '/home/mclaughlin6464/GitRepos/pylearn2/data'
#have model name as input? I don't even know what it'll necessarily be.
path = os.path.join(pylearn2.__path__[0], 'myStuff', 'mlp_2_best.pkl' )

pred_mat = predict(path, test_data)

#TODO use sklearn's version
def negative_log_liklihood(y, y_pred):
    return np.sum((y_pred-y)**2, axis = 0)

SST = negative_log_liklihood(test_labels, test_labels.mean(axis = 0))
SSR = negative_log_liklihood(test_labels, pred_mat)


R2 = 1 - SSR/SST
for i in xrange(test_labels.shape[1]):
    print 'R^2 %d is '%(i+1),R2[i]

#TODO get Harshil's Hexbins in here.
from matplotlib import pyplot as plt

for idx in xrange(6):

    plt.title('Relationship')
    plt.plot(np.linspace(test_labels[:,idx].min(), test_labels[:,idx].max()), np.linspace(test_labels[:,idx].min(), test_labels[:,idx].max()), 'r--')
    plt.scatter(test_labels[:,idx],pred_mat[:, idx])
    plt.show()
'''
