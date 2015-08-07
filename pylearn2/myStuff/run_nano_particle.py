import os
import pylearn2
os.environ['PYLEARN2_DATA_PATH'] = '/home/sean/GitRepos/pylearn2/data'
path = os.path.join(pylearn2.__path__[0], 'myStuff', 'nano_particle_1.yaml')
nParticles = 100
with open(path, 'r') as f:
    train = f.read()
hyper_params = {'start' : 0,
                'stop' : nParticles*6,
                'output' : nParticles*3,
                'dim_h0' : 50,
                'max_epochs' : 100,
                'learning_rate': 0.01,
                'N_wait': 10,
                'save_path' : '.'}
train = train % (hyper_params)
#print train

from pylearn2.config import yaml_parse
train = yaml_parse.load(train)
train.main_loop()