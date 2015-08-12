import os
import pylearn2
os.environ['PYLEARN2_DATA_PATH'] = '/media/Backup'
path = os.path.join(pylearn2.__path__[0], 'myStuff', 'nano_particle_1.yaml')

nParticles = 1
with open(path, 'r') as f:
    train = f.read()
hyper_params = {'nParticles' : nParticles,
                'start' : 0,
                'stop' : 90,
                'nvis': nParticles*6,
                'output' : nParticles*3,
                'valid_start' : 90,
                'valid_stop' : 95,
                'test_start': 95,
                'test_stop' : 100,
                'dim_h0' : 2,
                'max_epochs' : 500,
                'learning_rate': 0.01,
                'N_wait': 10,
                'save_path' : '.'}
train = train % (hyper_params)
#print train

from pylearn2.config import yaml_parse
train = yaml_parse.load(train)
train.main_loop()