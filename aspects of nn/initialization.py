import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
#utils is custom
from utils import *

#inits of weights with zeros(symmetry problem)- 50% accuracy
def init_zeros(layer_dims):
	params = {}
	L = len(layer_dims)
	for l in range(1,L):
		params["w"+str(l)] = np.zeros((layer_dims[l], layer_dims[l-1]))
		params["b"+str(l)] = np.zeros((layer_dims[l],1))
	return params

#inits of weights with random numbers from a distribution(exploding or vanishing gradients problem)- 80% accuracy
def init_random(layer_dims):
	params = {}
	L = len(layer_dims)
	for l in range(1,L):
		params["w"+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*10
		params["b"+str(l)] = np.zeros((layer_dims[l],1))
	return params

#inits of weights with random numbers with adjusted variance(He inits used for relu activations in model) - 95% accuracy
def init_he(layer_dims):
	params = {}
	L = len(layer_dims)
	for l in range(1,L):
		params["w"+str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*np.sqrt(2/layer_dims[l-1])
		params["b"+str(l)] = np.zeros((layer_dims[l],1))
	return params

#another init is xavier(var = 1/n) used while tanh activation is used
