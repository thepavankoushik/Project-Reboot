import numpy as np
import h5py
import matplotlib.pyplot as plt
from utils import sigmoid, sigmoid_backward, relu, relu_backward
np.random.seed(9)

#weights inits
def init_weights(layer_dims):
	params = {}
	L = len(layer_dims)
	for l in range(1,L):
		params["w"+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])
		params["b"+str(l)] = np.zeros((layer_dims[l],1))
	return params

def linear_forward(a,w,b):
	z = np.dot(w,a)+b
	cache = (a,w,b)
	return z,cache

def linear_activation_forward(a_prev, w,b, activation):
	z, linear_cache = linear_forward(a_prev, w,b)
	if activation == "sigmoid":
		a, activation_cache = sigmoid(z)
	else:
		a, activation_cache = relu(z)
	cache = (linear_cache, activation_cache)
	return a,cache


#model_forward
def l_model_forward(x,params):
	caches = []
	a = x
	L = len(params)//2
	for l in range(1,L):
		a_prev = a
		a, cache = linear_activation_forward(a_prev, params["w"+str(l)], params["b"+str(l)], "relu")
		caches.append(cache)
	al, cache = linear_activation_forward(a, params["w"+str(l+1)], params["b"+str(l+1)],"sigmoid")
	caches.append(cache)
	return al, caches

def compute_cost(al,y):
	m = y.shape[1]
	cost = -np.sum(np.multiply(np.log(al),y)+np.multiply(np.log(1-al),(1-y)))/m
	cost = np.squeeze(cost)
	return cost

#backprop
def linear_backward(dz, cache):
	a_prev, w,b = cache
	m = a_prev.shape[1]
	dw = np.dot(dz, a_prev.T)/m
	db = np.su(dz, axis= 1, keepdims = True)/m
	da_prev = np.dot(w.T, dz)
	return da_prev, dw, db

def linear_activation_backward(da, cache, activation):
	linear_cache, activation_cache = cache
	if activation == "relu":
		dz = relu_backward(da, activation_cache)
		da_prev, dw, db = linear_backward(dz, linear_cache)
	else:
		dz = sigmoid_backward(da, activation_cache)
		da_prev, dw, db = linear_backward(dz, linear_cache)
	return da_prev, dw, db


def l_model_backward(al,y, cachce):
	grads = {}
	L = len(cache)
	m = al.shape[1]
	y = y.reshape(al.shape)
	dal = -(np.divide(y,al)-np.divide(1-y, 1-al))
	current_cache = cache[-1]
	grads["da"+str(l-1)], grads["dw"+str(l)], grads["db"+str(l)] = linear_activation_backward(dal, current_cache, "sigmoid")
	for l in reversed(range(L-1)):
		current_cache = cache[l]
		grads["da"+str(l-1)], grads["dw"+str(l)], grads["db"+str(l)] = linear_activation_backward(grads["da"+str(l)], current_cache, "sigmoid")
	return grads


def update_params(params, grads, alpha):
	L = len(params)//2
	for l in range(L):
		params["w"+str(l+1)] -= alpha*grads["w"+str(l+1)]
		params["b"+str(l+1)] -= alpha*grads["b"+str(l+1)]
	return params



