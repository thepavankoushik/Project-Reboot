import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets
from utils import *
#minibatch gradient descent
def random_mini_batches(x,y, size = 64, seed = 9):
	np.random.seed(seed)
	m = x.shape[1]
	minibatches = []
	permuatation = list(np.random.permuatation(m))
	shuffledx = x[:, permuatation]
	shuffledy = y[:, permuatation]
	temp = m - size*(m//size)
	numofcompletebatches = math.floor(m/size)
	for k in range(numofcompletebatches):
		minibatchx = shuffledx[:,k*size:(k+1)*size]
		minibatchy = shuffledy[:,k*size:(k+1)*size]
		minibatches.append((minibatchx,minibatchy))
	if m%size != 0:
		minibatchx = shuffledx[:,k*size:(k*size)+temp]
		minibatchy = shuffledy[:,k*size:(k*size)+temp]
		minibatches.append((minibatchx,minibatchy))
	return minibatches


#momentum
def init_velocities(params):
	L = len(params)//2
	v = {}
	for l in range(l):
		v["dw"+str(l)] = np.zeros((params["w"+str(l)].shape))
		v["db"+str(l)] = np.zeros((params["b"+str(l)].shape))
	return v

def update_with_momentum(params, grads, v, beta, alpha):
	L = len(params)//2
	for l in range(L):
		v["dw"+str(l+1)] = beta*v["dw"+str(l+1)]+(1-beta)*grads["dw"+str(l+1)]
		v["db"+str(l+1)] = beta*v["db"+str(l+1)]+(1-beta)*grads["db"+str(l+1)]
		params["w"+str(l+1)] -= alpha*v["dw"+str(l+1)]
		params["b"+str(l+1)] -= alpha*v["db"+str(l+1)]
	return params, v


#adam

def init_adam(params):
	L = len(params)//2
	v = {}
	s = {}
	for l in range(l):
		v["dw"+str(l)] = np.zeros((params["w"+str(l)].shape))
		v["db"+str(l)] = np.zeros((params["b"+str(l)].shape))
		s["dw"+str(l)] = np.zeros((params["w"+str(l)].shape))
		s["db"+str(l)] = np.zeros((params["b"+str(l)].shape))
	return v,s

def update_with_adam(params, grads, v,s, beta1,beta2, alpha,t,epsilon = 1e -8):
	L = len(params)//2
	for l in range(L):
		v["dw"+str(l+1)] = beta1*v["dw"+str(l+1)]+(1-beta1)*grads["dw"+str(l+1)]
		v["db"+str(l+1)] = beta1*v["db"+str(l+1)]+(1-beta1)*grads["db"+str(l+1)]
		s["dw"+str(l+1)] = beta2*s["dw"+str(l+1)]+(1-beta2)*grads["dw"+str(l+1)]
		s["db"+str(l+1)] = beta2*s["db"+str(l+1)]+(1-beta2)*grads["db"+str(l+1)]
		v["dw"+str(l+1)] = v["dw"+str(l+1)]/(1-beta1**t)
		v["db"+str(l+1)] = v["db"+str(l+1)]/(1-beta1**t)
		s["dw"+str(l+1)] = s["dw"+str(l+1)]/(1-beta2**t)
		s["db"+str(l+1)] = s["db"+str(l+1)]/(1-beta2**t)
		params["w"+str(l+1)] -= alpha*(v["dw"+str(l+1)]/s["dw"+str(l+1)]**(1/2)+epsilon)
		params["b"+str(l+1)] -= alpha*(v["db"+str(l+1)]/s["db"+str(l+1)]**(1/2)+epsilon)
	return params, v, s

