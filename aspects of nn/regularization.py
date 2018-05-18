import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import scipy.io
from utils import *
trainx,trainy,testx,testy = loaddataset()

#ways to avoid overfitting the model(l2 or drop out)

#L2 regularization(also called weight decay)
def compute_cost_with_regularization(a3,y,params, lamda):
	m = y.shape[1]
	w1 = params["w1"]
	w2 = params["w2"]
	w3 = params["w3"]
	cross_entropy_cost = compute_cost(a3,y)
	l2 = (np.sum(np.square(w1))+np.sum(np.square(w2))+np.sum(np.square(w3)))
	cost = cross_entropy_cost + l2
	return cost


def backward_with_regularization(x,y,cache, lamda):
	m = y.shape[1]
	(z1,a1,w1,b1,z2,a2,w2,b2,z3,a3,w3,b3) = cache
	dz3 = a3-y
	dw3 = (np.dot(dz3, a2.T)/m) + (w3*lamda/m)
	db3 = np.sum(dz3)
	da2 = np.dot(dw3.T, dz3)
	dz2 = np.multiply(da2, np.int64(a2>0))
	dw2 = (np.dot(dz2, a1.T)/m) + (w2*lamda/m)
	db2 = np.sum(dz2)
	da1 = np.dot(dw2.T, dz2)
	dz1 = np.multiply(da1, np.int64(a1>0))
	dw1 = (np.dot(dz1, x.T)/m) + (w1*lamda/m)
	db1 = np.sum(dz1)
	grads = {"dz3":dz3,"dw3":dw3,"db3":db3,"da2":da2,"dz2":dz2,"dw2":dw2,"db2":db2,"da1":da1,"dz1":dz1,"dw1":dw1,"db1":db1}
	return grads


#dropout
def forward_dropout(x,params, keepprob = 0.5):
	w1,b1,w2,b2,w3,b3 = params
	z1 = np.dot(w1,x)+b1
	a1 = relu(z1)
	d1 = np.random.rand(a1.shape[0],a1.shape[1])
	d1 = (d1<keepprob)
	a1 = a1*d1
	a1 /= keepprob
	z2 = np.dot(w2,a1)+b2
	a2 = relu(z2)
	d2 = np.random.rand(a2.shape[0],a2.shape[1])
	d2 = (d2 < keepprob)
	a2 = a2*d2
	a2 /= keepprob
	z3 = np.dot(w3,a2)+b3
	a3 = sigmoid(z3)
	cache = (z1,d1,a1,z2,d2,a2,z3,a3)
	return a3, cache


def backward_dropout(x,y,cache, keepprob):
	m = x.shape[1]
	(z1,a1,d1,z2,a2,d2,z3,a3) = cache
	dz3 = a3-y
	dw3 = (np.dot(dz3, a2.T)/m) + (w3*lamda/m)
	db3 = np.sum(dz3)
	da2 = np.dot(dw3.T, dz3)
	da2 = da2 * d2
	da2 /= keepprob
	dz2 = np.multiply(da2, np.int64(a2>0))
	dw2 = (np.dot(dz2, a1.T)/m) + (w2*lamda/m)
	db2 = np.sum(dz2)
	da1 = np.dot(dw2.T, dz2)
	da1 = da1*d1
	da1 /= keepprob
	dz1 = np.multiply(da1, np.int64(a1>0))
	dw1 = (np.dot(dz1, x.T)/m) + (w1*lamda/m)
	db1 = np.sum(dz1)
	grads = {"dz3":dz3,"dw3":dw3,"db3":db3,"da2":da2,"dz2":dz2,"dw2":dw2,"db2":db2,"da1":da1,"dz1":dz1,"dw1":dw1,"db1":db1}
	return grads