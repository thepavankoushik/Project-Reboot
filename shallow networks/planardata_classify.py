#using a shallow net(2 layers)
#not tested
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from utils import load_dataset

#loading the dataset using utils
x,y = load_dataset()
shape_x = x.shape
shape_y = y.shape
m = shape_x[1]

#first trying to fit the data using sime LR
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(x.T, y.T)
print(clf.predict(x.T))

#now trying with multi layer nn model

#helper functions
def layer_sizes(x,y):
	nx = x.shape[0]
	nh = 4
	ny = y.shape[0]
	return(nx,nh,ny)

def init_weights(nx,nh,ny):
	np.random.seed(9)
	w1 = np.random.randn(nh,nx)*0.01
	b1 = np.zeros((nh,1))
	w2 = np.random.randn(ny,nh)*0.01
	b2 = np.zeros((ny,1))
	params = {"w1":w1, "b1":b1,"w2":w2,"b2":b2}
	return params

#forward propagation
def propagate(x,y,params):
	m = x.shape[1]
	w1 = params["w1"]
	b1 = params["b1"]
	w2 = params["w2"]
	b2 = params["b2"]
	z1 = np.dot(w1,x)+b1
	a1 = np.tanh(z1)
	z2 = np.dot(w2,a1)+b2
	a2 = 1/(1+np.exp(-z2))

	logprobs = np.multiply(np.log(a2),y)+np.multiply(np.log(1-a2),(1-y))
	cost =-np.sum(logprobs)/m
	cost = np.squeeze(cost)
	cache = {"z1":z1,"a1":a1,"z2":z2,"a2":a2}
	return cache, cost


#backprop for optimization

def optimize(params, cache, x,y, alpha = 1.2):
	m = x.shape[1]
	a1 = cache["a1"]
	a2 = cache["a2"]
	dz2 = a2-y
	dw2 = np.dot(dz2,a1.T)/m
	db2 = np.sum(dz2)/m
	dz1 = np.dot(w2.T, dz2)*(1-np.power(a1,2))
	dw1 = np.dot(dz1,x.T)/m
	db1 = np.sum(dz1)/m
	w1 = params["w1"]
	w2 = params["w2"]
	b1 = params["b1"]
	b2 = params["b2"]
	w1 -= alpha*dw1
	b1 -= alpha*db1
	w2 -= alpha*dw2
	b2 -= alpha*db2
	params = {"w1":w1,"b1":b1,"w2":w2,"b2":b2}
	grads = {"dw1":dw1,"db1":db1,"dw2":dw2,"db2":db2}
	return params, grads

#final model
def model(x,y,iterations):
	np.random.seed(9)
	nx = layer_sizes(x,y)[0]
	ny = layer_sizes(x,y)[2]
	nh = 4
	params = init_weights(nx,nh,ny)
	for i in range(iterations):
		cost, cache = propagate(x,y, params)
		params,grads = optimize(params, cache,x,y)
		if(i%1000 == 0):
			print(i,cost)

	return params

def predict(params, x):
	m = x.shape[1]
	w1 = params["w1"]
	b1 = params["b1"]
	w2 = params["w2"]
	b2 = params["b2"]
	z1 = np.dot(w1,x)+b1
	a1 = np.tanh(z1)
	z2 = np.dot(w2,a1)+b2
	a2 = 1/(1+np.exp(-z2))
	predictions = (a2>0.5)
	return predictions

params = model(x,y,10000)
predictions = predict(params, x)
print((np.dot(y,predictions.T)+np.dot(1-y, 1-predictions.T))/y.size)
