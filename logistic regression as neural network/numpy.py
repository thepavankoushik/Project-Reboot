#basic functions with numpy
##sigmoid function

import math
def basic_sigmoid(x):
	s = 1/(1+math.exp(-x))
	return s

print(basic_sigmoid(3))

import numpy as np
def sigmoid(x):
	s = 1/(1+np.exp(-x))
	return s

x = np.array([1,2,3])
print(sigmoid(x))

##sigmoid gradient

def sigmoid_derivative(x):
	s = 1/(1+np.exp(-x))
	ds = s*(1-s)
	return ds

print(sigmoid_derivative(x))

##reshaping arrays
def image2vector(image):
	v = image.reshape((image.shape[0]*image.shape[1]*image.shape[2]))
	return v
##normalizing rows
def normalize(x):
	x_norm = np.linalg.norm(x, axis = 1, keepdims = True)
	x = x/x_norm
	return x

##softmax
def softmax(x):
	x_exp = np.exp(x)
	x_sum = np.sum(x_exp, axis = 1, keepdims= True)
	s = x_exp/x_sum
	return s


#Loss function
def L1(yhat, y):
	loss = np.sum(abs(yhat-y))
	return L1
