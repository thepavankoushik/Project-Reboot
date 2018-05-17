import numpy as np
import matplotlib.pyplot as plt
import scipy
import h5py
    
    
def load_dataset():
    dataset = h5py.File('catvdog.hdf5', "r")
    train_set_x_orig = np.array(dataset["train_img"][:]) # your train set features
    train_set_y_orig = np.array(dataset["train_labels"][:]) # your train set labels

    
    test_set_x_orig = np.array(dataset["test_img"][:]) # your test set features
    test_set_y_orig = np.array(dataset["test_labels"][:]) # your test set labels

    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig


#loading the dataset
train_x, train_y, test_x, test_y = load_dataset()
m_train = train_x.shape[0]
m_test = test_x.shape[0]
nx = train_x.shape[1]

#image classification without convolution using only logistic R
#flatten the image
train_x = train_x.reshape(train_x.shape[0],-1).T
test_x = test_x.reshape(test_x.shape[0],-1).T
#normalizing
test_x = test_x/255
train_x = train_x/255

#helper functions
#sigmoid
def sigmoid(z):
	s = 1/(1+np.exp(-z))
	return s

#init of weights and bias(note: single layer only)
def init_weights(dim):
	w = np.zeros((dim, 1))
	b = 0
	return w,b

#forward propagation function
def propagate(w,b,X,Y):
	m = X.shape[1]
	A = sigmoid(np.dot(w.T, X)+b)
	cost = -np.sum((Y*np.log(A))+((1-Y)*np.log(1-A)))/m
	dw = np.dot(X, (A-Y).T)/m
	db = np.sum(A-Y)/m
	cost = np.squeeze(cost)
	grads = {"dw":dw,"db":db}
	return grads, cost


#backprop or optimizer
def optimize(w,b, X,Y, iterations, alpha):
	costs = []
	for i in range(iterations):
		grads, cost = propagate(w,b,X,Y)
		dw = grads["dw"]
		db = grads["db"]
		w -= alpha*dw
		b -= alpha*db
		if(i%100 == 0):
			costs.append(cost)
			print(i,cost)
	params = {"w":w,"b":b}
	grads = {"dw":dw, "db":db}
	return params, grads, costs


#predictor
def predict(w,b,X):
	m = X.shape[1]
	yhat = np.zeros((1,m))
	A = sigmoid(np.dot(w.T, X)+b)
	for i in range(A.shape[1]):
		if(A[0,i]<=0.5):
			yhat[0,i] = 0
		else:
			yhat[0,i] = 1
	return yhat


#model
def model(X_train, Y_train, X_test, Y_test, iterations, alpha):
	w,b = init_weights(X_train.shape[0])
	params, grads, costs = optimize(w,b,X_train, Y_train,iterations, alpha)
	w = params["w"]
	b = params["b"]
	yprediction = predict(w,b,X_test)
	print("test accuracy:", (100-np.mean(np.abs(yprediction-Y_test))*100))
	d = {"costs":costs, "yprediction":yprediction,"w":w,"b":b}
	return d


d = model(train_img, train_labels, test_img, test_labels, iterations = 2000, alpha = 0.005)
