import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from stepbystepdnn import *
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

layer_dims = [12288, 20, 7, 5, 1]
def l_layer_model(x,y,layer_dims, alpha=0.0075, iterations):
	costs = []
	params = init_weights(layer_dims)
	for i in range(iterations):
		al,caches = l_model_forward(x, params)
		cost = compute_cost(al,y)
		grads = l_model_backward(al,y,caches)
		params = update_params(params, grads, alpha)
		if(i%100 == 0):
			print(i, cost)
			costs.append(cost)
	plt.plot(costs)
	plt.show()
	return params

#predict function need to be written
def predict(x,y,params):
	al,_ = l_model_forward(x, params)
	m = x.shape[1]
	yhat = np.zeros((1,m))
	al = sigmoid(np.dot(w.T, X)+b)
	for i in range(A.shape[1]):
		if(al[0,i]<=0.5):
			yhat[0,i] = 0
		else:
			yhat[0,i] = 1
	print("test accuracy:", (100-np.mean(np.abs(yhat-y))*100))
	return yhat
params = l_layer_model(train_x, train_y, test_x, test_y, layer_dims, iterations = 2500)
pred_train = predict(test_x, test_y, params)