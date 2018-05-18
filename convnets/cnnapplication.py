#sign detection problem(classification)
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from utils import *

trainx, trainy, testx, testy = load_dataset()
trainx /= 255
testx /= 255
testy = convert_to_one_hot(testy,6).T
trainy = convert_to_one_hot(trainy,6).T

def create_placeholders(nh0,nw0,nc0,ny):
	x = tf.placeholder("float",shape = (None, nh0, nw0, nc0))
	y = tf.placeholder("float", shape = (None, ny))
	return x,y

def init_params():
	w1 = tf.get_variable("w1",[4,4,3,8],initializer = tf.contrib.layers.xavier_initializer_conv2d(seed=0))
	w2 = tf.get_variable("w2",[2,2,8,16],initializer = tf.contrib.layers.xavier_initializer_conv2d(seed = 0))
	params = {"w1":w1, "w2":w2}
	return params

def forward_prop(x,params):
	w1 = params["w1"]
	w2 = params["w2"]
	z1 = tf.nn.conv2d(x,w1,[1,1,1,1],padding = "same")
	a1 = tf.nn.relu(z1)
	p1 = tf.nn.max_pool(a1,ksize = [1,8,8,1],strides = [1,8,8,1], padding = "same")
	z2 = tf.nn.conv2d(p1,w2,[1,1,1,1],padding = "same")
	a2 = tf.relu(z2)
	p2 = tf.nn.max_pool(a2, ksize = [1,4,4,1], strides = [1,4,4,1], padding = "same")
	p2 = tf.nn.contrib.layers.flatten(p2)
	z3 = tf.contrib.layers.fully_connected(p2,num_outputs = 6, activation_fn = None)
	return z3



def compute_cost(z3,y):
	cost = tf.nn.softmax_cross_entropy_with_logits(logits = z3, labels = y)
	cost = tf.reduce_mean(cost)
	return cost


def model(trainx, trainy, testx, testy, alpha, iterations = 100, minibatchsize = 64):
	(m,nh0,nw0,nc0) = trainx.shape
	ny = ytrain.shape[1]
	costs = []
	x,y = create_placeholders(nh0,nw0,nc0,ny)
	params = init_params()
	z3 = forward_prop(x, params)
	cost = compute_cost(z3,y)
	optimizer = tf.train.AdamOptimizer().minimize(cost)
	init = tf.global_variables_initializer()
	with tf.Session as sess:
		sess.run(init)
		for epoch in range(iterations):
			minibatch_cost = 0
			nums = m // minibatchsize
			minibatches = random_mini_batches(trainx,trainy, minibatchsize, seed = 1)
			for minibatch in minibatches:
				minibatchx, minibatchy = minibatch
				_, tempcost = sess.run([optimizer, cost], feed_dict = {x:minibatchx, y:minibatchy})
				minibatchcost += tempcost/ nums
			if(i%5 == 0):
				print(epoch, minibatchcost)
				costs.append(minibatchcost)
		plt.plot(costs)
		plt.show()
		predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: trainx, Y: trainy})
        test_accuracy = accuracy.eval({X: testx, Y: testy})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
                
        return train_accuracy, test_accuracy, params


_,_,params = model(trainx, trainy, testx, testy)
