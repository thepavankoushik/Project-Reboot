import numpy  as np
from utils import *
import random

data = open('data.txt','r').read()
data = data.lower()
chars = list(set(data))
vocab_size = len(chars)
data_size = len(data)

char_to_index = {ch:i for i,ch in enumerate(sorted(chars))}
index_to_char = {i:ch for i,ch in enumerate(sorted(chars))}

def clip(gradients, maxvalue):
	for g in gradients:
		np.clip(g, -maxvalue, maxvalue, out= g)
	return gradients


def sample(params, char_to_index, seed):
	waa, wax, wya, by, ba = params
	vocab_size = by.shape[0]
	na = waa.shape[1]

	x = np.zeros((vocab_size,1))
	aprev = np.zeros((na, 1))
	indices = []
	idx = -1
	counter = 0
	new = char_to_index["\n"]
	while(idx != new and counter!=50):
		a = np.tanh(np.dot(waa, aprev)+np.dot(wax,x)+ba)
		z = np.dot(wya, a)+by
		y = softmax(z)
		idx = np.random.choice(list(range(vocab_size)), p = y.ravel())
		indices.append(idx)
		x = np.zeros((vocab_size,1))
		x[idx] = 1
		aprev = a
		seed += 1
		counter += 1
	indices.append(char_to_index["\n"])
	return indices




#train rnn model on data for getting proper params
def optimize(x,y,aprev,params, alpha=0.01):
	loss, cache = rnn_forward(x,y,aprev,params)
	grads, a = rnn_backward(x,y,params,cache)
	grads = clip(grads,5)
	params = update_params(params, grads, alpha)
	return loss, grads, a[-1]


def model(data, index_to_char, char_to_index, iterations = 35000, na = 50, dinonames = 7, vocab_size = 27):
	nx, ny = vocab_size
	params = initialize_params(na, nx, ny)
	loss = get_initial_loss(vocab_size, dinonames)
	with open("data.txt") as f:
		examples = f.readlines()
	examples = [x.lower().strip() for x in examples]
	np.random.shuffle(examples)
	aprev = np.zeros((na, 1))
	for j in range(iterations):
		index = j%len(examples)
		x = [char_to_index[i] for i in examples]
		y = [1:]+[char_to_index["\n"]]
		curr_loss, grads, aprev = optimize(x,y,aprev, params)
		loss = smooth(loss, curr_loss)
		if j%2000 == 0:
			for name in range(dinonames):
				indices = sample(params, char_to_index, seed)
				print(indices)
				#prints indices instead of continous chars
		return params

params = model(data, index_to_char, char_to_index)

