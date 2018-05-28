import numpy as np

def softmax(x):
	ex = np.exp(x)
	return ex/ex.sum(axis = 0)

def sigmoid(x):
	return (1/(1+np.exp(-x)))

def rnn_cell_forward(xt, aprev, params):
	wax, waa, wya, ba, by = params
	anext = np.tanh(np.dot(wax,xt)+np.dot(waa, aprev)+ba)
	ytpred = softmax(np.dot(wya,anext)+by)
	cache = (anext, ytpred, xt, params)
	return anext, ytpred, cache


def rnn_forward(x,a0, params):
	wax, waa, wya, ba, by = params
	caches = []
	nx, m, tx = x.shape
	ny, na = wya.shape
	a = np.zeros((na, m, tx))
	y = np.zeros((ny, m, tx))
	anext = a0
	for t in range(tx):
		anext, ytpred, cache = rnn_cell_forward(x[:,:,t], anext, params)
		a[:,:,t] = anext
		y[:,:,t] = ytpred
		caches.append(cache)
	return a,y,caches


def lstm_cell_forward(xt, aprev, cprev, params):
	wf, wu, wo, wy,wc,bc, bf, bu, bo, by = params
	nx, m = x.shape
	ny, na = wy.shape
	concat = np.zeros((nx+na,m))
	concat[:na,:] = aprev
	concat[na:,:] = xt
	ugate = sigmoid(np.dot(wu,concat)+bu)
	fgate = sigmoid(np.dot(wf,concat)+bf)
	ogate = sigmoid(np.dot(wo,concat)+bo)
	cdash = np.tanh(np.dot(wc,concat)+bc)
	ct = ugate*cdash + fgate*cprev
	at = ot*np.tanh(ct)
	yt = softmax(np.dot(wy,at)+by)
	cache = (at,ct, yt, params)
	return at, ct, yt, cache


def lstm_forward(x,a0,params):
	caches = []
	nx,m,tx = x.shape
	ny, na = wy.shape
	a = np.zeros((na, m, tx))
	c = np.zeros((na, m, tx))
	y = np.zeros((ny, m, tx))
	anext = a0
	cnext = np.zeros((na, m))
	for t in range(tx):
		anext, cnext, yt, cache = lstm_cell_forward(x[:,:,t], anext, cnext, params)
		a[:,:,t] = anext
		c[:,:,t] = cnext
		y[:,:,t] = yt
		caches.append(cache)
	return a,y,c,caches
