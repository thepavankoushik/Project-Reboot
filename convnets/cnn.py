import numpy as np
import h5py
import matplotlib.pyplot as plt

def zeropad(x,pad):
	s = np.pad(x,((0,0),(pad, pad),(pad, pad),(0,0)),"constant",constant_values = (0,0))
	return s

def conv_step(aslice,w,b):
	s = np.multiply(aslice,w)
	z = np.sum(s)
	z = np.add(z,b)
	return z

def conv_forward(aprev, w,b, hparams):
	(m,nhprev,nwprev,ncprev) = aprev.shape
	(f,f,ncprev,nc) = w.shape
	stride = hparams["stride"]
	pad = hparams["pad"]
	nh = int(((nhprev-f+2*pad)/stride)+1)
	nw = int(((nwprev-f+2*pad)/stride)+1)
	z = np.zeros((m, nh, nw, nc))
	aprev = zeropad(aprev, pad)
	for i in range(m):
		aprevd = aprev[i]
		for h in range(nh):
			for w in range(nw):
				for c in range(nc):
					vs = stride*h
					ve = vs+f
					hs = stride*w
					he = hs+f
					aslice = aprevd[vs:ve, hs:he,:]
					z[i,h,w,c] = conv_step(aslice,w[:,:,:,c],b[:,:,:,c])
	cache = (aprev,w,b,hparams)
	return z, cache


def pool_forward(aprev, hparams, mode = "max"):
	(m,nhprev,nwprev,ncprev) = aprev.shape
	(f,f,ncprev,nc) = w.shape
	stride = hparams["stride"]
	pad = hparams["pad"]
	nh = int(((nhprev-f)/stride)+1)
	nw = int(((nwprev-f)/stride)+1)
	a = np.zeros((m, nh, nw, nc))
	aprev = zeropad(aprev, pad)
	for i in range(m):
		aprevd = aprev[i]
		for h in range(nh):
			for w in range(nw):
				for c in range(nc):
					vs = stride*h
					ve = vs+f
					hs = stride*w
					he = hs+f
					aslice = aprevd[vs:ve, hs:he,c]
					if mode == "max":
						a[i,h,w,c] = np.max(aslice)
					else:
						a[i,h,w,c] = np.average(aslice)
	cache = (aprev, hparams)
	return a,cache
