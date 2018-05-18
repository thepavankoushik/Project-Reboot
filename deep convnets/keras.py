import numpy as np
from keras import layers
from keras.layers import Input, Dense,Activation,ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout,GlobalMaxPooling2D,GlobalAveragePooling2D
from keras.model import model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_do
import keras.backend as keras
import matplotlib.pyplot as plt

trainx, trainy , testx, testy = laod_dataset()
trainx/= 255
testx/= 255
def model(input_shape):
	x_input = Input(input_shape)
	x = ZeroPadding2D((3,3))(x_input)
	x = Conv2D(32,(7,7),strides = (1,1), name = "conv0")(x)
	x = BatchNormalization(axis = 3, name = "bn0")(x)
	x = Activation('relu')(x)
	x = MaxPooling2D((2,2), name = "max_pool")(x)
	x = Flatten()(x)
	x = Dense(1,activation = "sigmoid", name = "fc")(x)
	model = Model(inputs = x_input, outputs= x, name = "happymodel")
	return model


m = model(trainx.shape[1:])
m.compile(optimize = 'adam', loss = "binary_crossentropy",metrics = ["accuracy"])
m.fit(x = trainx, y = trainy, epochs = 40, batchsize = 16)
preds = m.evaluate(x = testx, y = testy)
print(preds)
#m.predict(x) also useful
