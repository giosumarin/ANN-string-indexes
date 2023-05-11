import tensorflow as tf
import gc
from tensorflow.keras.models import Sequential,Model
#from tensorflow.keras.layers import Dense, Bidirectional, LSTM, Embedding, BatchNormalization, Flatten, Conv1D, Input, MaxPooling1D, Concatenate, Flatten
from tensorflow.keras.layers import *
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras import layers
import numpy as np
#from tensorflow.keras.layers import LSTM, Flatten, Conv1D, LocallyConnected1D, CuDNNLSTM, CuDNNGRU, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from utils import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def custom_sigmoid(x, coeff=1): # 1/(1+e^-(coeff*x))
    return (tf.keras.activations.sigmoid(coeff*x))

def LSTM_CNN(maxstrlength, alphabet_size, h):
	model = Sequential()
	print(h)
	ini = Input(shape=(maxstrlength,alphabet_size))
	L= LSTM(h[0], stateful=False, return_sequences=True)(ini)
	L= LSTM(h[1], stateful=False, return_sequences=False)(L)	
	C = Conv1D(20, kernel_size=4, activation='relu', padding='same')(ini)
	C = Conv1D(10, kernel_size=2, activation='relu', padding='valid')(C)
	C = Conv1D(2, kernel_size=3, activation='relu', strides=2, padding='same')(C) 
	C = Flatten()(C)
	prev = Concatenate()([L, C])
	for val in np.arange(2, len(h)):
		prev = Dense(h[val], activation='relu')(prev)
	out = Dense(1, activation='sigmoid')(prev)
	model = Model(inputs = ini, outputs = out)
	return model

# "Step" Network,  where characters input are stacked atop each other. The output of the last character layer is input for the layer of the second last character, along with the input of the second last char, and so on.
# cat_sise: list with the size of each input. In our exp it is the same for all categories, since each character is coded with the same length. But the code accept even more general cases
# cat_names: list with the names each category is to be assigned to.
# neurons_hidden1_list: list with the hidden neurons in the dense layer associated to each input category. 
# nhidden:  list contaning the number of hidden neurons in each layer atop the last concatenate layer

def SMLP(cat_size, cat_names, neurons_hidden1_list, nhidden, trainable_list=None):
	if trainable_list is None:
		trainable_list = np.repeat(True, len(cat_names)).tolist()
	
	num_inp_cat = len(cat_size)
	inputs= []
	ini = Input(shape = (cat_size[num_inp_cat - 1], ), name= cat_names[num_inp_cat - 1])
	prev_hidden_layer = Dense(neurons_hidden1_list[num_inp_cat - 1], activation="relu",
					name="D"+str(num_inp_cat), 
					trainable=trainable_list[num_inp_cat - 1])(ini)
	inputs.append(ini)
	
	for i in reversed(range(num_inp_cat-1)):
		input_cat = Input(shape = (cat_size[i], ), name= cat_names[i])
		prev_hidden_layer = Dense(neurons_hidden1_list[i], activation="relu",
			name="D"+str(i),
			trainable=trainable_list[i])( Concatenate()([input_cat,prev_hidden_layer]) ) 
	  
		inputs.append(input_cat)	
		
	if len(nhidden)>0 :		
		for i in range(len(nhidden)):
			h = nhidden[i]
			h2  = Dense(h, activation = "relu")(prev_hidden_layer)
			prev_hidden_layer = h2	
		
		output = Dense(1, activation = custom_sigmoid)(prev_hidden_layer)	
	else:
		output = Dense(1, activation = custom_sigmoid)(prev_hidden_layer)
	
	mod = Model(inputs = inputs, outputs = output)
	return mod
	

def FC(maxstrlength, alphabet_size, nhidden):
	model = Sequential()
	init = tf.keras.initializers.lecun_uniform(seed=0)
	model.add(Input(shape=(maxstrlength,alphabet_size,)))
	model.add(Flatten())

	for nh in nhidden:
		model.add(Dense(nh, activation='relu', kernel_initializer=init))
	
	model.add(Dense(1, activation='sigmoid'))
	return model
		

def CNN_flat(maxstrlength, alphabet_size, nhidden):
        model = Sequential()
        init = tf.keras.initializers.lecun_uniform(seed=0)
        model.add(Input(shape=(maxstrlength, alphabet_size)))
        model.add(tf.keras.layers.Reshape((maxstrlength*alphabet_size,1), input_shape=(maxstrlength, alphabet_size)))
        model.add(Conv1D(32, kernel_size=3, strides=2, activation='relu', padding='valid'))
        model.add(Conv1D(16, kernel_size=3, strides=2, activation='relu', padding='same'))
        model.add(Conv1D(8, kernel_size=3, activation='relu', padding='valid'))
        model.add(GlobalMaxPooling1D(data_format="channels_first"))

        for nh in nhidden:
                model.add(Dense(nh, activation='relu', kernel_initializer=init))

        model.add(Dense(1, activation='sigmoid'))
        return model

def CNN(maxstrlength, alphabet_size, nhidden):
	model = Sequential()
	init = tf.keras.initializers.lecun_uniform(seed=0)
	model.add(Input(shape=(maxstrlength, alphabet_size)))
	model.add(Conv1D(27, kernel_size=4, activation='relu', padding='same'))
	model.add(Conv1D(10, kernel_size=2, activation='relu', padding='valid'))
	model.add(Conv1D(2, kernel_size=2, activation='relu', padding='valid'))
	model.add(Flatten())
	for nh in nhidden:
		model.add(Dense(nh, activation='relu', kernel_initializer=init))
        
	model.add(Dense(1, activation='sigmoid'))
	return model


def biLSTMcud(maxstrlength, alphabet_size, h):
	model = Sequential()
	model.add(Input(shape=(maxstrlength,alphabet_size)))
	model.add(Bidirectional(CuDNNLSTM(h[0], stateful=False, return_sequences=True)))
	model.add(Bidirectional(CuDNNLSTM(h[1], stateful=False, return_sequences=False)))
	for val in np.arange(2, len(h)):
		model.add(Dense(h[val], activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	return model

def LSTM_(maxstrlength, alphabet_size, h):
	model = Sequential()
	print(h)
	model.add(Input(shape=(maxstrlength,alphabet_size)))
	model.add(LSTM(h[0], stateful=False, return_sequences=True))
	model.add(LSTM(h[1], stateful=False, return_sequences=False))
	for val in np.arange(2, len(h)):
		model.add(Dense(h[val], activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	return model	
	
def LSTM_multi(maxstrlength, alphabet_size, h):
	model = Sequential()
	model.add(Input(shape=(maxstrlength,alphabet_size)))
	model.add(LSTM(h[0], stateful=False, return_sequences=True))
	model.add(LSTM(h[1], stateful=False, return_sequences=True))
	model.add(Flatten())
	for val in np.arange(2, len(h)):
		model.add(Dense(h[val], activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	return model


def LSTMcud(maxstrlength, alphabet_size, h):

	model = Sequential()
	model.add(Input(shape=(maxstrlength,alphabet_size)))
	model.add(CuDNNLSTM(h[0], stateful=False, return_sequences=True))
	model.add(CuDNNLSTM(h[1], stateful=False, return_sequences=False))
	for val in np.arange(2, len(h)):
		model.add(Dense(h[val], activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	return model  
	
def biLSTM(maxstrlength, alphabet_size, h):
        model = Sequential()
        model.add(Input(shape=(maxstrlength,alphabet_size)))
        model.add(Bidirectional(LSTM(h[0], stateful=False, return_sequences=True)))
        model.add(Bidirectional(LSTM(h[1], stateful=False, return_sequences=False)))
        for val in np.arange(2, len(h)):
                model.add(Dense(h[val], activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        return model

def create_model(modelstr, maxstrlength, char_size, nhidden):
	if modelstr=="CNN":
			model = CNN(maxstrlength, char_size, nhidden[1:])
	elif modelstr == "LSTMcud":
		if tf.test.is_gpu_available():
			model = LSTMcud(maxstrlength, char_size, nhidden)
		else: 
			print("No GPU available. Use LSTM model")
			sys.exit()	
	elif modelstr == "LSTM":
		model = LSTM_(maxstrlength, char_size, nhidden)		
	elif modelstr == "biLSTMcud":
		if tf.test.is_gpu_available():
			model = biLSTMcud(maxstrlength, char_size, nhidden)
		else: 
			print("No GPU available. Use biLSTM model")
			sys.exit()	
	elif modelstr == "biLSTM":
		model = biLSTM(maxstrlength, char_size, nhidden)	
	elif modelstr == "LSTM_multi":
		model = LSTM_multi(maxstrlength, char_size, nhidden)		
	elif modelstr=="MLP":
		model = FC(maxstrlength, char_size, nhidden)	

	return model, None

def create_SMLP(maxstrlength, char_size, nhidden, step, X):
	h = nhidden[0] # parameter b of the SMLP model 
	cat_size = np.repeat(char_size, maxstrlength).tolist()
	cat_names = ['I'+str(i+1) for i in range(maxstrlength)]
	neurons_hidden1_list = [h - step*i if h - step*i > 0 else 1 for i in range(maxstrlength)]
	
	X = {
		cat_names[i]: X[:,i,:] for i in range(maxstrlength)     
	}

	model = SMLP(cat_size, cat_names, neurons_hidden1_list, nhidden[1:])

	del cat_size

	return model, X, cat_names

def train_enrich(model, X, Y, epochs, lr_rates_refined, bestModel, patience, loss, batchsize, codifica, times, maxstrlength, file, len_b, b, cat_names, kk, modelstr, verbose=0):

	early_stopping = EarlyStopping(monitor='loss', mode='min', patience=patience, verbose=0)
	mc = ModelCheckpoint(bestModel,	monitor='loss', mode='min',
							verbose=0,
							save_best_only=True, 
							save_weights_only=True)
	callbacks_list = [early_stopping, mc]

	X_ = X
	Y_ = Y
	del X,Y
	gc.collect()
	lr_ = []
	num_epochs_ = []
	for _, val in enumerate(lr_rates_refined):
		lr_.append(val)
		if val > 5e-5:
			lr_.append(val)
			num_epochs_.append(epochs)	
			num_epochs_.append(epochs)
		else:
			num_epochs_.append(epochs)			
		
	lr_rates_refined = lr_
	del lr_

	IDX = []
	for ss, lr in enumerate(lr_rates_refined):
		gc.collect()
		if verbose:
			print(f"\t ss:{ss}, learning rate corrente:{lr}")
		optim = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2 = 0.999, epsilon=1e-8, decay=0, amsgrad=False)
		model.compile(loss = loss, optimizer=optim)		
		history = model.fit(X_, Y_, epochs=num_epochs_[ss], batch_size=batchsize, verbose=verbose, shuffle=True, callbacks=callbacks_list)
		
		gc.collect()
		model.load_weights(bestModel)
		# save room
		del X_
		gc.collect()
		_, X, Y, _, _, _, _ = load_data(codifica, file, verbose=0)


		if modelstr == "SMLP":
			X = {
				cat_names[i]: X[:,i,:] for i in range(maxstrlength)
			}

		y_pred = model.predict(X, batch_size = 5000, verbose = 0).reshape(-1,)
		del X
		gc.collect()
		errvec = abs(Y - y_pred)
		cumul = np.sum(errvec)
		maxerr = np.max(errvec)*len_b
		if verbose: 
			print("\t\t avg err: {:.0f} - Max err: {:.0f}".format(cumul, maxerr))
		if ss%2 == 0 and lr > 5e-5:
			#idx contains the the indices of the top kk values of errvec
			idx = np.argpartition(errvec, -kk)[-kk:]
			idx = set(idx)
			IDX = set(IDX)
			diff = idx - IDX
			IDX = IDX.union(diff)			
		
			# X and Y will contain the original strings in the first len_b positions
		X_, Y_ = enrich_toperrorItems(IDX, times, b, Y, maxstrlength, codifica)		 
			
		if modelstr == "SMLP":
			X_ = {
				cat_names[i]: X_[:,i,:] for i in range(maxstrlength)     
			}
		
		del errvec, y_pred
			
	del X_, Y_
				
	del b
	gc.collect()
	_, X, Y, _, _, _, _ = load_data(codifica, file, verbose=0)
	if modelstr == "SMLP":
		X = {
			cat_names[i]: X[:,i,:] for i in range(maxstrlength)     
		}

	y_pred = model.predict(X, batch_size=5000, verbose = 0).reshape(-1,)
	diffs = np.abs(Y-y_pred)

	del X

	return diffs



def train(model, X, Y, epochs, lr_rates_refined, bestModel, patience, loss, batchsize, len_b, verbose=False):
	early_stopping = EarlyStopping(monitor='loss', mode='min', patience=patience, verbose=0)
	mc = ModelCheckpoint(bestModel,	monitor='loss', mode='min',
							verbose=0,
							save_best_only=True, 
							save_weights_only=True)
	callbacks_list = [early_stopping, mc]
	for ss, lr in enumerate(lr_rates_refined):
		gc.collect()
		if verbose:
			print(f"\t ss:{ss}, learning rate corrente:{lr}")
		optim = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2 = 0.999, epsilon=1e-8, decay=0, amsgrad=False)
		model.compile(loss = loss, optimizer=optim)	

		history =model.fit(X, Y, epochs=epochs, batch_size=batchsize, verbose=verbose, shuffle=True, callbacks=callbacks_list)
		gc.collect()
		model.load_weights(bestModel)


		if type(model).__name__ == "SMLP":
			X = {
				cat_names[i]: X[:,i,:] for i in range(maxstrlength)
			}

		y_pred = model.predict(X, batch_size = 5000, verbose = 0).reshape(-1,)
		gc.collect()
		errvec = abs(Y - y_pred)
		cumul = np.sum(errvec)
		maxerr = np.max(errvec)*len_b
		if verbose: 
			print("\t\t avg err: {:.0f} - Max err: {:.0f}".format(cumul, maxerr))

		del errvec, y_pred
				
	gc.collect()

	y_pred = model.predict(X, batch_size=5000, verbose = 0).reshape(-1,)
	diffs = np.abs(Y-y_pred)

	return diffs
