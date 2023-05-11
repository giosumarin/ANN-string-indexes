import numpy as np
import string, random
import gc, os
from itertools import chain
from pathlib import Path
import ray.cloudpickle as pickle
import click
import ast
import sys


class PythonLiteralOption(click.Option):

    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except:
            raise click.BadParameter(value)


def LCP(a, b):
	lcp = 0
	for j in range(min(len(a),len(b))):
		if a[j] == b[j]:
			lcp+=1
		else:
			break	
	return lcp

# function to compute the longest common prefix of strings in the input list
def LCP_list(mylist):
	lcp = 0
	for i in range(len(mylist)-1):
		a = mylist[i]
		b = mylist[i+1]
		temp_lcp = LCP(a,b)
		if temp_lcp > lcp:
			lcp = temp_lcp
		
	return lcp	


# Onehot encoding: each string is padded with "." and then tranformed to a binary string obtained by concatenating the one hot encode of its characters
def encode_dataset(a, maxstrlength, nstrings):
	# defining labels
	Y = np.arange(1, nstrings+1)/nstrings
	# taking unique characters
	characters = ''.join(sorted(set(chain.from_iterable(a))))+'.'
	nchars = len(characters)
	# dictionary of characters: int --> char
	char_to_int = dict(zip([x for x in characters], range(1, nchars)))
	char_to_int['.'] = 0
	X = np.zeros((len(a), maxstrlength, len(characters)), dtype=np.uint8, order='F')
	for i, b in enumerate(a):
		# padding
		word = b + ('.' * (maxstrlength - len(b)))
		# word coding in int
		integer_encoded = [char_to_int[char] for char in word]
		# one hot
		for j, value in enumerate(integer_encoded):
			if value != 0:
				X[i,j,value-1] = np.uint8(1)
			else:
				X[i,j, nchars - 1] = np.uint8(1)
			
	return X, Y, nchars, nchars
	

# Binary encoding: each string is padded with "." and then each caracter is associate with an integer, increasing from 1 (a) to 27 (.).
#  Then each character is encoded with the binary string converting the decimal value associated with the character. 
#   Each string is then  the concatenation of the binary encode of its characters
def encode_dataset_bin(a, maxstrlength, nstrings):

	Y = np.arange(1, nstrings+1)/nstrings
	
	characters = ''.join(set(chain.from_iterable(a)))+'.'
	nchars = len(characters)
	char_to_int = dict(zip([x for x in characters], range(1,nchars+1)))
	char_to_int['.'] = 0
	nbits = int(np.ceil(np.log2(len(characters))))
	X = np.zeros((len(a), maxstrlength, nbits), dtype=np.uint8, order='F')

	for i, b in enumerate(a):
		word = b + ('.' * (maxstrlength - len(b)))
		integer_encoded = [char_to_int[char] for char in word]
		for j, value in enumerate(integer_encoded):			
			if value == 0:
				value = nchars
			binstr = bin(value)[2:].zfill(nbits)			
			for k, bitval in enumerate(binstr):
				if bitval == '1':
					X[i,j,k] = 1
	return X, Y, nbits, nchars

# function to replicate strings in positions "ids" in the original string list b, "times" times, and to append it to the list b.
#   the numpy array X is then created to ffed the NN.	
def enrich_toperrorItems(ids, times, b, Y, maxstrlength, codifica):
	toadd = [b[p] for p in ids]
	toaddY = [Y[p] for p in ids]	
	# replicating the top kk maximum errors items "times" times 
	b = b + toadd*times
	Y = Y.tolist()
	Y = Y + toaddY*times
	del toadd, toaddY	
	Y = np.asarray(Y)
	if codifica=="ohe":
		X, _, _ ,_= encode_dataset(b, maxstrlength, len(b))		

	elif codifica.lower() == "bin":
		X, _, _, _ = encode_dataset_bin(b, maxstrlength, len(b))
	
	return X, Y	

# Function to load strings for file 'file' and encoding then in the format enc
def load_data(enc, file, verbose=1):
	# file containing the dataset
	filename= Path(file)
	if(verbose):
		print("\n...................\nFile to be read:"+str(filename))
	b = []
	with open(filename, "r") as inF:
		for line in inF:
			b.append(line.strip())
	len_b = len(b)
	b = sorted(b)
	
	# removing file extension
	fileout = str(filename.with_suffix(''))+"_"+enc+".pkl"

	if os.path.isfile(fileout):
		if(verbose):
			print("File present, load pkl...")		
		
		with open(fileout, 'rb') as f:
			data = pickle.load(f)
		
		max_lcp = data['maxLCP']	
		X_t = data['X']
		Y_t= data['Y']
		char_size = data['asize']
		nchars = data['nchars']	
		maxstrlength = data['maxstrlen']
		if(verbose):
			print("Imported", len_b, "strings. max length:", 
					maxstrlength, ", nchars:", nchars)	
		del data		
	else: 
		if(verbose):
			print("Encoding strings...")	

		maxstrlength = len(b[0])
		for parola in b:
			if len(parola) > maxstrlength:
				maxstrlength = len(parola)
		if(verbose):
			print("Encoded", len_b, "strings. Max length:", maxstrlength)
		
		#truncating at LCP + 1
		max_lcp = LCP_list(b)
		if max_lcp < maxstrlength - 1:
			print("\tMax LCP:", max_lcp, "\t truncating strings...\n")
			b[:] = (elem[:(max_lcp+1)] for elem in b)	
			maxstrlength = max_lcp+1
		
		if enc.lower() == "bin":
			X_t, Y_t, char_size, nchars = encode_dataset_bin(b, maxstrlength, len_b)
		elif enc.lower() == "ohe":
			X_t, Y_t, char_size, nchars = encode_dataset(b, maxstrlength, len_b)
		else:
			print("Encoding string not recognised: use 'bin' or 'ohe'")
			sys.exit()

		data = {'X': X_t, 'Y': Y_t, 'asize': char_size, 
			'maxstrlen': maxstrlength, 
			'nchars': nchars, 
			'maxLCP': max_lcp}
			
		with open(fileout, 'wb') as f:
			pickle.dump(data, f)			
			
	return b, X_t, Y_t, char_size, maxstrlength, nchars, max_lcp

