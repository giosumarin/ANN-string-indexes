
#!/usr/bin/env python3
# coding: utf-8


import sys, ast
import random
from numpy.random import seed 
import numpy as np
import os, gc
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from utils import *
from models import *
import click


import timeit

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

random_seed = 56
seed(random_seed)
tf.random.set_seed(random_seed)
random.seed(random_seed)

@click.command()
@click.option("--codifica", default="ohe", help="Encoding for data")
@click.option("--file", default="data/az-words_truncated20.txt", help="Dataset file path")
@click.option('--batchsize', default=75, help='Batch size')
@click.option('--modelstr', default="SMLP", help='Model to train')
@click.option('--loss', default="mae", help='Loss function')
@click.option('--nhidden', cls=PythonLiteralOption, default=[])
@click.option('--enrich', is_flag=True, default=False, help='True to replicate maximum errors strings, False to avoid enrichment')
@click.option('--step', default=0, help='argument d of SMLP model')
@click.option('--penr', default=0.0, help='fraction of worst strings to be enriched')
@click.option('--times', default=0, help='number of replicates for top-error strings')

def main(codifica, file, batchsize, modelstr, loss, nhidden, enrich, step, penr, times):
	if not os.path.exists("pretrained_models"):
		os.makedirs("pretrained_models")
		
	if modelstr == "SMLP":
		bestModel = "pretrained_models/"+file.split('/')[-1]+"."+codifica+"."+modelstr+".step"+str(step)+".bs"+str(batchsize)+"nh"+str(nhidden)+".enr"+str(penr)+"_stepvar_"+str(0)+".times."+str(times)+".h5"
	else:
		bestModel = "pretrained_models/"+file.split('/')[-1]+"."+codifica+"."+modelstr+".bs"+str(batchsize)+"nh"+str(nhidden)+".enr"+str(penr)+".times."+str(times)+".h5"

		
	b, X, Y, char_size, maxstrlength, _, _ = load_data(codifica, file)
	gc.collect()
	len_b = len(b)


	# build the model
	if modelstr == "SMLP":
		model, X, _ = create_SMLP(maxstrlength, char_size, nhidden, step, X)
	else:
		model, _ = create_model(modelstr, maxstrlength, char_size, nhidden)	


	model.load_weights(bestModel)

	start = timeit.default_timer()
	y_pred = model.predict(X, batch_size=5000, verbose = 0).reshape(-1,)
	stop = timeit.default_timer()	
	diffs = np.abs(Y - y_pred)

	del X

	if modelstr == "SMLP":
		modelstr = file.split('/')[-1]+"."+modelstr+"."+str(nhidden)+".step."+str(step)
	else:
		modelstr = file.split('/')[-1]+"."+modelstr+"."+str(nhidden)	
	modelstr = modelstr+".enr."+str(penr)+"_times"+str(times)

	space_dense= model.count_params()*32
	
	if not os.path.exists("results_test"):
		os.makedirs("results_test")
		
	file1 = open('results_test/Results.txt', 'a')

	file1.write(modelstr+"\t"+codifica+"\t"+str(batchsize)+ "\t"+str("{0:.0e}".format(len_b))+"\t"+str(round(maxstrlength, 3))+ "\t"+str("{0:.2e}".format(np.ceil(np.max(diffs)*len_b)))+"\t"+str("{0:.2e}".format(np.round(np.sum(diffs), 1)))+"\t"+str(round(space_dense/len_b, 3))+"\t"+str("{0:.2e}".format((stop-start)/len_b))+"\n")
	file1.close()

if __name__ == '__main__':
    main()
