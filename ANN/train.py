
#!/usr/bin/env python3
# coding: utf-8

#FARE REPOSITORY
import random
from numpy.random import seed 
import numpy as np
import os, gc
import tensorflow as tf
from utils import *
from models import *
import click

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

random_seed = 123
seed(random_seed)
tf.random.set_seed(random_seed)
random.seed(random_seed)



@click.command()
@click.option("--codifica", default="ohe", help="Encoding for data")
@click.option("--file", default="data/az-words_truncated20.txt", help="Dataset file path")
@click.option('--batchsize', default=75, help='Batch size')
@click.option('--epochs', default=300, help='Number of epochs')
@click.option('--patience', default=5, help='Patience for EarlyStopping')
@click.option('--modelstr', default="SMLP", help='Model to train')
@click.option('--loss', default="mae", help='Loss function')
@click.option('--nhidden', cls=PythonLiteralOption, default=[])
@click.option('--enrich', is_flag=True, default=False, help='True to replicate maximum errors strings, False to avoid enrichment')
@click.option('--step', default=0, help='argument d of SMLP model')
@click.option('--penr', default=0.0, help='fraction of worst strings to be enriched')
@click.option('--times', default=0, help='number of replicates for top-error strings')
@click.option('--verbose', is_flag=True, default=True, help='True to verbose, False to avoid info-print')


def main(codifica, file, batchsize, epochs, patience, modelstr, loss, nhidden, enrich, step, penr, times, verbose):

	if not os.path.exists("models"):
		os.makedirs("models")

	if modelstr == "SMLP":
		bestModel = "models/"+file.split('/')[-1]+"."+codifica+"."+modelstr+".step"+str(step)+".bs"+str(batchsize)+"nh"+str(nhidden)+".enr"+str(penr)+"_stepvar_"+str(0)+".times."+str(times)+".h5"
	else:
		bestModel = "models/"+file.split('/')[-1]+"."+codifica+"."+modelstr+".bs"+str(batchsize)+"nh"+str(nhidden)+".enr"+str(penr)+".times."+str(times)+".h5"

	b, X, Y, char_size, maxstrlength, _, _ = load_data(codifica, file)
	gc.collect()
	len_b = len(b)

	kk = int(np.ceil(penr*len_b)) # kk is the  fraction of strings to be enriched
	Y = np.asarray(Y)

	lr_rates_refined = [5e-04, 1e-4, 5e-05, 1e-5, 5e-06]



	if modelstr == "SMLP":
		model, X, cat_names = create_SMLP(maxstrlength, char_size, nhidden, step, X)
	else:
		model, cat_names = create_model(modelstr, maxstrlength, char_size, nhidden)

	if enrich:
		diffs = train_enrich(model, X, Y, epochs, lr_rates_refined, bestModel, patience, loss, batchsize, codifica, times, maxstrlength, file, len_b, b, cat_names, kk, modelstr, verbose)
	else:
		diffs = train(model, X, Y, epochs, lr_rates_refined, bestModel, patience, loss, batchsize, len_b, verbose)

	del X

	if modelstr == "SMLP":
		modelstr = file.split('/')[-1]+"."+modelstr+"."+str(nhidden)+".step."+str(step)
	else:
		modelstr = file.split('/')[-1]+"."+modelstr+"."+str(nhidden)	
	modelstr = modelstr+".enr."+str(penr)+"_times"+str(times)

	space_dense= model.count_params()*32
	if not os.path.exists("results"):
		os.makedirs("results")
		
	file1 = open('results/Results.txt', 'a')

	file1.write(modelstr+"\t"+codifica+"\t"+str(batchsize)+ "\t"+str("{0:.0e}".format(len_b))+"\t"+str(round(maxstrlength, 3))+ "\t"+str("{0:.2e}".format(np.ceil(np.max(diffs)*len_b)))+"\t"+str("{0:.2e}".format(np.round(np.sum(diffs), 1)))+"\t"+str(round(space_dense/len_b, 3))+"\n")
	file1.close()

if __name__ == '__main__':
    main()
