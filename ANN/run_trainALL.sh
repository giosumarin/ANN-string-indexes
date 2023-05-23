#!/bin/bash

#IMPOSTARE EPOCHE A 300

#!/bin/bash

##AZ
python3 train.py --codifica "ohe" --file "data/az-words_truncated20.txt" --batchsize 75 --epochs 300 --patience 5 --modelstr SMLP --loss "mae" --nhidden \[50,50\] --step 10 
python3 train.py --codifica "ohe" --file "data/az-words_truncated20.txt" --batchsize 75 --epochs 300 --patience 5 --modelstr SMLP --loss "mae" --nhidden \[50,50\] --enrich --step 10 --penr 0.0005 --times 20
python3 train.py --codifica "ohe" --file "data/az-words_truncated20.txt" --batchsize 75 --epochs 300 --patience 5 --modelstr SMLP --loss "mae" --nhidden \[50,50\] --step 5 
python3 train.py --codifica "ohe" --file "data/az-words_truncated20.txt" --batchsize 75 --epochs 300 --patience 5 --modelstr SMLP --loss "mae" --nhidden \[50,50\] --enrich --step 5 --penr 0.0005 --times 20
python3 train.py --codifica "ohe" --file "data/az-words_truncated20.txt" --batchsize 75 --epochs 300 --patience 5 --modelstr SMLP --loss "mae" --nhidden \[100,50\] --step 10 
python3 train.py --codifica "ohe" --file "data/az-words_truncated20.txt" --batchsize 75 --epochs 300 --patience 5 --modelstr SMLP --loss "mae" --nhidden \[100,50\] --enrich --step 10 --penr 0.0005 --times 20
python3 train.py --codifica "ohe" --file "data/az-words_truncated20.txt" --batchsize 75 --epochs 300 --patience 5 --modelstr LSTM --loss "mae" --nhidden \[20,20,50\] --enrich --penr 0.0005 --times 20
python3 train.py --codifica "ohe" --file "data/az-words_truncated20.txt" --batchsize 75 --epochs 300 --patience 5 --modelstr biLSTM --loss "mae" --nhidden \[20,20,50\] --enrich --penr 0.0005 --times 20
python3 train.py --codifica "ohe" --file "data/az-words_truncated20.txt" --batchsize 75 --epochs 300 --patience 5 --modelstr SMLP --loss "mae" --nhidden \[30,50\] --enrich --step 3 --penr 0.0005 --times 20
python3 train.py --codifica "ohe" --file "data/az-words_truncated20.txt" --batchsize 75 --epochs 300 --patience 5 --modelstr SMLP --loss "mae" --nhidden \[20,50\] --enrich --step 2 --penr 0.0005 --times 20
python3 train.py --codifica "ohe" --file "data/az-words_truncated20.txt" --batchsize 75 --epochs 300 --patience 5 --modelstr LSTM_multi --loss "mae" --nhidden \[20,20,50\] --enrich --penr 0.0005 --times 20
python3 train.py --codifica "ohe" --file "data/az-words_truncated20.txt" --batchsize 75 --epochs 300 --patience 5 --modelstr CNN --loss "mae" --nhidden \[50,50\] --enrich --penr 0.0005 --times 20
python3 train.py --codifica "ohe" --file "data/az-words_truncated20.txt" --batchsize 75 --epochs 300 --patience 5 --modelstr SMLP --loss "mae" --nhidden \[20,50\] --enrich --penr 0.0005 --times 20
python3 train.py --codifica "ohe" --file "data/az-words_truncated20.txt" --batchsize 75 --epochs 200 --patience 5 --modelstr MLP --loss "mae" --nhidden \[50,50\] --enrich --penr 0.0005 --times 20
python3 train.py --codifica "bin" --file "data/az-words_truncated20.txt" --batchsize 75 --epochs 300 --patience 5 --modelstr "SMLP" --loss "mae" --nhidden \[20,50\] --step 2 --verbose

##UK
python3 train.py --codifica "ohe" --file "data/uk-2002_truncated20.txt" --batchsize 150 --epochs 300 --patience 5 --modelstr "SMLP" --loss "mae" --nhidden \[30,50\] --enrich --step 3 --penr 0.0005 --times 20
python3 train.py --codifica "ohe" --file "data/uk-2002_truncated20.txt" --batchsize 150 --epochs 300 --patience 5 --modelstr "SMLP" --loss "mae" --nhidden \[20,50\] --enrich --step 2 --penr 0.0005 --times 20
python3 train.py --codifica "ohe" --file "data/uk-2002_truncated20.txt" --batchsize 150 --epochs 300 --patience 5 --modelstr "SMLP" --loss "mae" --nhidden \[50,50\] --enrich --step 10 --penr 0.0005 --times 20
python3 train.py --codifica "ohe" --file "data/uk-2002_truncated20.txt" --batchsize 150 --epochs 300 --patience 5 --modelstr "SMLP" --loss "mae" --nhidden \[50,50\] --enrich --step 5 --penr 0.0005 --times 20
python3 train.py --codifica "ohe" --file "data/uk-2002_truncated20.txt" --batchsize 150 --epochs 300 --patience 5 --modelstr "SMLP" --loss "mae" --nhidden \[100,50\] --enrich --step 10 --penr 0.0005 --times 20
python3 train.py --codifica "ohe" --file "data/uk-2002_truncated20.txt" --batchsize 150 --epochs 100 --patience 5 --modelstr "LSTM" --loss "mae" --nhidden \[20,20,50\] --enrich --penr 0.0005 --times 20
python3 train.py --codifica "ohe" --file "data/uk-2002_truncated20.txt" --batchsize 150 --epochs 100 --patience 5 --modelstr "biLSTM" mae --nhidden \[20,20,50\] --enrich --penr 0.0005 --times 20
python3 train.py --codifica "bin" --file "data/uk-2002_truncated20.txt" --batchsize 150 --epochs 300 --patience 5 --modelstr "SMLP" --loss "mae" --nhidden \[20,50\] --step 2 
python3 train.py --codifcia "ohe" --file "data/uk-2002_truncated20.txt" --batchsize 150 --epochs 200 --patience 5 --modelstr "MLP" --loss "mae" --nhidden \[50,50\] --enrich --penr 0.0005 --times 20

# DNA
python3 train.py --codifica "ohe" --file "data/dna-k-mer.txt" --batchsize 300 --epochs 300 --patience 5 --modelstr "SMLP" --loss "mae" --nhidden \[30,50\] --enrich --step 3 --penr 0.0005 --times 20
python3 train.py --codifica "ohe" --file "data/dna-k-mer.txt" --batchsize 300 --epochs 300 --patience 5 --modelstr "SMLP" --loss "mae" --nhidden \[20,50\] --enrich --step 2 --penr 0.0005 --times 20
python3 train.py --codifica "ohe" --file "data/dna-k-mer.txt" --batchsize 300 --epochs 300 --patience 5 --modelstr "SMLP" --loss "mae" --nhidden \[50,50\] --enrich --step 5 --penr 0.0005 --times 20
python3 train.py --codifica "ohe" --file "data/dna-k-mer.txt" --batchsize 300 --epochs 300 --patience 5 --modelstr "SMLP" --loss "mae" --nhidden \[100,50\] --enrich --step 10 --penr 0.0005 --times 20
python3 train.py --codifica "ohe" --file "data/dna-k-mer.txt" --batchsize 300 --epochs 100 --patience 5 --modelstr "LSTM" --loss "mae" --nhidden \[20,20,50\] --enrich --penr 0.0005 --times 20
python3 train.py --codifica "ohe" --file "data/dna-k-mer.txt" --batchsize 300 --epochs 100 --patience 5 --modelstr "biLSTM" --loss "mae" --nhidden \[20,20,50\] --enrich --penr 0.0005 --times 20
python3 train.py --codifica "ohe" --file "data/dna-k-mer.txt" --batchsize 300 --epochs 300 --patience 5 --modelstr "SMLP" --loss "mae" --nhidden \[100,50\] --enrich --step 10 --penr 0.0005 --times 15
python3 train.py --codifica "bin" --file "data/dna-k-mer.txt" --batchsize 300 --epochs 200 --patience 5 --modelstr "SMLP" --loss "mae" --nhidden \[20,50\] --step 2
python3 train.py --codifica "ohe" --file "data/dna-k-mer.txt" --batchsize 300 --epochs 200 --patience 5 --modelstr "MLP" --loss "mae" --nhidden \[50,50\] --enrich --penr 0.0005 --tmes 20

## GEO
python3 train.py --codifica "ohe" --file "data/GeoNames_truncated20.txt" --batchsize 150 --epochs 200 --patience 5 --modelstr "SMLP" --loss "mae" --nhidden \[50,50\] --enrich --step 10 --penr 0.0005 --times 20
python3 train.py --codifica "ohe" --file "data/GeoNames_truncated20.txt" --batchsize 150 --epochs 200 --patience 5 --modelstr "SMLP" --loss "mae" --nhidden \[50,50\] --enrich --step 5 --penr 0.0005 --times 20
python3 train.py --codifica "ohe" --file "data/GeoNames_truncated20.txt" --batchsize 150 --epochs 200 --patience 5 --modelstr "SMLP" --loss "mae" --nhidden \[100,50\] --enrich --step 10 --penr 0.0005 --times 20
python3 train.py --codifica "ohe" --file "data/GeoNames_truncated20.txt" --batchsize 150 --epochs 50 --patience 3 --modelstr "LSTM" --loss "mae" --nhidden \[20,20,50\] --enrich --penr 0.0005 --times 20
python3 train.py --codifica "ohe" --file "data/GeoNames_truncated20.txt" --batchsize 150 --epochs 50 --patience 3 --modelstr "biLSTM" --loss "mae" --nhidden \[20,20,50\] --enrich --penr 0.0005 --times 20
python3 train.py --codifica "ohe" --file "data/GeoNames_truncated20.txt" --batchsize 150 --epochs 200 --patience 5 --modelstr "SMLP" --loss "mae" --nhidden \[30,50\] --enrich --step 3 --penr 0.0005 --times 20
python3 train.py --codifica "ohe" --file "data/GeoNames_truncated20.txt" --batchsize 150 --epochs 200 --patience 5 --modelstr "SMLP" --loss "mae" --nhidden \[20,50\] --enrich --step 2 --penr 0.0005 --times 20
python3 train.py --codifica "bin" --file "data/GeoNames_truncated20.txt" --batchsize 150 --epochs 200 --patience 5 --modelstr "SMLP" --loss "mae" --nhidden \[20,50\] --step 2
python3 train.py --codifica "ohe" --file "data/GeoNames_truncated20.txt" --batchsize 150 --epochs 100 --patience 3 --modelstr "MLP" --loss "mae" --nhidden \[50,50\] --enrich --penr 0.0005 --times 20


