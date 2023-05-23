# ANN-string-indexes

To execute the `run_trainALL.sh` and `run_testALL.sh` executable, run the following commands (which require Python 3) in your terminal (Linux distriburion):

```
python -m venv /path/to/new/virtual/environment
source /path/to/new/virtual/environment/bin/activate
pip install numpy==1.23.3 ray==2.0.0 tensorflow==2.8.0 click==8.0.3
```

In the `pretrained_models` folder there are the pretrained models corresponding to the results shown in Figure 5. To reproduce those results it is necessary to run the bash script `run_testALL.sh` , the results will be written in `results_test/Results.txt`.

If you want to retrain the models you have to run the `run_trainALL.sh` script, the models will be saved in the model folder and the results will be written in `results/Results.txt`.
