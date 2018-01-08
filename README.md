# SCT
- code for model to predict story development
## required 

- tensorflow 1.3.0 

- python3.6 
## qucik start
This model need at least 6GB of GPU RAM. if you meet these conditions you can use flowing cmd to qucik start

python cli.py

if you don't have a gpu, you can run model by flow

python cli.py --device_type cpu

if your gpu RAM < 6GB

you can use --batch_size to set batch set <100

## note
- cli.py contain Model and program configuration files.

- model3.py is our best model

- model.py is our sentences concateation moel mentail in our paper

- data in this file we cantain the test data file we had dealwithed


