# SCT
code for model to predict story development
## required 

- tensorflow 1.3.0 

- python3.6 
## qucik start
- Run as following to quick start the model if your running envirement at least GPU RAM 6GB:

python cli.py

- Run as following if no GPU to use:

python cli.py --device_type cpu

- set batch-size <100 if no enough GPU RAM

## note
- cli.py 

contain model and program configuration files

- model.py 

original concatenation model

- model3.py 

our best-result model

- data

prepared test-set data


