# SCT
code for model to predict story development
## required 

 tensorflow 1.3.0 

 python3.6 
 
## qucik start
 Run as following to quick start the model if your running envirement at least GPU RAM 6GB:

     python start.py

 Run as following if no GPU to use:

     python start.py --device_type cpu

 set batch-size <100 if no enough GPU RAM

## note

    start.py 

contain parameters in the experiment  
     
    model_c.py 

contain model and program configuration files

    tag.csv 

contain the tagged data information described in paper

    prepro_sentences.py

Data preprocessing code

    flie

The required files in code preprocessing.

    prepro_data/val_cut_300

Contains pre-processed data information.



