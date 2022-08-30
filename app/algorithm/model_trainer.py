#!/usr/bin/env python

import os, warnings, sys
import pprint
warnings.filterwarnings('ignore')  

import numpy as np, pandas as pd
from sklearn.utils import shuffle

import algorithm.preprocessing.pipeline as pp_pipe
import algorithm.preprocessing.preprocess_utils as pp_utils
import algorithm.utils as utils
from algorithm.model.mc_classifier import Classifier
from algorithm.utils import get_model_config


# get model configuration parameters 
model_cfg = get_model_config()


def get_trained_model(data, data_schema, hyper_params):  
    
    # set random seeds
    utils.set_seeds()
    
    # perform train/valid split 
    train_data = data 
    # print('train_data shape:',  train_data.shape) 
    
    # preprocess data
    print("Pre-processing data...")
    train_data, _, preprocess_pipe = preprocess_data(train_data, None, data_schema)       
    train_X, train_y = train_data['X'].astype(np.float), train_data['y']
    # print(train_X.shape, train_y.shape) 
    
    # balance the targetclasses  
    train_X, train_y = get_resampled_data(train_X, train_y)
    # print(train_X.shape, train_y.shape) ; sys.exit()
    
    # Create and train model     
    print('Fitting model ...')  
    model = train_model(train_X, train_y, hyper_params)    
    
    return preprocess_pipe, model


def train_model(train_X, train_y, hyper_params):    
    # get model hyper-paameters parameters 
    model_params = { **hyper_params }
    
    # Create and train model   
    model = Classifier(  **model_params )  
    
    model.fit(
        train_X=train_X, train_y=train_y
    )  
    return model


def preprocess_data(train_data, valid_data, data_schema):
    # print('Preprocessing train_data of shape...', train_data.shape)
    pp_params = pp_utils.get_preprocess_params(train_data, data_schema, model_cfg) 
        
    # we want to get target_classes from both train and validation data. otherwise, we might
    # have a case where some target class was only observed in validation data. 
    if valid_data is not None: 
        full_data = pd.concat([train_data, valid_data], axis=0, ignore_index=True)
    else: 
        full_data = train_data
    pp_params["target_classes"] = pp_utils.get_target_classes(full_data , pp_params)
        
    preprocess_pipe = pp_pipe.get_preprocess_pipeline(pp_params, model_cfg)
    train_data = preprocess_pipe.fit_transform(train_data)
    # print("Processed train X/y data shape", train_data['X'].shape, train_data['y'].shape)
    
    if valid_data is not None: 
        valid_data = preprocess_pipe.transform(valid_data)
        # print("Processed valid X/y data shape", valid_data['X'].shape, valid_data['y'].shape)
        
    return train_data, valid_data, preprocess_pipe 


def get_resampled_data(X, y):  
    # if some minority class is observed only 1 time, and a majority class is observed 100 times
    # we dont over-sample the minority class 100 times. We have a limit of how many times
    # we sample. max_resample is that parameter - it represents max number of full population
    # resamples of the minority class. For this example, if max_resample is 3, then, we will only
    # repeat the minority class 2 times over (plus original 1 time). 
    max_resample = model_cfg["max_resample_of_minority_classes"]
    unique, class_count = np.unique(y, return_counts=True)
    # class_count = [ int(c) for c in class_count]
    max_obs_count = max(class_count)
    
    
    resampled_X, resampled_y = [], []
    for i, count in enumerate(class_count):
        if count == 0: continue
        # find total num_samples to use for this class
        size = max_obs_count if max_obs_count / count < max_resample else count * max_resample
        # if observed class is 50 samples, and we need 125 samples for this class, 
        # then we take the original samples 2 times (equalling 100 samples), and then randomly draw
        # the other 25 samples from among the 50 samples
        
        full_samples = size // count        
        idx = y == i
        for _ in range(full_samples):
            resampled_X.append(X.loc[idx])
            resampled_y.append(y.loc[idx])
        # find the remaining samples to draw randomly        
        remaining =  size - count * full_samples 
        idx_list = list(X.loc[idx].index)
        sampled_idx = np.random.choice(idx_list, size = remaining, replace=True)
        resampled_X.append(X.iloc[sampled_idx])
        resampled_y.append(y.iloc[sampled_idx])
    
    resampled_X = pd.concat(resampled_X, axis=0, ignore_index=True)    
    resampled_y = pd.concat(resampled_y, axis=0, ignore_index=True)    
    
    # shuffle the arrays
    resampled_X, resampled_y = shuffle(resampled_X, resampled_y)
    return resampled_X, resampled_y