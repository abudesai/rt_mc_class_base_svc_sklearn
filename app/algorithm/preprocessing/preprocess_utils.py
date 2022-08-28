import numpy as np, pandas as pd
import joblib
import pprint
import sys, os


def get_cat_and_num_vars_lists(data_schema):      
    cat_vars, num_vars = [], []
    attributes = data_schema["inputDatasets"]["multiClassClassificationBaseMainInput"]["predictorFields"]   
    for attribute in attributes: 
        if attribute["dataType"] == "CATEGORICAL":
            cat_vars.append(attribute["fieldName"])
        elif attribute["dataType"] == "NUMERIC":
            num_vars.append(attribute["fieldName"])

    # print("# cat_vars: ", len(cat_vars), "# num_vars: ", len(num_vars)); sys.exit()
    return cat_vars, num_vars 


def verify_data_columns_in_schema(data, pp_params): 
    all_vars = pp_params["cat_vars"] + pp_params["num_vars"]
    useable_vars = [var for var in all_vars if var in data.columns]
    
    if len(useable_vars) == 0:
        raise Exception('''
            Error: Given training data does not have any input attributes expected as per 
            the input schema. Do you have the wrong data, or the wrong schema? ''')
    return 


def get_vars_with_nas(data, pp_params):     
    vars_with_na = [var for var in data.columns 
                    if data[var].isnull().sum() > 0] 
    
    cat_na = [var for var in pp_params["cat_vars"] if var in vars_with_na]
    num_na = [var for var in pp_params["num_vars"] if var in vars_with_na]
    # print("# cat_na: ", len(cat_na), "# num_na: ", len(num_na)); sys.exit()

    return cat_na, num_na


def get_cat_vars_with_missing_impute_for_na(data, pp_params, model_cfg): 
    threshold = model_cfg['pp_params']['cat_params']['max_perc_miss_for_most_freq_impute']
    with_string_missing = [ var for var in pp_params["cat_na"] 
            if data[var].isnull().mean() >=  threshold]
    return with_string_missing



def get_cat_vars_with_frequent_cat_impute_for_na(data, pp_params, model_cfg): 
    threshold = model_cfg['pp_params']['cat_params']['max_perc_miss_for_most_freq_impute']
    with_freq_cat = [ var for var in pp_params["cat_na"] 
            if data[var].isnull().mean() <  threshold]
    return with_freq_cat



def get_preprocess_params(data, data_schema, model_cfg): 
    # initiate the pp_params dict
    pp_params = {}   
            
    # set the id attribute
    pp_params["id_field"] = data_schema["inputDatasets"]["multiClassClassificationBaseMainInput"]["idField"]   
    
    # set the target attribute
    pp_params["target_attr_name"] = data_schema["inputDatasets"]["multiClassClassificationBaseMainInput"]["targetField"]      
    
    # get the list of categorical and numeric variables and set in the params dict
    cat_vars, num_vars = get_cat_and_num_vars_lists(data_schema)    
    pp_params["cat_vars"], pp_params["num_vars"] = cat_vars, num_vars       
    
    # create list of variables to retain in the data - id, cat_vars, and num_vars
    pp_params["retained_vars"] = [pp_params["id_field"]] + \
        [pp_params["target_attr_name"]] + cat_vars + num_vars    
    
    # verify that the given data matches the input_schema
    verify_data_columns_in_schema(data, pp_params)    
    
    # get list of categorical and numeric variables with missing values
    cat_na, num_na = get_vars_with_nas(data, pp_params) 
    pp_params["cat_na"], pp_params["num_na"] = cat_na, num_na     
    
    # get list of categorical variables where the perc of missing values exceeds threshold (see model_config.json)
    # for these variables, we will set the missing values to 'missing' in pre-processing pipeline
    with_string_missing = get_cat_vars_with_missing_impute_for_na(data, pp_params, model_cfg)
    with_freq_cat = get_cat_vars_with_frequent_cat_impute_for_na(data, pp_params, model_cfg)
    pp_params["cat_na_impute_with_str_missing"], pp_params["cat_na_impute_with_freq"] = with_string_missing, with_freq_cat
    
    # pprint.pprint(pp_params)    
    return pp_params


def get_target_classes(data, pp_params): 
    target_classes = list(data[pp_params["target_attr_name"]].drop_duplicates())
    return target_classes