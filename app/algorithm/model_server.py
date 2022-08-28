import numpy as np, pandas as pd
import os, sys

import algorithm.utils as utils
import algorithm.preprocessing.pipeline as pipeline
import algorithm.model.mc_classifier as mc_classifier


# get model configuration parameters 
model_cfg = utils.get_model_config()


class ModelServer:
    def __init__(self, model_path): 
        self.model_path = model_path
        self.preprocessor = None
        self.model = None
    
    
    def _get_preprocessor(self): 
        if self.preprocessor is None: 
            try: 
                self.preprocessor = pipeline.load_preprocessor(self.model_path)
                return self.preprocessor
            except: 
                print(f'Could not load preprocessor from {self.model_path}. Did you train the model first?')
                return None
        else: return self.preprocessor
    
    def _get_model(self): 
        if self.model is None: 
            try: 
                self.model = mc_classifier.load_model(self.model_path)
                return self.model
            except: 
                print(f'Could not load model from {self.model_path}. Did you train the model first?')
                return None
        else: return self.model
        
    
    def _get_predictions(self, data, data_schema, return_probs = True):  
        preprocessor = self._get_preprocessor()
        model = self._get_model()
        
        if preprocessor is None:  raise Exception("No preprocessor found. Did you train first?")
        if model is None:  raise Exception("No model found. Did you train first?")
                    
        # transform data - returns a dict of X (transformed input features) and Y(targets, if any, else None)
        proc_data = preprocessor.transform(data)          
        # Grab input features for prediction
        pred_X, pred_ids = proc_data['X'].astype(np.float), proc_data['ids']      
        # make predictions
        if return_probs:
            preds = model.predict_proba( pred_X )
        else: 
            preds = model.predict( pred_X )
        
        return preds, pred_ids
    
    
    def predict_proba(self, data, data_schema):          
        preds, pred_ids = self._get_predictions(data, data_schema, return_probs=True)
        class_names = pipeline.get_class_names(self.preprocessor, model_cfg)        
        id_field_name = data_schema["inputDatasets"]["multiClassClassificationBaseMainInput"]["idField"]  
        id_df = pd.DataFrame(pred_ids, columns=[id_field_name])
        
        # return the prediction df with the id and class probability fields        
        preds_df = pd.concat( [ id_df, pd.DataFrame(preds, columns = class_names)], axis=1 )
        return preds_df 
    
    
    
    def predict(self, data, data_schema):        
        preds_df = self.predict_proba(data, data_schema)
        class_names = [ str(c) for c in preds_df.columns[1:] ]          
        preds_df["prediction"] = pd.DataFrame(preds_df[class_names], columns = class_names).idxmax(axis=1)     
        preds_df.drop(class_names, axis=1, inplace=True) 
        return preds_df
