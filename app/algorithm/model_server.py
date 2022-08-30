import numpy as np, pandas as pd
import os, sys
import pprint
import json
from lime import lime_tabular

import algorithm.utils as utils
import algorithm.preprocessing.pipeline as pipeline
import algorithm.model.mc_classifier as mc_classifier


# get model configuration parameters 
model_cfg = utils.get_model_config()


class ModelServer:
    def __init__(self, model_path, data_schema): 
        self.model_path = model_path
        self.data_schema = data_schema
        self.preprocessor = None
        self.model = None
        self.id_field_name = self.data_schema["inputDatasets"]["multiClassClassificationBaseMainInput"]["idField"]  
        self.has_local_explanations = True
        self.MAX_LOCAL_EXPLANATIONS = 3
    
    
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
        
    
    def _get_predictions(self, data, return_probs = True):  
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
    
    
    def predict_proba(self, data):          
        preds, pred_ids = self._get_predictions(data, return_probs=True)
        class_names = pipeline.get_class_names(self.preprocessor, model_cfg)         
        id_df = pd.DataFrame(pred_ids, columns=[self.id_field_name])
        
        # return the prediction df with the id and class probability fields        
        preds_df = pd.concat( [ id_df, pd.DataFrame(preds, columns = class_names)], axis=1 )
        return preds_df 
    
    
    def predict(self, data):        
        preds_df = self.predict_proba(data)
        class_names = [ str(c) for c in preds_df.columns[1:] ]          
        preds_df["prediction"] = pd.DataFrame(preds_df[class_names], columns = class_names).idxmax(axis=1)     
        preds_df.drop(class_names, axis=1, inplace=True) 
        return preds_df
   
    
    def explain_local(self, data): 
        
        if data.shape[0] > self.MAX_LOCAL_EXPLANATIONS:
            msg = f'''Warning!
            Maximum {self.MAX_LOCAL_EXPLANATIONS} explanation(s) allowed at a time. 
            Given {data.shape[0]} samples. 
            Selecting top {self.MAX_LOCAL_EXPLANATIONS} sample(s) for explanations.'''
            print(msg)
        
        preprocessor = self._get_preprocessor()        
        # transform data - returns a dict of X (transformed input features) and Y(targets, if any, else None)
        proc_data = preprocessor.transform(data.head(self.MAX_LOCAL_EXPLANATIONS))    
        pred_X, ids = proc_data['X'].astype(np.float), proc_data['ids']   
        
        class_names = pipeline.get_class_names(self.preprocessor, model_cfg)
        feature_names = list(pred_X.columns)
        
        print(f"Generating local explanations for {pred_X.shape[0]} sample(s).")   
        
        explainer = lime_tabular.LimeTabularExplainer(pred_X.values, mode="classification",
                        class_names=class_names, feature_names=feature_names )
        
        model = self._get_model()
        all_explanations = []   
        for i, row in pred_X.iterrows():             
            sample_expl_dict = {}
            sample_expl_dict[self.id_field_name] = ids[i]      
            
            explanation = explainer.explain_instance(row, model.predict_proba, top_labels=len(class_names))
            sample_expl_dict['predicted_class'] = class_names[int(explanation.predict_proba.argmax())] 
            sample_expl_dict['predicted_class_prob'] = round(float(explanation.predict_proba.max()), 5)
                        
            sample_expl_dict["explanations_per_class"] = {}
            for j, c in enumerate(class_names): 
                class_exp_dict = {}
                class_exp_dict['class_prob'] = round(float(explanation.predict_proba[j]), 5)               
                class_exp_dict['intercept'] = np.round(explanation.intercept[j], 5)                
                feature_impacts = {}
                for feature_idx, feature_impact in explanation.local_exp[j]:
                    feature_impacts[feature_names[feature_idx]] = np.round(feature_impact, 5)
                
                class_exp_dict["feature_impacts"] = feature_impacts                
                sample_expl_dict["explanations_per_class"][str(c)] = class_exp_dict                
            
            all_explanations.append(sample_expl_dict)
        
        all_explanations = json.dumps(all_explanations, cls=utils.NpEncoder, indent=2)
        return all_explanations
        


