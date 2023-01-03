
import numpy as np, pandas as pd
import joblib
import sys
import os, warnings
warnings.filterwarnings('ignore') 



from sklearn.svm import SVC

model_fname = "model.save"
MODEL_NAME = "multi_class_base_svc_sklearn"


class Classifier(): 
    
    def __init__(self, C = 1.0, kernel = "rbf", degree = 3, **kwargs) -> None:
        self.C = float(C)
        self.kernel = kernel
        self.degree = int(degree)        
        self.model = self.build_model()     
        self.train_X = None
        
        
    def build_model(self): 
        model = SVC(C = self.C, degree = self.degree, kernel = self.kernel, probability=True)
        return model
    
    
    def fit(self, train_X, train_y):    
        self.train_X = train_X    
        self.model.fit(train_X, train_y)            
        
    
    def predict(self, X): 
        preds = self.model.predict(X)
        return preds 
    
    
    def predict_proba(self, X): 
        preds = self.model.predict_proba(X)
        return preds 
    

    def summary(self):
        self.model.get_params()
        
    
    def evaluate(self, x_test, y_test): 
        """Evaluate the model and return the loss and metrics"""
        if self.model is not None:
            return self.model.score(x_test, y_test)        

    
    def save(self, model_path): 
        joblib.dump(self, os.path.join(model_path, model_fname))
        


    @classmethod
    def load(cls, model_path):         
        model = joblib.load(os.path.join(model_path, model_fname))
        return model


def save_model(model, model_path):
    model.save(model_path)
      

def load_model(model_path): 
    model = joblib.load(os.path.join(model_path, model_fname))   
    return model


