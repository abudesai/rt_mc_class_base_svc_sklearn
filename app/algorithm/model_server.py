import numpy as np, pandas as pd
import os, sys
import pprint
import json
from lime import lime_tabular
import warnings
import pprint

warnings.filterwarnings("ignore")
os.environ["MPLCONFIGDIR"] = os.getcwd() + "/configs/"


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
        self.id_field_name = self.data_schema["inputDatasets"][
            "multiClassClassificationBaseMainInput"
        ]["idField"]
        self.has_local_explanations = True
        self.MAX_LOCAL_EXPLANATIONS = 3

    def _get_preprocessor(self):
        if self.preprocessor is None:
            self.preprocessor = pipeline.load_preprocessor(self.model_path)
        return self.preprocessor

    def _get_model(self):
        if self.model is None:
            self.model = mc_classifier.load_model(self.model_path)
        return self.model

    def _get_predictions(self, data, return_probs=True):
        preprocessor = self._get_preprocessor()
        model = self._get_model()

        if preprocessor is None:
            raise Exception("No preprocessor found. Did you train first?")
        if model is None:
            raise Exception("No model found. Did you train first?")

        # transform data - returns a dict of X (transformed input features) and Y(targets, if any, else None)
        proc_data = preprocessor.transform(data)
        # Grab input features for prediction
        pred_X, pred_ids = proc_data["X"].astype(np.float), proc_data["ids"]
        # make predictions
        if return_probs:
            preds = model.predict_proba(pred_X)
        else:
            preds = model.predict(pred_X)

        return preds, pred_ids

    def predict_proba(self, data):
        preds, pred_ids = self._get_predictions(data, return_probs=True)
        class_names = pipeline.get_class_names(self.preprocessor, model_cfg)
        id_df = pd.DataFrame(pred_ids, columns=[self.id_field_name])

        # return the prediction df with the id and class probability fields
        preds_df = pd.concat([id_df, pd.DataFrame(preds, columns=class_names)], axis=1)
        return preds_df

    def predict(self, data):
        preds_df = self.predict_proba(data)
        class_names = [str(c) for c in preds_df.columns[1:]]
        preds_df["prediction"] = pd.DataFrame(
            preds_df[class_names], columns=class_names
        ).idxmax(axis=1)
        preds_df.drop(class_names, axis=1, inplace=True)
        return preds_df

    def predict_to_json(self, data): 
        predictions_df = self.predict_proba(data)
        predictions_df.columns = [str(c) for c in predictions_df.columns]
        class_names = predictions_df.columns[1:]

        predictions_df["__label"] = pd.DataFrame(
            predictions_df[class_names], columns=class_names
        ).idxmax(axis=1)

        # convert to the json response specification
        id_field_name = self.id_field_name
        predictions_response = []
        for rec in predictions_df.to_dict(orient="records"):
            pred_obj = {}
            pred_obj[id_field_name] = rec[id_field_name]
            pred_obj["label"] = rec["__label"]
            pred_obj["probabilities"] = {
                str(k): np.round(v, 5)
                for k, v in rec.items()
                if k not in [id_field_name, "__label"]
            }
            predictions_response.append(pred_obj)
        return predictions_response

    def explain_local(self, data):

        if data.shape[0] > self.MAX_LOCAL_EXPLANATIONS:
            msg = f"""Warning!
            Maximum {self.MAX_LOCAL_EXPLANATIONS} explanation(s) allowed at a time. 
            Given {data.shape[0]} samples. 
            Selecting top {self.MAX_LOCAL_EXPLANATIONS} sample(s) for explanations."""
            print(msg)

        preprocessor = self._get_preprocessor()
        model = self._get_model()
        # transform data - returns a dict of X (transformed input features) and Y(targets, if any, else None)
        proc_data = preprocessor.transform(data.head(self.MAX_LOCAL_EXPLANATIONS))
        pred_X, ids = proc_data["X"].astype(np.float), proc_data["ids"]

        class_names = pipeline.get_class_names(self.preprocessor, model_cfg)
        feature_names = list(pred_X.columns)

        print(f"Generating local explanations for {pred_X.shape[0]} sample(s).")

        explainer = lime_tabular.LimeTabularExplainer(
            model.train_X.values,
            mode="classification",
            class_names=class_names,
            feature_names=feature_names,
        )

        model = self._get_model()
        explanations = []
        for i, row in pred_X.iterrows():      

            explanation = explainer.explain_instance(
                row, model.predict_proba, top_labels=len(class_names)
            )            

            pred_class_idx =  int(explanation.predict_proba.argmax())
            pred_class = str( class_names[pred_class_idx] )
            pred_class_prob = np.round(explanation.predict_proba.max(), 5)
            probabilities = {
                k:np.round(v, 5) for k,v in zip(class_names, explanation.predict_proba)
            }
            
            sample_expl_dict = {}
            for j, c in enumerate(class_names):
                class_exp_dict = {}
                class_exp_dict["class_prob"] = round(
                    float(explanation.predict_proba[j]), 5
                )
                class_exp_dict["intercept"] = np.round(explanation.intercept[j], 5)
                feature_impacts = {}
                for feature_idx, feature_impact in explanation.local_exp[j]:
                    feature_impacts[feature_names[feature_idx]] = np.round(
                        feature_impact, 5
                    )

                class_exp_dict["feature_scores"] = feature_impacts
                sample_expl_dict[str(c)] = class_exp_dict

            explanations.append({
                self.id_field_name: ids[i],
                "label": pred_class,
                "label_prob": pred_class_prob,
                "probabilities": probabilities,
                "explanations": sample_expl_dict
            })
        # ------------------------------------------------------
        
        explanations = {"predictions": explanations}
        explanations = json.dumps(explanations, cls=utils.NpEncoder, indent=2)
        return explanations
