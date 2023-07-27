#----------------------------------------- IMPORTING DEPENDENCIES -------------------------------------#
import os
import sys
import dill
import pickle
import numpy as np 
import pandas as pd
from src.exception import CustomException
from src.logger import logger
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
#----------------------------------------- FUNCTIONS/CLASSES -------------------------------------#

def save_object(file_path, obj):
    """
        saving the object in a specific path
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logger.info(f"Object Saved at : {file_path}")
    except Exception as e:
        raise CustomException(e, sys) from e
    

def load_object(file_path):
    """
        Loading the object from a specific path
    """
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        raise CustomException(e, sys) from e
    
    
def evaluate_models(X_train, y_train,X_test,y_test,models,params):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]
            gs = GridSearchCV(model,para,cv=3,n_jobs=3,refit=False)
            gs.fit(X_train,y_train)
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)  # Train model
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            # appending the the model name as key and the score as value
            report[list(models.keys())[i]] = test_model_score
        return report

    except Exception as e:
        raise CustomException(e, sys) from e