import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
        
def evaluate_models(X_train,y_train,X_test,y_test,models,param):
    try:
        report={}
        
        for i in range(len(list(models))):
#             model=list(models.values())[i]
#             model=model_class()
            model_class = list(models.values())[i]
            model = model_class()
            
            para = list(param.values())[i]
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
                         
            model.fit(X_train,y_train)
            
            y_train_pred=model.predict(X_train)
            y_test_pred=model.predict(X_test)
                         
            train_model_score=r2_score(y_train,y_train_pred)
            test_model_score=r2_score(y_test,y_test_pred)
                         
            report[list(models.keys())[i]]=test_model_score
                         
        return report
# def evaluate_models(X_train, y_train, X_test, y_test, models):
#     try:
#         report = {}

#         for model_name, model_class in models.items():
#             # Create an instance of the model
#             model = model_class()

#             # Fit the model to the training data
#             model.fit(X_train, y_train)

#             # Make predictions on both the training and testing data
#             y_train_pred = model.predict(X_train)
#             y_test_pred = model.predict(X_test)

#             # Evaluate the model using R-squared scores
#             train_model_score = r2_score(y_train, y_train_pred)
#             test_model_score = r2_score(y_test, y_test_pred)

#             # Store the test R-squared score in the report dictionary
#             report[model_name] = test_model_score

        
    except Exception as e:
        raise CustomException(e,sys)
    