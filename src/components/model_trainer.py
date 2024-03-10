import os
import sys

from src.pipeline.exception import CustomException
from src.pipeline.logger import logging
from src.components.data_ingestion import DataIngestion
from src.pipeline.utils import save_object, evaluate_models

from dataclasses import dataclass

import pandas as pd
import numpy as np

from catboost import CatBoostRegressor
from sklearn.ensemble import (
       AdaBoostRegressor,
       GradientBoostingRegressor,
       RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from joblib import dump, load
import dill


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifact','model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("splittin train test data into input and output")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models={
                
                'Random Forest':RandomForestRegressor,
                "DecisionTree": DecisionTreeRegressor ,
                 "GradientBoosting": GradientBoostingRegressor ,
                  "LinearRegressor":LinearRegression  ,
                 "KNeighbors": KNeighborsRegressor,
                 "XGBRegressor": XGBRegressor,
                 "CatBoost": CatBoostRegressor,
                 "AdaBoost":  AdaBoostRegressor
            }
            
            
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
          
                
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model=models[best_model_name]
            
            if best_model_score< 0.6:
                return CustomException("No best model")
            
            logging.info(f"best model found on train and test data")
            
            save_object(
                    file_path=self.model_trainer_config.trained_model_file_path,
                    obj=best_model
                )
            
            # Load the saved model
#             with open(self.model_trainer_config.trained_model_file_path, "rb") as file_obj:
#                 best_model = dill.load(file_obj)
              # Instantiate the selected model class
            best_model_instance = best_model()

            best_model_instance.fit(X_train, y_train)
              
            predicted=best_model_instance.predict(X_test)

            r2_square=r2_score(y_test,predicted)
            return r2_square
               
                
        except Exception as e:
            raise CustomException(e,sys)