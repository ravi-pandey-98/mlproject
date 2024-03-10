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
            
            params={
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                
                
                "DecisionTree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                
                "GradientBoosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "LinearRegression":{},
                
                "KNeighbors":{},
                
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoost":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            
            
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)
          
                
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