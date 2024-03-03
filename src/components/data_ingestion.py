import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

#in data ingetion component any input that is required will be given through this class
#inside a class to define a class variable we usee init but by using this decorator dataclass we will be able to directly define the class variable, in this case class variable is train_data_path
@dataclass 
class DataIngestionConfig:   
    train_data_path: str=os.path.join('artifact',"train.csv") 
     #data ingestion output will be saved in this path here artifact and train.csv is file name
    
    test_data_path: str=os.path.join('artifact',"test.csv")
        
    raw_data_path: str=os.path.join('artifact',"raw.csv")
        
'''if you are only defining variables the you can use the decorator dataclass but if you have some other functions inside the class then it is
   suggested to use init'''

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig() #above 3 paths will be saved in the class variable ingestion_config
        
    def initiate_data_ingestion(self):#if data in db this fn reads it from there
        logging.info("entered the data ingestion method or component")
        try:
            df=pd.read_csv('notebook\StudentsPerformance.csv')
            logging.info("read the dataset as df")
            # create artifact folder
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            logging.info('train test split initiated')
            train_set,test_set=train_test_split(df,test_size=.2,random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            
            logging.info("ingestion of the data is complete")
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys) 
        
        