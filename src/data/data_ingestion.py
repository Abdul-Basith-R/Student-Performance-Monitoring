# import logging
import os
import sys
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.features import build_features
from src.models import train_model 

@dataclass
class DataIngestionConfig:
 train_data_path: str = os.path.join(os.path.join(
      "data", "processed", "train.csv"))
 test_data_path: str = os.path.join(os.path.join(
      "data", "processed", "test.csv"))
 raw_data_path: str = os.path.join(
     os.path.join("data", "raw", "stud.csv"))
     
class DataIngestion:
 def __init__(self):
  self.ingestion_config = DataIngestionConfig()

 def initiate_data_ingestion(self):
  try:
   df = pd.read_csv(self.ingestion_config.raw_data_path)
   logging.info("Read the CSV file")

   logging.info("Train Test Split initiated")
   train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
   train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
   test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
   logging.info("Ingestion Complete")

   return(self.ingestion_config.train_data_path,self.ingestion_config.test_data_path)
  except Exception as e:
   print(e)

if __name__ == "__main__":
 obj = DataIngestion()
 train_data,test_data = obj.initiate_data_ingestion()
 dataTransformation = build_features.DataTransformation()
 train_arr,test_arr,_ = dataTransformation.initiate_data_transformation(train_data,test_data)
 train = train_model.ModelTrainer()
 print(train.initiate_model_trainer(train_array=train_arr,test_array=test_arr))
 