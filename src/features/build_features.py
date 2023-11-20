from dataclasses import dataclass
import sys
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path = os.path.join("models", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.dataTransformationConfig = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ['reading_score', 'writing_score']
            categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch',
       'test_preparation_course']
            numerical_pipeline = Pipeline(
                steps=[
                    (
                        "Imputer",SimpleImputer(strategy="median")
                    ),
                    (
                        "scaler",StandardScaler()
                    )]
                )
            logging.info("Numerical Columns Standard Scaling Completed")
            categorical_pipeline = Pipeline(
                steps=[(
                    "imputer",SimpleImputer(strategy="most_frequent")
                ),
                (
                    "one_hot_encoder",OneHotEncoder()
                ),
                (
                    "scaler",StandardScaler()
                )]
            )
            logging.info("Categorical Column Encoding Completed")


            preprocessor=ColumnTransformer([
                ("num_pipeline",numerical_pipeline,numerical_columns),
                ("cat_pipeline",categorical_pipeline,categorical_columns)
            ])

            return preprocessor
        except Exception as e:
            print(e)
            
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data Completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()
            target_column = "math_score"
            
            input_feature_training_df = train_df.drop(columns=[target_column],axis=1)
            target_feature_training_df = train_df[target_column]

            input_feature_testing_df = test_df.drop(columns=[target_column],axis=1)
            target_feature_testing_df = test_df[target_column]

            logging.info("Applying preprocessing object on training  dataframe and testing dataframe")

            input_feature_training_array = preprocessing_obj.fit_transform(input_feature_training_df)
            input_feature_testing_array = preprocessing_obj.transform(input_feature_testing_df)

            train_arr = np.c_[input_feature_training_array,np.array(target_feature_training_df)]
            test_arr = np.c_[input_feature_testing_array,np.array(target_feature_testing_df)]

            logging.info("Saved Preprocessing object")
            save_object(filepath=self.dataTransformationConfig.preprocessor_ob_file_path,obj=preprocessing_obj)
            return(
                train_arr,
                test_arr,
                self.dataTransformationConfig.preprocessor_ob_file_path
            )
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
