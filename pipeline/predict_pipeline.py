import pandas as pd
from src.logger import logging
from src.utils import load_object
import sys
import os

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            model_path = os.path.join('models','model.pkl')
            preprocessor_path = os.path.join("models","preprocessor.pkl")
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            print("loaded Model and Preprocessor")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(e)


class CustomData:
    def __init__(self, gender: str,race_ethnicity: str, parental_level_of_education, lunch:str, test_preparation_course:str, reading_score:int, writing_score:int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def data_frame(self):
        try:
            cutom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(cutom_data_input_dict)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(e)
