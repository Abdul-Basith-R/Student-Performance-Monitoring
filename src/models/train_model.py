import sys
import os
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor 
from xgboost import XGBRFRegressor

from src.logger import logging
from src.utils import save_object,evaluate


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("models", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and test input data")

            X_train, X_test, y_train, y_test = (train_array[:,:-1],test_array[:,:-1],train_array[:,-1],test_array[:,-1])

            models={
                "AdaBoost Regressor" : AdaBoostRegressor(),
                "Random Forest":RandomForestRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "CatBoost Regressor":CatBoostRegressor(),
                "Linear Regression": LinearRegression(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "XGboost":XGBRFRegressor()
            }

            params = {
                "Decision Tree Regressor":{
                    'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
                    'splitter':['best','random'],
                    'max_features':['sqrt','log2'],
                },
                'Random Forest':{
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_features': ['sqrt', 'log2'],
                    'n_estimators':[8,16,32,64,128,256]
                },
                'Gradient Boosting':{
                    'loss':['squared_error','huber','absolute error','quantile'],
                    'learning_rate': [0.1,0.01,0.05,0.001],
                    'subsample': [0.6,0.7,0.8,0.75,0.85,0.9],
                    'criterion':['squared_error', 'friedman_mse'],
                    'max_features': ['sqrt', 'log2'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]

                },
                "Linear Regression": {},
                "XGboost":{
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoost Regressor":{
                    'depth':[6,10,8],
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'loss':['linear','square','exponential'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            model_performance: dict = evaluate(
                xtrain=X_train, ytrain=y_train, xtest=X_test, ytest=y_test, models=models, params=params)

            best_model_score = max(sorted(model_performance.values()))

            # To get best model name from dict

            if best_model_score > 0.6:
                best_model_name = list(model_performance.keys())[
                    list(model_performance.values()).index(best_model_score)
                ]
                best_model = models[best_model_name]
                save_object(filepath=ModelTrainerConfig.trained_model_file_path,obj=best_model)
                return best_model_score,best_model_name
            else:
                logging.info("NNo best Model found")

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(e)