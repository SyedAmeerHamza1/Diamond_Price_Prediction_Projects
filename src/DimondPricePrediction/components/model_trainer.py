import pandas as pd
import numpy as np
import os
import sys

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet


from src.DimondPricePrediction.exception import CustomException
from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.utils.utils import save_obj, evaluate_models

from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    trained_model_file_path= os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_train_config= ModelTrainerConfig()


    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info("splitting dependent and independent variables from train and test data")

            X_train, X_test, y_train, y_test=(
                train_arr[:,:-1],
                test_arr[:,:-1],
                train_arr[:, -1],
                test_arr[:, -1]
            )

            models={
                "Linear_Regressen":LinearRegression(),
                "Lasso":Lasso(),
                "Ridge":Ridge(),
                "ElasticNet":ElasticNet()
            }

            params={
                "Linear_Regressen":{
                    #"positive":["True", "False"]
                },
                "Lasso":{
                    "alpha":[1.0,0.5,1.5,0.8],
                    "max_iter": [1000,800,900,1500]
                },
                "Ridge":{
                    "alpha":[1.0,0.5,1.5,0.8],
                    "max_iter": [1000,800,900,1500]
                },
                "ElasticNet":{
                    "alpha":[1.0,0.5,1.5,0.8],
                    "max_iter": [1000,800,900,1500],
                    "l1_ratio":[0.5, 0.8,0.2,0.7]
                }
            }
        
            model_report:dict= evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params)

            print(model_report)
            print("\n=========================\n")

            logging.info(f"model report:{model_report}")

            # To get best model score from dict
            best_model_score= max(sorted(model_report.values()))

            best_model_name= list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model= models[best_model_name]

            print(f"Best model found, Model Name:{best_model_name}, R2 Score:{best_model_score}")
            print("\n=========================\n")

            logging.info(f"Best model found, Model Name:{best_model_name}, R2 Score:{best_model_score}")

            save_obj(
                file_path=self.model_train_config.trained_model_file_path,
                obj=best_model
            )
        
        except Exception as e:
            raise CustomException(e, sys)