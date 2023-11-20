import pandas
import os
import sys

from src.DimondPricePrediction.exception import CustomException
from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.components.data_ingestion import DataIngestion
from src.DimondPricePrediction.components.data_transformation import DataTransformation
from src.DimondPricePrediction.components.model_trainer import ModelTrainer

Data_ingestion= DataIngestion()
train_data_path, test_data_path= Data_ingestion.initiate_data_ingestion()


Data_transformation= DataTransformation()
train_arr, test_arr=Data_transformation.initilize_data_transformation(train_data_path, test_data_path)

Model_trainer= ModelTrainer()
Model_trainer.initiate_model_training(train_arr, test_arr)