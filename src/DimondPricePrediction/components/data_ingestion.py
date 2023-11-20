import pandas as pd
import numpy as np
import os
import sys

from src.DimondPricePrediction.exception import CustomException
from src.DimondPricePrediction.logger import logging

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path



@dataclass
class DataIngestionConfig:
    raw_data_path:str=os.path.join("artifacts", "raw.csv")
    train_data_path:str=os.path.join("artifacts", "train.csv")
    test_data_path:str=os.path.join("artifacts", "test.csv")

class DataIngestion:
    def __init__(self):
        self.Ingestion_Config= DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion has started")

        try:
            df=pd.read_csv(Path(os.path.join("notebooks/data","diamonds.csv")))
            logging.info("Read Dataset as DataFrame")

            os.makedirs(os.path.dirname(os.path.join(self.Ingestion_Config.raw_data_path)), exist_ok=True)

            df.to_csv(self.Ingestion_Config.raw_data_path, index=False)
            logging.info("Saved Row dataset in Artifacts folder")

            train_data, test_data= train_test_split(df, test_size=0.25, random_state=42)
            logging.info("Train test split is completed")

            train_data.to_csv(self.Ingestion_Config.train_data_path)

            test_data.to_csv(self.Ingestion_Config.test_data_path)

            logging.info("Data Ingestion is completes")

            return(
                self.Ingestion_Config.train_data_path,
                self.Ingestion_Config.test_data_path

            )


        except Exception as e:
            raise CustomException(e, sys)
