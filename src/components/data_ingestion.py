import os
import sys
from src.logger import logger
from dataclasses import dataclass
import pandas as pd
from src.exception import CustomException
from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig:
    """
        Data ingestion Config class which return three file paths.
    """
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")


class DataIngestion:
    """
        1 - Here this class is taking the config path
        2 - taking the dataset from a specific folder
        3 - checking the artifects folder exist or not and splitting the data into train and test
        4 - returning the train and test set file path
    """
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def get_data(self) -> pd.DataFrame:
        return pd.read_csv("notebooks/dataset/students.csv")

    def initiate_data_ingestion(self):
        logger.info("Entered Data ingestion method or Component..")
        try:
            df = self.get_data()
            logger.info("Read the dataset as dataframe")
            # checking the Folder is Exist Or Not
            os.makedirs(os.path.dirname(
                self.ingestion_config.train_data_path), exist_ok=True)
            # Saving the data set as data.csv
            df.to_csv(self.ingestion_config.raw_data_path,
                      index=False, header=True)
            logger.info("Train and Test data Split Initiated")
            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=42)
            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,
                            index=False, header=True)
            logger.info("Data Ingestion Completed...")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys) from e