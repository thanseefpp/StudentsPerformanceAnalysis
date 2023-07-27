#----------------------------------------- IMPORTING DEPENDENCIES -------------------------------------#
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.pipeline.train_pipeline import ModelTrainer
from src.logger import logger
#----------------------------------------- FUNCTIONS/CLASSES -------------------------------------#


def train_pipeline():
    obj = DataIngestion()
    train_path,test_path = obj.initiate_data_ingestion()
    data_transformer = DataTransformation()
    train_arr,test_arr,_ = data_transformer.initiate_data_transformation(train_path=train_path,test_path=test_path)
    model_train = ModelTrainer()
    r2score = model_train.initiate_model_train(train_arr,test_arr)
    logger.info(f"Model Training Done and the R2 Score is :{r2score}")
if __name__ == "__main__":
    train_pipeline()