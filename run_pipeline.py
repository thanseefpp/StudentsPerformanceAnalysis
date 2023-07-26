from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

if __name__ == "__main__":
    obj = DataIngestion()
    train_path,test_path = obj.initiate_data_ingestion()
    data_transformer = DataTransformation()
    train_arr,test_arr,_ = data_transformer.initiate_data_transformation(train_path=train_path,test_path=test_path)
    
    
    