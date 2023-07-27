import sys
import pandas as pd
from src.exception import CustomException
from src.components.data_transformation import DataTransformationConfig
from src.pipeline.train_pipeline import ModelTrainerConfig
from src.utils import load_object
from src.logger import logger


class PredictPipeline:
    def __init__(self):
        self.trained_model_path = ModelTrainerConfig()
        self.preprocessor_file_path = DataTransformationConfig()

    def predict(self, features):
        try:
            model_path = self.trained_model_path.trained_model_file_path
            preprocessor_path = self.preprocessor_file_path.preprocessor_obj_file_path
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            logger.info("Before Starting the prediction")
            return model.predict(data_scaled)
        except Exception as e:
            raise CustomException(e, sys) from e


class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):
        
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self) -> pd.DataFrame:
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys) from e
