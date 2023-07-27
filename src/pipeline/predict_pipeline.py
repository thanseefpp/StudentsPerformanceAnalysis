import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.components.data_transformation import DataTransformationConfig
from src.pipeline.train_pipeline import ModelTrainerConfig
from dataclasses import dataclass


class PredictPipeline:
    def __init__(self):
        self.trained_model_path = ModelTrainerConfig()
        self.preprocessor_file_path = DataTransformationConfig()

    def predict(self, features):
        try:
            model_path = self.trained_model_path.trained_model_file_path
            preprocessor_path = self.preprocessor_file_path.preprocessor_obj_file_path
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled = preprocessor.transform(features)
            return model.predict(data_scaled)
        except Exception as e:
            raise CustomException(e, sys) from e


@dataclass
class StudentDataTypeConfig:
    gender: str
    race_ethnicity: str
    parental_level_of_education: str
    lunch: str
    test_preparation_course: str
    reading_score: int
    writing_score: int


class CustomData:
    def __init__(self):
        self.student_data_type = StudentDataTypeConfig()

    def get_data_as_data_frame(self) -> pd.DataFrame:
        try:
            custom_data_input_dict = {
                "gender": [self.student_data_type.gender],
                "race_ethnicity": [self.student_data_type.race_ethnicity],
                "parental_level_of_education": [self.student_data_type.parental_level_of_education],
                "lunch": [self.student_data_type.lunch],
                "test_preparation_course": [self.student_data_type.test_preparation_course],
                "reading_score": [self.student_data_type.reading_score],
                "writing_score": [self.student_data_type.writing_score],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys) from e
