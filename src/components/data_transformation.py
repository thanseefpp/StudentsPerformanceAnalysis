# import os
# import sys
# from src.exception import CustomException
# from src.logger import logger
# from src.utils import save_object

# import pandas as pd
# import numpy as np

# from sklearn.impute import SimpleImputer
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder,StandardScaler

# from dataclasses import dataclass

import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logger
import os

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')
    

class DataTransformation:
    
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            logger.info(f'Numerical Columns :{numerical_columns}, done pipeline')
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ('one_hot_encoder',OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            logger.info(f'Categorical Columns :{categorical_columns}, done pipeline')
            return ColumnTransformer(
                [
                    ("numerical_pipeline",numerical_pipeline,numerical_columns),
                    ("categorical_pipeline",categorical_pipeline,categorical_columns)
                ]
            )
        except Exception as e:
            raise CustomException(e,sys) from e
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logger.info(f"Reading Train and Test datasets, train len :{train_df.shape}, test len: {test_df.shape}")
            
            pre_processing_obj = self.get_data_transformer_object()
            
            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]
            
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            logger.info('Applying preprocessing object on training dataframe and testing dataframe.')
            
            # transforming the train and test to array
            input_feature_train_arr=pre_processing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=pre_processing_obj.fit_transform(input_feature_test_df)
            
            # concatenation along the second axis
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logger.info("train and test concatenation done")
            
            # saving the preprocessor as .pkl file
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path, 
                obj=pre_processing_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys) from e
        