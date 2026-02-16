
from project.pipeline.training_pipeline import Training_Pipeline 
from project.exception import CustomException 
from project.logger import logging
import sys

if __name__ == '__main__':
    try:
        logging.info('Started Training_Pipeline.......')
        pipeline = Training_Pipeline()
        pipeline.run_pipeline()
        logging.info('Training_Pipeline Completed')

    except Exception as e:
        raise CustomException (e,sys)


