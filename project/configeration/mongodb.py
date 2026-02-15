import os
import sys
import pymongo
import certifi
from project.logger import logging
from project.exception import CustomException
from project.constants import MONGODB_URL_KEY,Data_Base_Name


from dotenv import load_dotenv
load_dotenv()
ca = certifi.where()

class MongoDBClient:

    def __init__(self, database=Data_Base_Name): 
        try:
            mongodb_url = os.getenv(MONGODB_URL_KEY)
            if not mongodb_url:
                logging.info('Missing mongodb_url')
                raise ValueError('MongoDB URL is not found')
            
            self.client = pymongo.MongoClient(mongodb_url, tlsCAFile=ca)
            self.database = self.client[database]
            self.database_name = database

            logging.info(f'Connected to MongoDB database: {database}')
        
        except Exception as e:
            raise CustomException(e, sys)
        


        
