
import pymongo
from pymongo.mongo_client import MongoClient

from pymongo.server_api import ServerApi
import pandas as pd
import logging
logging.basicConfig(level=logging.DEBUG, format="{asctime} - {levelname} - {message}",
 style="{",
datefmt="%m-%d-%Y %H:%M")

import os
from dotenv import load_dotenv

class Chats:

    @classmethod
    def retreive_user_details(cls,client, username):
        db = client['finbloom_dev']
        collection = db['users']
       # Assuming you have a connection to the database
        try:
            data = list(collection.find({"username": username}))
            # Branches DataFrame
            df = pd.DataFrame(data)
            if(df.empty):
                logging.info("No data found for the given username.")
                user_dict = {
                    "username": username,
                    "created_at": pd.Timestamp.now()
                }
                collection.insert_one(user_dict)
            else:
                df['_id'] = df['_id'].astype(str)
            logging.info("Data read successfully.")

        except Exception as e:
            logging.info("Error connecting to the database:", e)
        return df
    

    @staticmethod
    def retreive_user_chats(client, username):
        db = client['finbloom_dev']
        collection = db['chats']
       # Assuming you have a connection to the database
        try:
            data = list(collection.find({"username": username}).sort("updated_at", -1))
            # Branches DataFrame
            df = pd.DataFrame(data)
            if(df.empty):
                logging.info("No data found for the given username.")
            else:
                df['_id'] = df['_id'].astype(str)
                df['chat_id'] = df['_id'].astype(str)
            logging.info("Data read successfully.")

        except Exception as e:
            logging.info("Error connecting to the database:", e)
        return df
    


    def create_user_chat(this, client, username, chat_content, chat_title=None, chat_id=None, role=None):
        if(chat_id is None and chat_title is None):
            raise Exception("chat_id and chat_title cannot be None")
        
        db = client['finbloom_dev']
        collection = db['chats']
       # Assuming you have a connection to the database
        try:
            if(chat_id is None):
                chat_dict = {
                    "username": username,
                    "chat_title": chat_title,
                    "created_at": pd.Timestamp.now(),
                    "updated_at": pd.Timestamp.now()
                }
                result = collection.insert_one(chat_dict)
                chat_id = f"{result.inserted_id}"
                logging.info(f"new Chat successfully: {chat_id}")
            else:
                collection.find_one_and_update({"_id": chat_id}, {"$set": { "updated_at": pd.Timestamp.now()}}, new=True)
                logging.info(f"updated Chat successfully: {chat_id}")
            if(chat_content):
                chat_dict = {
                    "username": username,
                    "chat_id": chat_id,
                    "role": role,
                    "chat_content": chat_content,
                    "created_at": pd.Timestamp.now(),
                    "updated_at": pd.Timestamp.now()
                }
                msg_collection = db['chat_messages']
                content_result = msg_collection.insert_one(chat_dict)
                new_chat_details = content_result.inserted_id
                logging.info(f"new Chat successfully: {new_chat_details}")
        except Exception as e:
            logging.info("Error connecting to the database:", e)
        return chat_id



    def retreive_user_chat_details(this, client, username, chat_id):
        db = client['finbloom_dev']
        collection = db['chat_messages']
       # Assuming you have a connection to the database
        logging.info(f"chat_id: {chat_id}, username: {username}")
        try:
            data = list(collection.find({"username": username, "chat_id": chat_id}))
            # Branches DataFrame
            df = pd.DataFrame(data)
            if(df.empty):
                logging.info("No data found for the given username.")
            else:
                df['_id'] = df['_id'].astype(str)
                #df['chat_message_id'] = df['_id'].astype(str)
            logging.info("Data read successfully.")

        except Exception as e:
            logging.info("Error connecting to the database:", e)
        return df