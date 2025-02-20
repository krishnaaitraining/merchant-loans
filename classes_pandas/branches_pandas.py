
import os
from dotenv import load_dotenv
import psycopg2
import configparser

from pymongo import MongoClient

class branch:
    #class variable
    number_of_branches = 0
    
    
    def __init__(self, bid, bname=None, bstr=None, bcity=None, bzip=None, bphone=None):
        #instance variable
        self.branch_id = bid
        self.branch_name = bname
        self.stree = bstr
        self.phone = bphone
        self.city = bcity
        self.zip = bzip
        #connect here
        


    def create_branch(self, member_id, name):
        config = configparser.ConfigParser()
        config.read('credentials.cfg')

       # Assuming you have a connection to the database
        try:
            connection = psycopg2.connect(
                dbname=config['postgres']['DB_NAME'],
                user=config['postgres']['DB_USER'],
                password=config['postgres']['DB_PASSWORD'],
                host=config['postgres']['DB_HOST'],
                port=config['postgres']['DB_PORT'],
                sslmode="require"
            )
            cursor = connection.cursor()
            
            #run the insert query

            query = f"insert into branches values (?, ?, ?)".format(self.branch_name, self.address, self.phone)
            cursor.execute(query)

            cursor.commit()
        except Exception as e:
            print("Error connecting to the database:", e)
        finally:    
            if connection:
                cursor.close()
                connection.close()
                #close connection
                #close cursor

    @staticmethod
    def create_branch_from_df():
        from sqlalchemy import create_engine
        import pandas as pd
        config = configparser.ConfigParser()
        config.read('credentials.cfg')
        
       # Assuming you have a connection to the database
        try:
            # Database connection parameters
            DATABASE_TYPE = 'postgresql'
            DBAPI = 'psycopg2'
            ENDPOINT = config['postgres']['DB_HOST'] # e.g., localhost or remote server
            USER = config['postgres']['DB_USER']
            PASSWORD = config['postgres']['DB_PASSWORD']
            PORT = 5432 # Default PostgreSQL port
            DATABASE = config['postgres']['DB_NAME']
            # Creating the engine
            engine = create_engine(f"{DATABASE_TYPE}+{DBAPI}://{USER}:{PASSWORD}@{ENDPOINT}:{PORT}/{DATABASE}")
                        
            # Branches DataFrame
            df = pd.read_csv('branches.csv')
            # Saving DataFrame to PostgreSQL
            df.to_sql('branches', engine, if_exists='append', index=False)
            print("Data saved to PostgreSQL successfully.")

        except Exception as e:
            print("Error connecting to the database:", e)


    @staticmethod
    def create_branch_mongo_from_df():
        import pymongo
        from pymongo.mongo_client import MongoClient

        from pymongo.server_api import ServerApi
        import pandas as pd
        import os
        from dotenv import load_dotenv
        # Load environment variables from .env file
        load_dotenv()

        uri = os.getenv("MONGO_DB_CONNECTION")
        #print(uri)
        client = MongoClient(uri, server_api=ServerApi('1'))
        db = client['members_test']
        collection = db['branches']
       # Assuming you have a connection to the database
        try:
                        
            # Branches DataFrame
            df = pd.read_csv('branches.csv')

            data_dict = df.to_dict(orient='records')

            # 4. Insert data into MongoDB
            collection.insert_many(data_dict)

            # Close the connection
            client.close()
            print("Data saved to mongodb successfully.")

        except Exception as e:
            print("Error connecting to the database:", e)

    @classmethod
    def retrieve_branch_mongo_from_df(cls):
        import pymongo
        from pymongo.mongo_client import MongoClient
        
        from pymongo.server_api import ServerApi
        import pandas as pd
        import os
        from dotenv import load_dotenv
        # Load environment variables from .env file
        load_dotenv()

        uri = os.getenv("MONGO_DB_CONNECTION")
        #print(uri)
        client = MongoClient(uri, server_api=ServerApi('1'))
        db = client['members_test']
        collection = db['branches']
       # Assuming you have a connection to the database
        try:
            data = list(collection.find())

            # Branches DataFrame
            df = pd.DataFrame(data)
            
            df['_id'] = df['_id'].astype(str)

            df.to_csv('branches_from_mongo.csv', index=False)
            client.close()
            print("Data read successfully.")

        except Exception as e:
            print("Error connecting to the database:", e)

branch.create_branch_from_df()

#br = branch(2)
#br.create_branch(1, "new branch")

#branch.create_branch(1, "new branch", "new")

#br.create_branch_mongo_from_df()

#branch.create_branch_mongo_from_df()

#branch.retrieve_branch_mongo_from_df()



