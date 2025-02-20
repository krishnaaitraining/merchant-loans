import configparser
import pandas as pd
import psycopg2
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

class Transactions:


    @classmethod
    def retrieve_transactions(cls):
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
                        
            query = "select * from fi_members as mem, fi_member_transactions as trans where mem.member_id = trans.member_id;"
            df_postgres = pd.read_sql(query, engine)
            df_postgres.to_csv('transactions_from_db.csv', index=False)
            print(df_postgres.head())
            print("Data saved to PostgreSQL successfully.")

        except Exception as e:
            print("Error connecting to the database:", e)


Transactions.retrieve_transactions()