
import configparser
import pandas as pd
import psycopg2
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

class Member:

    member_serial_number = 0

    def __init__(self):
        '''self.id = None
        self.first_name = fname
        self.last_name = lname
        self.email = email
        self.phone_number = None

        self.age = age
        self.address = address
        self.salary = salary
        self.__ssn = None'''

    
    #private method
    def __retrieve_ssn(self):
        print(self.__ssn)

    def retrieve_member_details(self):
        member_serial_number = 20
        #Retrieve details from database using Email
        return "details"
    
    def get_connection(self):
        import configparser
        # Initialize the config parser
        config = configparser.ConfigParser()
        config.read('credentials.cfg')
        import os
        #from dotenv import load_dotenv
        import psycopg2
        connection = None
        # Database connection details
        # Default PostgreSQL port
        # Establishing the connection
        try:
            connection = psycopg2.connect(
                dbname=config['postgres']['DB_NAME'],
                user=config['postgres']['DB_USER'],
                password=config['postgres']['DB_PASSWORD'],
                host=config['postgres']['DB_HOST'],
                port=config['postgres']['DB_PORT'],
                sslmode="require"
            )
            print("Database connection successful!")
            
        except Exception as e:
            print("Error connecting to the database:", e)
        return connection

        #save member to database


    def create_new_member(self):

        connection = self.get_connection()
        if(connection is not None):
            cursor = connection.cursor()
            cursor.execute(f"""
            INSERT INTO fi_members (first_name, last_name, email)
            VALUES (%s, %s, %s) RETURNING member_id ;
            """, (self.first_name,self.last_name, self.email))
            connection.commit()
            id = cursor.fetchone()[0]
            print(f"Record inserted successfully!: {id}")
        #raise exception if email already exists
        else:
            raise Exception ("connection could not be established")
        return "id of inserted row"
    
    @classmethod
    def retrieve_members(cls):
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
                        
            query = "SELECT * FROM fi_members"
            df_postgres = pd.read_sql(query, engine)
            df_postgres.to_csv('members_fromdb.csv', index=False)
            print(df_postgres.head())
            print("Data saved to PostgreSQL successfully.")

        except Exception as e:
            print("Error connecting to the database:", e)


Member.retrieve_members()   

#mem = Member()
#mem.first_name = "test"
#mem.last_name = "new"


#new_member = Member(email="classa1@one.com", fname="testclass", lname="testclass")


#new_member.create_new_member()

"""print(new_member.member_serial_number)
print(Member.member_serial_number)

print(new_member.first_name)
print(Member.first_name) #error

print(Member.retrieve_serial_number_class())
print(new_member.retrieve_serial_number_class())

print(Member.retrieve_serial_number_static())
print(new_member.Member.retrieve_serial_number_static())

print(new_member.__ssn)"""