from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import sqlite3


# Define default arguments for the DAG
default_args = {
 'owner': 'airflow',
 'start_date': datetime(2024, 1, 1),
 'retries': 1
}
# Define the DAG
dag = DAG(
 'simple_etl_pipeline',
 default_args=default_args,
 schedule_interval='@daily',
 catchup=False
)
# Task 1: Extract data
def extract_data():
    data = {'name': ['Alice', 'Bob', 'Charlie'], 'age': [25, 30, 35]}
    df = pd.DataFrame(data)
    df.to_csv('/tmp/raw_data.csv', index=False)

# Task 2: Transform data
def transform_data():
    df = pd.read_csv('/tmp/raw_data.csv')
    df['age'] = df['age'] + 5 # Adding 5 years to age
    df.to_csv('/tmp/transformed_data.csv', index=False)


# Task 3: Load data into a database
def load_data():
    conn = sqlite3.connect('/tmp/database.db')
    df = pd.read_csv('/tmp/transformed_data.csv')
    df.to_sql('people', conn, if_exists='replace', index=False)
    conn.close()

extract_task = PythonOperator(
 task_id='extract_data',
 python_callable=extract_data,
 dag=dag
)

transform_task = PythonOperator(
 task_id='transform_data',
 python_callable=transform_data,
 dag=dag
)

load_task = PythonOperator(
 task_id='load_data',
 python_callable=load_data,
 dag=dag
)

# Define task dependencies
extract_task >> transform_task >> load_task