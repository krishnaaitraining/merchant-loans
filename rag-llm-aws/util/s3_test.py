import boto3
import os
from dotenv import load_dotenv  
load_dotenv()

# Creating an S3 access object
obj = boto3.client("s3")
# AWS Credentials
ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
REGION = os.getenv("AWS_REGION")
BUCKET_NAME = "finbloom-testbucket1new"



def list_contents(bucket_name):
    # Create S3 Client
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        region_name=REGION,
    )

    # List objects in the bucket
    response = s3_client.list_objects_v2(Bucket=bucket_name)

    # Check if objects exist
    if "Contents" in response:
        print("Files in the S3 bucket:")
        for obj in response["Contents"]:
            print(obj["Key"])
    else:
        print("Bucket is empty or doesn't exist.")


def upload_file(bucket_name, file_name):
    # Create S3 Client
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        region_name=REGION,
    )

    # Upload File
    s3_key = "uploaded-file.txt"  # Change to the desired S3 object name

    try:
        s3_client.upload_file(file_name, bucket_name, s3_key)
        print(f"File '{file_name}' uploaded successfully as '{s3_key}' in bucket '{BUCKET_NAME}'.")
    except Exception as e:
        print(f"Error uploading file: {e}")

#upload_file(BUCKET_NAME, "allmovies.csv")


def download_file(bucket_name, file_name):
    # Create S3 Client
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        region_name=REGION,
    )

    # Download File
    download_path = "downloaded-file.txt"  # Change to your local path

    try:
        s3_client.download_file(bucket_name, file_name, download_path)
        print(f"File '{file_name}' downloaded successfully as '{download_path}'.")
    except Exception as e:
        print(f"Error downloading file: {e}")

list_contents(BUCKET_NAME)