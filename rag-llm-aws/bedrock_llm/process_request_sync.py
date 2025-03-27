"""Helper utilities for working with Amazon Bedrock from Python notebooks"""
# Python Built-Ins:
import os
from typing import Optional
import sys
import json

# External Dependencies:
import boto3
from botocore.config import Config
import botocore
from datetime import datetime

import os, json
from dotenv import load_dotenv  
load_dotenv()


# AWS Credentials
ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

def get_bedrock_client(
    assumed_role: Optional[str] = None,
    region: Optional[str] = None,
    runtime: Optional[bool] = True,
):
    """Create a boto3 client for Amazon Bedrock, with optional configuration overrides

    Parameters
    ----------
    assumed_role :
        Optional ARN of an AWS IAM role to assume for calling the Bedrock service. If not
        specified, the current active credentials will be used.
    region :
        Optional name of the AWS Region in which the service should be called (e.g. "us-east-2").
        If not specified, AWS_REGION or AWS_DEFAULT_REGION environment variable will be used.
    runtime :
        Optional choice of getting different client to perform operations with the Amazon Bedrock service.
    """

    target_region = 'us-east-2'

    print(f"Create new client\n  Using region: {target_region}")
    session_kwargs = {"region_name": target_region}
    client_kwargs = {**session_kwargs}

    retry_config = Config(
        region_name=target_region,
        retries={
            "max_attempts": 10,
            "mode": "standard",
        },
    )
    session = boto3.Session(**session_kwargs)
    client_kwargs["aws_access_key_id"] = ACCESS_KEY
    client_kwargs["aws_secret_access_key"] = SECRET_KEY


    service_name='bedrock-runtime'

    bedrock_client = session.client(
        service_name=service_name,
        **client_kwargs
    )

    print("boto3 Bedrock client successfully created!")
    print(bedrock_client._endpoint)
    return bedrock_client

bedrock_runtime = get_bedrock_client()


# If you'd like to try your own prompt, edit this parameter!
prompt_data = """Command: Give me a travel plan from NY to Miami.
"""

# Define one or more messages using the "user" and "assistant" roles.
message_list = [{"role": "user", "content": [{"text": prompt_data}]}]

# Configure the inference parameters.
inf_params = {"maxNewTokens": 3000, "topP": 0.9, "topK": 20, "temperature": 0.7}
system_list = [
    {
        "text": "You are a content expert."
    }
]

modelId = "meta.llama3-3-70b-instruct-v1:0" #meta.llama3-2-1b-instruct-v1:0"  # (Change this, and the request body, to try different models)
accept = "application/json"
contentType = "application/json"

start_time = datetime.now()

request_body = {
    "prompt": prompt_data,
    "max_gen_len": 512,
    "temperature": 0.2,
    "top_p": 0.9
}
response = bedrock_runtime.invoke_model(
    body=json.dumps(request_body), modelId=modelId
)

response_body = json.loads(response['body'].read())
print(response_body)