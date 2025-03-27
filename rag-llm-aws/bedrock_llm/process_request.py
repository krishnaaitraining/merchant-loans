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
        config=retry_config,
        **client_kwargs
    )

    print("boto3 Bedrock client successfully created!")
    print(bedrock_client._endpoint)
    return bedrock_client



def main(bedrock_runtime):
    # If you'd like to try your own prompt, edit this parameter!
    prompt_data = """Command: Write me a blog about making strong business decisions as a leader.

    Blog:
    """

    # Define one or more messages using the "user" and "assistant" roles.
    message_list = [{"role": "user", "content": [{"text": prompt_data}]}]

    # Configure the inference parameters.
    inf_params = {"maxNewTokens": 250, "topP": 0.9, "topK": 20, "temperature": 0.7}
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
    response = bedrock_runtime.invoke_model_with_response_stream(
        body=json.dumps(request_body), modelId=modelId, accept=accept, contentType=contentType
    )
    chunk_count = 0
    time_to_first_token = None

    # Process the response stream
    stream = response.get("body")
    if stream:
        for event in stream:
            chunk = event.get("chunk")
            if chunk:
                # Print the response chunk
                chunk_json = json.loads(chunk.get("bytes").decode())
                # Pretty print JSON
                # print(json.dumps(chunk_json, indent=2, ensure_ascii=False))
                if "generation" in chunk_json:
                    yield chunk_json['generation']
                elif "completion" in chunk_json:
                    yield chunk_json['completion']
                elif "text" in chunk_json:
                    yield chunk_json['text']
                elif "delta" in chunk_json:
                    yield chunk_json['delta']
                elif "error" in chunk_json:
                    yield chunk_json['error']
                else:
                    yield chunk_json

                chunk_count += 1
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")
                    # print(f"{current_time} - ", end="")
                    
        print(f"Total chunks: {chunk_count}")
    else:
        print("No response stream received.")


bedrock_runtime = get_bedrock_client()

for text_chunk in main(bedrock_runtime):
    print(text_chunk)
print("everything completed")