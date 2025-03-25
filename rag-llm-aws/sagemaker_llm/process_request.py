import logging
import asyncio
from .LineIterator import LineIterator
import os, json
from dotenv import load_dotenv  
load_dotenv()


# AWS Credentials
ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
ENDPOINT_NAME = os.getenv("AWS_SM_ENDPOINT_NAME")
REGION = os.getenv("AWS_REGION")

async def process_sync_completion(data, temperature=0.0, max_tokens=512, conn=None):


    import json
    import sagemaker
    import boto3

    sagemaker_runtime = boto3.client('sagemaker-runtime',
                                        aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
                region_name=REGION)
    endpoint_name = ENDPOINT_NAME


    try:
        inference_params = {
                "do_sample": False,
                "top_p": 0.6,
                "temperature": 0.1,
                "top_k": 50,
                "max_new_tokens": 1024,
                "repetition_penalty": 1.03,
                "return_full_text": False
                }
        payload = {
            "inputs":  data,
            "parameters": inference_params,
            "stream": False
        }


        try:
            # Invoke the endpoint
            response = sagemaker_runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                Body= json.dumps(payload).encode("utf-8"), #json.dumps(data),
                ContentType='application/json',
                Accept='application/json'
            )
            #print(response['Body'].read())
            # Read the response
            result = response['Body'].read().decode('utf-8')
            data =json.loads(result)
            #print(f"Result: {data}")
            for item in data:
                gen_text = item['generated_text']
                start_index = gen_text.rfind('<|eot_id|>assistant')
                str_len = len('<|eot_id|>assistant')
                full_answer = gen_text[start_index+str_len:]
                print(f"full answer: {full_answer}")
                next_sug_index = full_answer.find('assistant\n\n')
                first_answer = None
                if(next_sug_index > 0):
                    first_answer = full_answer[0:next_sug_index]
                    return first_answer
                else:
                    return full_answer
        except Exception as e:
            logging.error("Error:", e)
    except Exception as ex:
        logging.error(f"error in processing post request to model", exc_info=True)


async def process_async_completion(data, temperature=0.0, max_tokens=512, conn=None):
    import json
    import sagemaker
    import boto3


    sagemaker_runtime = boto3.client('sagemaker-runtime',
                                        aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
                region_name=REGION)
    #endpoint_name = "finbloom-testing-llama3-8b-instruct"
    endpoint_name = ENDPOINT_NAME

    try:


        inference_params = {
                "do_sample": False,
                "top_p": 0.6,
                "temperature": 0.1,
                "top_k": 50,
                "max_new_tokens": 1024,
                "repetition_penalty": 1.03,
                "return_full_text": False
                }


        payload = {
            "inputs":  data,
            "parameters": inference_params,
            "stream": True
        }
        try:
            logging.info(f"invokign sm")
            response_stream = sagemaker_runtime.invoke_endpoint_with_response_stream(
                EndpointName=endpoint_name,
                Body=json.dumps(payload).encode("utf-8"),
                ContentType="application/json"
            )
            logging.info(f"cm invokation successful")
            return response_stream['Body']
        except Exception as e:
            logging.error(f"Error: {e}", exc_info=True)
            raise e
    except Exception as ex:
        logging.error(f"error in processing post request to model: {ex}", exc_info=True)
        raise ex
