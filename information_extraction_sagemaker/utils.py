import re
import json
import logging
import pandas as pd

logging.getLogger("sagemaker").setLevel(logging.ERROR)
logging.getLogger("sagemaker.config").setLevel(logging.ERROR)

import boto3
from sagemaker.predictor import Predictor
from sagemaker.base_deserializers import JSONDeserializer
from sagemaker.base_serializers import JSONSerializer
from sagemaker.jumpstart.model import JumpStartModel
from sagemaker.s3 import S3Uploader
from sagemaker import Session
from sagemaker import hyperparameters

sagemaker_client = boto3.client("sagemaker")
iam = boto3.client("iam")




def get_role_arn(
    sagemaker_session=None,
    sagemaker_execution_role_name="SageMakerExecutionRole",
):
    """Get the Amazon SageMaker Execution Role ARN
    If there is a SageMaker session with a role, return it. Otherwise, get
    the role from the IAM directly.

    Args:
        sagemaker_session (sagemaker.Session): A SageMaker Session
        sagemaker_execution_role_name (str): The name of the role to get (not
        requried if a sagemaker_session is provided)
    """
    if not sagemaker_session:
        sagemaker_session = Session()
    arn = sagemaker_session.get_caller_identity_arn()

    if ":role/" in arn:
        return arn
    arn = iam.get_role(RoleName=sagemaker_execution_role_name)["Role"]["Arn"]
    return arn


def get_region():
    """Get the region of the current SageMaker notebook instance"""
    return Session().boto_region_name


def llama2_chat(
    predictor,
    user,
    temperature=0.1,
    max_tokens=512,
    top_p=0.9,
    system=None,
):
    """Constructs the payload for the llama2 model, sends it to the endpoint,
    and returns the response."""

    inputs = []
    if system:
        inputs.append({"role": "system", "content": system})
    if user:
        inputs.append({"role": "user", "content": user})

    payload = {
        "inputs": [inputs],
        "parameters": {
            "max_new_tokens": max_tokens,
            "top_p": top_p,
            "temperature": temperature,
        },
    }
    response = predictor.predict(payload, custom_attributes="accept_eula=true")
    return response


def llama2_parse_output(response):
    if len(response) > 0:
        response = response[0]

    generation = response["generation"]

    if isinstance(generation, dict):
        return generation["content"].strip()

    return generation.strip()



def parse_output(response):
    if len(response) > 0:
        response = response[0]

    return response["generated_text"].strip()


