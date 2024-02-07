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

# pylint: disable=invalid-name

incorrect_responses_log_file = "data/log_incorrect_responses.jsonl"
error_responses_log_file = "data/log_error_responses.jsonl"


INTENTS = [
    {
        "main_intent": "profile_update",
        "sub_intents": [
            "contact_info",
            "payment_info",
            "members",
        ],
    },
    {
        "main_intent": "health_cover",
        "sub_intents": [
            "add_extras",
            "add_hospital",
            "remove_extras",
            "remove_hospital",
            "new_policy",
            "cancel_policy",
        ],
    },
    {
        "main_intent": "life_cover",
        "sub_intents": [
            "new_policy",
            "cancel_policy",
            "beneficiary_info",
        ],
    },
    {
        "main_intent": "customer_retention",
        "sub_intents": [
            "complaint",
            "escalation",
            "free_product_upgrade",
        ],
    },
    {
        "main_intent": "technical_support",
        "sub_intents": [
            "portal_navigation",
            "login_issues",
        ],
    },
]

FT_PROMPT = """Identify the intent classes from the given user query, delimited with ####. Intents are categorized into two levels: main intent and sub intent. In your response, provide only ONE set of main and sub intents that is most relevant to the query. Write your response ONLY in this format <main-intent>:<sub-intent>. ONLY Write the intention.

OUTPUT EXAMPLE:
profile_update:contact_info

OUTPUT EXAMPLE:
technical_support:portal_navigation

#### QUERY:
{query}
####
"""


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


def get_or_create_endpoint(endpoint_name, endpoint_config_name=None):
    """
    Get or create an endpoint with the given name.

    This function first checks if an endpoint with the specified `endpoint_name`
    exists. If it does, it returns the existing endpoint. If the endpoint does
    not exist, it then checks for an existing endpoint configuration.
    If an endpoint configuration with the given name exists, it creates a new
    endpoint with this configuration and returns it.
    """
    # Check if the endpoint already exists
    endpoints_list = sagemaker_client.list_endpoints(
        NameContains=endpoint_name, MaxResults=100
    )
    endpoints = [
        ep
        for ep in endpoints_list["Endpoints"]
        if ep["EndpointName"] == endpoint_name
    ]
    if len(endpoints) > 0:
        print("Endpoint already exists. Using it...")
        return endpoints[0]

    # If endpoint does not exist, check if the endpoint configuration exists
    if not endpoint_config_name:
        endpoint_config_name = endpoint_name

    endpoint_configs_list = sagemaker_client.list_endpoint_configs(
        NameContains=endpoint_config_name, MaxResults=100
    )

    endpoint_configs = [
        ep
        for ep in endpoint_configs_list["EndpointConfigs"]
        if ep["EndpointConfigName"] == endpoint_config_name
    ]
    if len(endpoint_configs) > 0:
        print("Endpoint configuration already exists. Creating endpoint...")
        sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
        )
        sagemaker_client.get_waiter("endpoint_in_service").wait(
            EndpointName=endpoint_name
        )
        return endpoint_configs[0]

    return None


def get_predictor(
    endpoint_name,
    model_id=None,
    model_version=None,
    inference_instance_type=None,
    endpoint_config_name=None,
    region=None,
    **kwargs,
):
    """
    Get or create a predictor for the given endpoint name.
    If the endpoint exists, it returns a predictor for the endpoint.
    if the endpoint does not exist, it deploys a new endpoint with the given
    model_id and model_version, and returns a predictor for the new endpoint.
    """

    res = get_or_create_endpoint(endpoint_name, endpoint_config_name)
    if res:
        predictor = Predictor(
            endpoint_name=endpoint_name,
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer(),
        )
        return predictor

    # If there is no endpoint or endpoint configuration, create new ones
    print("Creating endpoint configuration and deploying endpoint...")
    model = JumpStartModel(
        region=region,
        role=get_role_arn(),
        model_id=model_id,
        model_version=model_version,
    )
    predictor = model.deploy(
        endpoint_name=endpoint_name,
        instance_type=inference_instance_type,
    )
    return predictor


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


def llama2_chat_output_intent_formatter(response):
    res = llama2_parse_output(response)
    res = res.split("\n", 1)[0].strip()

    intents_list = res.split(":")
    return intents_list


def mistral(predictor, user, temperature=0.1, max_tokens=64, top_p=0.8):
    """
    Constructs the payload for the mistral model, sends it to the endpoint,
    and returns the response.
    """
    parameters = {
        "max_new_tokens": max_tokens,
        "top_p": top_p,
        "do_sample": True,
        "temperature": temperature,
    }
    payload = {"inputs": user, "parameters": parameters}

    response = predictor.predict(payload)
    return response


def mistral_output_intent_formatter(response):
    res = parse_output(response).strip()

    # Remove unwanted characters at the beginning and end of the response
    res = re.sub(r"^[#\s]+|[#\s]+$", "", res)
    res = re.split(r"[#\n `\'();{}]", res, maxsplit=1)[0].strip()

    intents_list = res.split(":")

    return intents_list


def flant5(predictor, user, temperature=0.1, max_tokens=64, top_p=0.9):
    """
    Constructs the payload for the flant5 model, sends it to the endpoint,
    and returns the response.
    """
    payload = {
        "inputs": user,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "num_return_sequences": 3,
            "top_p": top_p,
            "do_sample": True,
        },
    }

    response = predictor.predict(payload)
    return response


def flant5_output_intent_formatter(response):
    res = parse_output(response)
    intents_list = res.split(":")
    return intents_list


def parse_output(response):
    if len(response) > 0:
        response = response[0]

    return response["generated_text"].strip()


def load_dataset(path):
    """load jsonl file into a list"""
    with open(path, encoding="utf-8") as file:
        lines = file.readlines()
    return [json.loads(line) for line in lines]


def write_dict_to_jsonl_file(json_list, file_path, overwrite=False):
    if overwrite:
        mode = "w"
    else:
        mode = "a"

    with open(file_path, mode, encoding="utf-8") as file:
        for item in json_list:
            file.write(json.dumps(item) + "\n")


def prepare_data_for_finetuning(dataset):
    output = []

    completion = "{main_intent}:{sub_intent} \n\n"
    for line in dataset:
        output.append(
            {
                "query": line["text"],
                "response": completion.format(
                    main_intent=line["main_intent"],
                    sub_intent=line["sub_intent"],
                ),
            }
        )

    return output


def upload_train_and_template_to_s3(bucket_prefix, train_path, template_path):
    output_bucket = Session().default_bucket()
    destination = f"s3://{output_bucket}/{bucket_prefix}"
    S3Uploader.upload(train_path, destination)
    S3Uploader.upload(template_path, destination)
    return destination


def evaluate_model(
    predictor,
    dataset,
    prompt_template,
    llm,
    response_formatter,
    max_tokens=15,
    incorrect_responses_log_file=incorrect_responses_log_file,
    error_responses_log_file=error_responses_log_file,
    system_message=None,
):
    """
    Evaluates a model's performance on a given dataset.

    This function iterates over each item in the dataset, generates a response
    from the model using the provided prompt template, and compares the model's
    response intents with the actual intents in the dataset. It tracks the number
    of correct and erroneous responses, and logs incorrect and error responses if
    respective log files are provided. The function also updates and returns a
    distribution of responses by main and sub intents.

    Parameters:
    - predictor: The model's prediction function.
    - dataset: A list of dictionaries containing the dataset for evaluation.
    - prompt_template: A string template for generating prompts from the dataset.
    - llm: A function to interact with the language model.
    - response_formatter: A function to format the model's response.
    - max_tokens (int): The maximum number of tokens for the model's response.
    - incorrect_responses_log_file: File path for logging incorrect responses.
    - error_responses_log_file: File path for logging error responses.
    - system_message: An optional system message to be passed to the model.

    Returns:
    - dict: A dictionary containing evaluation metrics such as the number of
      correct, incorrect, and error responses, accuracy, and a distribution
      DataFrame.

    The function updates the evaluation distribution and logs files for incorrect
    and error responses based on the model's performance.
    """
    correct = 0
    error = 0
    distribution = []

    for i, line in enumerate(dataset):
        print(
            f"\r{i+1}/{len(dataset)} - corrects: {correct} - errors: {error}",
            end="",
        )
        update_eval_distribution(
            distribution=distribution,
            main_intent=line["main_intent"],
            sub_intent=line["sub_intent"],
            increase_counter=True,
        )

        prompt = prompt_template.format(query=line["text"])
        response = ""
        try:
            response = llm(
                predictor=predictor,
                user=prompt,
                max_tokens=max_tokens,
                system=system_message,
            )
            intents_list = response_formatter(response)
            main_intent = intents_list[0].strip()
            sub_intent = intents_list[1].strip()
        except Exception as ex:
            error += 1
            if error_responses_log_file:
                error_response = {
                    "text": line["text"],
                    "response": response,
                    "error": str(ex),
                }
                write_dict_to_jsonl_file(
                    [error_response], error_responses_log_file
                )
            continue

        if (
            main_intent == line["main_intent"].strip()
            and sub_intent == line["sub_intent"].strip()
        ):
            correct += 1
            update_eval_distribution(
                distribution=distribution,
                main_intent=main_intent,
                sub_intent=sub_intent,
                correct=True,
            )
        else:
            if incorrect_responses_log_file:
                incorrect_inference = {
                    "text": line["text"],
                    "main_intent": line["main_intent"],
                    "sub_intent": line["sub_intent"],
                    "response_main_intent": main_intent,
                    "response_sub_intent": sub_intent,
                }
                write_dict_to_jsonl_file(
                    [incorrect_inference], incorrect_responses_log_file
                )
    print()
    distribution = pd.DataFrame(distribution)
    distribution["incorrect"] = distribution["count"] - distribution["correct"]
    distribution["accuracy"] = (
        distribution["correct"] * 100 / distribution["count"]
    )

    return {
        "correct": correct,
        "incorrect": len(dataset) - correct,
        "error": error,
        "accuracy": correct * 100 / len(dataset),
        "distribution": distribution,
    }


def update_eval_distribution(
    distribution,
    main_intent,
    sub_intent,
    correct=False,
    increase_counter=False,
):
    intent = [
        intent
        for intent in distribution
        if intent["main_intent"] == main_intent
        and intent["sub_intent"] == sub_intent
    ]
    if len(intent) > 0:
        if increase_counter:
            intent[0]["count"] += 1
        if correct:
            intent[0]["correct"] += 1

    else:
        distribution.append(
            {
                "main_intent": main_intent,
                "sub_intent": sub_intent,
                "count": 1,
                "correct": 0,
            }
        )


def print_eval_result(response, dataset):
    print(
        "Total size:",
        len(dataset),
        "accuracy:",
        response["accuracy"],
        "correct:",
        response["correct"],
        "incorrect:",
        response["incorrect"],
        "error:",
        response["error"],
    )
    print("\nDistribution:")
    print(response["distribution"].to_string(line_width=100))
