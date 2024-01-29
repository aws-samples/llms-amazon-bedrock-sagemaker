# Knowledge-Powered Question Answering with Amazon SageMaker and LlamaIndex

## Overview
This code supports a blog post [Building Knowledge-Powered Question Answering Applications using Amazon SageMaker and LlamaIndex](link) which demonstrates building question and answering applications with the ability to query user-defined data sources. The application is supported by endpoints deployed in Amazon SageMaker Jumpstart. We will use a text embedding model from Hugging Face and the llama chat LLM from Meta. The data sources in are all in .pdf form, and can be ingested and vectorised using LlamaIndex. This makes querying the knowledge base more efficient and accurate. 

## Prerequisites
For this tutorial you'll need an AWS account with Amazon SageMaker Domains and appropriate IAM permissions. Instructions on launching the required content can be found [here](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-set-up.html).If you don’t already have a SageMaker domain, refer to Onboard to Amazon SageMaker Domain to create one. In this post we are using the [‘AmazonSageMakerFullAccess’](https://docs.aws.amazon.com/aws-managed-policy/latest/reference/AmazonSageMakerFullAccess.html) and we assume that you use administrator user credentials for the exercises and procedures in this guide. If you choose to create and use another user, you must grant that user **minimum permissions**. You can also explore how you can use [Amazon SageMaker Role Manager](https://docs.aws.amazon.com/sagemaker/latest/dg/role-manager.html) to build and manage persona-based IAM roles for common machine learning needs directly through the Amazon SageMaker console. 




We use the `us-east-1` region to build our application.

You will need access to `ml.g5.2xlarge` and `ml.g5.48xlarge` instances for endpoint usage when deploying the text embeddings and chat generation models. [Quota increases](https://docs.aws.amazon.com/servicequotas/latest/userguide/request-quota-increase.html) can be requested via the console.

Additionally, the code in this repository is based on **langchain** `version 0.1.0` and **llama_index** `version 0.9.29`.

## Steps

### 1. Deploy Hugging Face embeddings model with SageMaker Jumpstart
Create and deploy an endpoint for the Hugging Face GPT-J text embeddings model so that we can create text embeddings for natural language questions and answers. 

### 2. Use Meta Llama chat LLM with SageMaker Jumpstart
Create and deploy an endpoint for the Llama 70B Chat model model so that we can create a chat experience using an LLM for the user to interact with when querying the application.

### 3. Use LlamaIndex to ingest .pdfs and build an index
Create a knowledge base using the folder of pdfs `pressrelease` so that the application has a knowledge base from which it can form repsonses to user queries.

### 4. Build an agent using LangChain
Use the LangChain framework to build an agent that can access tools e.g. to query a press release during it's chain of thought process.

### 5. Clean Up
Delete the real-time endpoints and avoid excess costs when the endpoints are not in use.

## Contribute
If you would like to contribute to the project, see [CONTRIBUTING](https://github.com/pnipinto/llms-amazon-bedrock-sagemaker/blob/main/CONTRIBUTING.md#security-issue-notifications) for more information.

## License
The license for this repository depends on the section.  Data set for the course is being provided to you by permission of Amazon and is subject to the terms of the [Amazon License and Access](https://www.amazon.com/gp/help/customer/display.html?nodeId=201909000). You are expressly prohibited from copying, modifying, selling, exporting or using this data set in any way other than for the purpose of completing this course. The lecture slides are released under the CC-BY-SA-4.0 License.  The code examples are released under the MIT-0 License. See each section's LICENSE file for details.
