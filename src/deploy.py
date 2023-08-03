# Deploy (Create model, endpoint config, and endpoint)

import os, sys
import boto3
from sagemaker import image_uris

# Note: Remember to create these env vars in GitHub: 
# AWS_ACCESS_KEY_ID
# AWS_SECRET_ACCESS_KEY
# AWS_SESSION_TOKEN
# AWS_ROLE
# AWS_REGION

n = len(sys.argv)
if n>1:
    model_version = sys.argv[1]
    stage = sys.argv[2]
    model_url = sys.argv[3]
else:
    print("Insufficient arguments provided. Pls provide model version, stage and model url.")
    exit()

session = boto3.Session()
aws_region=os.environ["AWS_REGION"]
sagemaker_client = session.client('sagemaker', region_name=aws_region)
sagemaker_role= os.environ["AWS_ROLE"]

# TODO: Create a custom container based on the model instead of this hard-coded one
framework='tensorflow'
framework_version = '2.9'
instance_type = 'ml.t2.medium'
container = image_uris.retrieve(region=aws_region, framework=framework, version=framework_version, image_scope='inference', instance_type=instance_type)

name_without_dots = model_version.replace('.', '-')

model_name = f'td-fm-model-{name_without_dots}'
try:
    create_model_response = sagemaker_client.create_model(
        ModelName = model_name,
        ExecutionRoleArn = sagemaker_role,
        PrimaryContainer = {
            'Image': container,
            'ModelDataUrl': model_url,
        })
except Exception as e:
    print("\nEncountered the following error when creating the model: ", e)
    if 'Cannot create already existing model' in str(e):
        print("\nBecause this model already exists, we will reuse it instead of creating a new one.")

endpoint_config_name = f'td-fm-ep-config-{name_without_dots}-{stage}'
try:
    endpoint_config_response = sagemaker_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "variant1",
                "ModelName": model_name, 
                # "InstanceType": instance_type,
                # "InitialInstanceCount": 1,
                "ServerlessConfig": {
                    "MemorySizeInMB": 1024,
                    "MaxConcurrency": 10
                }
            }
        ]
    )
except Exception as e:
    print("\nEncountered the following error when creating the endpoint config: ", e)
    

endpoint_name = f'td-fm-ep-{stage}'
try:
    create_endpoint_response = sagemaker_client.create_endpoint(
                                                EndpointName=endpoint_name, 
                                                EndpointConfigName=endpoint_config_name) 
except Exception as e:
    print("\nEncountered the following error when creating the endpoint: ", e)
    if 'Cannot create already existing endpoint' in str(e):
        print("\nBecause this endpoint already exists, we will update it to use the new model version.")
        create_endpoint_response = sagemaker_client.update_endpoint(
                                                    EndpointName=endpoint_name, 
                                                    EndpointConfigName=endpoint_config_name)