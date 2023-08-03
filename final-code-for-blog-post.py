############ Final result ############




############ What do we need ############
# AWS credentials and SageMaker role
# S3 buckets for training data and model output
# ECR repository for the custom model container. But I'm using one of the provided containers, so I haven't set this up yet.


# Deploying a model in SageMaker is a three-step process:
# Create a SageMaker model
# Create an endpoint configuration
# Create an endpoint


# Notes: 
# The Amazon S3 bucket where the model artifacts are stored must be in the same region as the model that you are creating.

############ Chapter 1: Training ############

## First, you train your model and evaluate its performance
## Once you have a reasonable baseline, and want to tune hyperparameters, create a dvc pipeline
## If you training jobs are long running, at this point you will also want to use DVCLive to track your experiments in real-time
## Every time you see significant improvements in model performance, it's a good idea to register new versions for the model in the Studio model registry

# DONE: Training code, DVCLive, Git, DVC stages

######### Chapter 2: Deployment #########

## When you have a decent model, deploy it
## You can deploy using the AWS console or AWS cli or Python SDK or whatever method you prefer
## But if your model is expected to evolve constantly, it is a hassle to manually redeploy it each time. So, automate this process
## For automation, write a deployment script and create a CI action that runs this script when you need to redeploy
## You can trigger the CI action easily by assigning appropriate stages to the desired model versions in the Studio model registry

# TODO: Create a custom container based on the model instead of the hard-coded one that I am currently using
# TODO: Allow specifying instance type???

######### Chapter 3: Inference #########

# TODO: Move input image pre-processing to the custom container so that a raw image input can be provided for inference
# TODO: Add post-processing to return the predicted class name instead of predicted probabilities

######### Chapter 4: Inference UI #########

## TODO: Write this chapter


######### Chapter 5: Redeployment #########

# DONE: MR stages



# # If you need to get details of your endpoint for any reason, you can do that as follows
# response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
# from pprint import pprint
# pprint(response)
# # response dict contains 'EndpointName', 'EndpointArn', 'EndpointConfigName', 'ProductionVariants', 'EndpointStatus', 'CreationTime', 'LastModifiedTime', 'ResponseMetadata'


## Issues I encountered

# There are just too many docs and tutorials for SageMaker deployment, which can be overwhelming if you don't know exactly what you are looking for.
# First, I tried using MLEM for deployment, starting with our docs and tutorials (for deployment on Fly.io, Heroku, SageMaker). The issues I faced are listed in this doc: https://docs.google.com/document/d/1_5bK9wrXIKhNcpdCaZ_J9QUMhYGphzY8NEMOpPBYRcI/edit. I wasn’t able to deploy to SageMaker. I got this error`` No such command '/opt/ml/model/model.mlem'``, which I wan't able to figure out.
# I was more familiar with model training and less with deployment. And SageMaker was new to me. So, just picking up was hard.
# There are several ways to deploy models to Sagemaker, including some outdated / deprecated ones. Finding the right method took quite a while, with a lot of time time spent on debugging why a certain method did not work and finding and switching to a new method. It was tricky because with some (deprecated) methods, I could successfully deploy a model, but updating the deployment to use a new version was not possible, and I found it out only after spending quite some time learning how to deploy the model (and feeling happy that I had figured it out).
# SageMaker endpoint creation takes several minutes. If it fails, it retries once. So, the total time before you see the failure in AWS console is quite long (20 mins or more).
# When trying to create a web UI for inference (using Flask, I ran into multiple issues including REST api call authentication/signature issues). I've still not resolved them.
# At one point, I was using incorrect commits for version registration - I was using older commits in which I didn't have the required dvc.yaml structure. Since Studio does not give out any hints/warnings, I spent a lot of time trying to understand what I was doing wrong. https://iterativeai.slack.com/archives/C01116VLT7D/p1690401468138789 At one point, my GitHub actions weren't even getting triggered (and I went crazy trying to figure this out). It was error on my part (using incorrect commits), but some hints in Studio would have been helpful.
# At this point, there was also some weird issue causing version/stages to disappear from Studio and stage assignments to not work. It is possible that I messed up the repo somehow, which caused the issues. But it's not clear how. That repp is still available for debugging.
# I could not pip install dvc[s3] during deployment (https://iterativeai.slack.com/archives/CB41NAL8H/p1690784984350249). Took a lot of trials to figure out why this was happening, how to (unsuccessfully) resolve it and what installation method to finally use.
# Because the AWS Sandbox account expires every day, remembering to updates the credentials in the GH repo was sometimes an issue. And figuring out how to use those credentials can also be tricky - in my machine I was specifying the aws profile and in the GH repo, I had to pass the session token. Just understanding this also took a bit of time.
# I did not use dvc in the beginning. So, after training, I'd use boto3 to upload the artifacts to s3. After I started using dvc, I did not need that anymore. So, there was some wasted effort in making s3 upload/download/etc work in the code.
# I was expecting that there'd be a dvc command to get the remote url for a tracked file. I found it in the dvc python api. But learning that the dvc get command has an option for this wasn't easy, coz I was searching the docs for something along the lines of get url.


## What I found useful

# Chatgpt. I was struggling to figure out the end to end process from training to deployment and inference. I asked it questions along the lines of "what next?" and its answers were super helpful. It also helped when I was truing out different methods for deployment, by answering questions like "how do I do this if I'm following this method of deployment?" Docs, tutorials and Stackoverflow are of course useful, but chatgpt was like a personal tutor, answering questions with the context in mind, and remembering what I was trying to do over a period of several days. I could ask it for code suggestions, and if I didn't understand some part of the code, I could ask specifically about that. If its code suggestion didn't work, I could share the error message with it, and it would correct the code suggestion.
# GTO GH action - I've used this so I don't need to write any code to parse the Git tags myself.


## More Things-to-do

# Prepare a clean example repo for the blog post