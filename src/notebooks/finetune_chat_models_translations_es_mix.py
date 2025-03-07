# -*- coding: utf-8 -*-
# This script fine-tunes an OpenAI model for Spanish-to-Mixtec translation.

import json
import openai
import os
import pandas as pd
from pprint import pprint
import random

# Initialize OpenAI client with API key, organization, and project
client = openai.OpenAI(
    api_key=os.environ.get("KEYGPT"),  # Retrieves API key from environment variables
    organization="org-CijVCVQH1zqycayGlJnTKgJH",
    project="proj_ail4ANzX7UrB4qeFXRLgatPh",
)

# Function to upload a file for fine-tuning
def upload_file(file_name: str, purpose: str) -> str:
    with open(file_name, "rb") as file_fd:
        response = client.files.create(file=file_fd, purpose=purpose)  # Upload file
    return response.id  # Return the file ID

# Define file names for training and validation datasets
training_file_name = r"train_espanol_mixtec.jsonl"
validation_file_name = r"val_espanol_mixtec.jsonl"

# Upload the training and validation files and get their IDs
training_file_id = upload_file(training_file_name, "fine-tune")
validation_file_id = upload_file(validation_file_name, "fine-tune")

# Print the file IDs
print("Training file ID:", training_file_id)
print("Validation file ID:", validation_file_id)

# Define the model to be fine-tuned and the experiment details
MODEL = "gpt-4o-mini-2024-07-18"
PROJECT_NAME = f"Finetuning_mixtec_translator"
EXPERIMENT_NAME = f"{MODEL}-GPT_SPANISH_MIXTEC_05_05"

# Start the fine-tuning job with Weights & Biases (wandb) integration for tracking
response = client.fine_tuning.jobs.create(
    training_file=training_file_id,  # Use uploaded training file
    validation_file=validation_file_id,  # Use uploaded validation file
    model=MODEL,  # Specify the base model for fine-tuning
    suffix=EXPERIMENT_NAME,  # Name of the fine-tuned model
    integrations=[
        {
            "type": "wandb",  # Integration with Weights & Biases
            "wandb": {
                "entity": "cicata-vision1",  # W&B entity name
                "project": PROJECT_NAME,  # W&B project name
                "name": EXPERIMENT_NAME,  # W&B experiment name
            }
        }
    ]
)

# Store the fine-tuning job ID
job_id = response.id

# Print the job ID and its status
print("Job ID:", response.id)
print("Status:", response.status)

# Retrieve the job status
response = client.fine_tuning.jobs.retrieve(job_id)

# Print job details
print("Job ID:", response.id)
print("Status:", response.status)
print("Trained Tokens:", response.trained_tokens)  # Number of tokens processed

# Retrieve and display fine-tuning events in reverse order (most recent first)
response = client.fine_tuning.jobs.list_events(job_id)
events = response.data
events.reverse()

for event in events:
    print(event.message)  # Print each event message

# Retrieve final fine-tuned model ID
response = client.fine_tuning.jobs.retrieve(job_id)
fine_tuned_model_id = response.fine_tuned_model

# Check if fine-tuning is complete
if fine_tuned_model_id is None:
    raise RuntimeError(
        "Fine-tuned model ID not found. Your job has likely not been completed yet."
    )

# Print the ID of the fine-tuned model
print("Fine-tuned model ID:", fine_tuned_model_id)