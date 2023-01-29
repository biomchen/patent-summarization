#!/usr/bin/env python3

# see https://huggingface.co/google/pegasus-big_patent
# for more details
# update: model card has been removed.
# It currently has a fine-tunned model with JAX/Flax
# https://github.com/google-research/pegasus/tree/main/pegasus/flax
# The codes here haven't been updated.

import requests
from constant import huggin_api_key

API_URL = "https://api-inference.huggingface.co/models/google/pegasus-big_patent"
api_key_string = f"Bearer {huggin_api_key}"
headers = {"Authorization": api_key_string}

def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

if __name__ == '__main__':
    text =  input("Put the text here: ")
    results = query({
        "inputs" : text,
    })
    if 'error' in results:
        print(results['error'])
    else:
        print(results[0]['generated_text'])