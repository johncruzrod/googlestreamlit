import streamlit as st
import requests
import json

# Load secrets
project_id = st.secrets["project_id"]
api_key = st.secrets["api_key"]

# API setup
endpoint = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/publishers/google/models/gemini-1.0-pro:streamGenerateContent"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

st.title('Simple Streamlit Gemini API Tester')

# User input
user_input = st.text_input("Enter your text prompt:", "Hello, world!")

if st.button('Send Prompt'):
    # Construct the request body
    data = {
        "contents": {
            "role": "user",
            "parts": [{"text": user_input}]
        },
        "generation_config": {
            "temperature": 0.2,
            "topP": 0.8,
            "topK": 40
        }
    }
    
    # Send POST request to the API
    response = requests.post(endpoint, headers=headers, data=json.dumps(data))
    
    if response.status_code == 200:
        # Parse the response
        result = response.json()
        st.write("Response from API:")
        st.json(result)
    else:
        st.error(f"Failed to retrieve data: {response.status_code}")

