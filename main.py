import streamlit as st
import vertexai
from vertexai.generative_models import GenerativeModel
import vertexai.preview.generative_models as generative_models
from google.oauth2 import service_account
import google.auth.transport.requests

def get_credentials():
    # Retrieve secrets from Streamlit's secret storage
    info = st.secrets["gcp"]
    credentials = service_account.Credentials.from_service_account_info(info)
    return credentials

def multiturn_generate_content(user_input):
    credentials = get_credentials()
    # Use the credentials to authenticate the Vertex AI session
    vertexai.init(project=st.secrets["gcp"]["project_id"], location="us-central1", credentials=credentials)
    model = GenerativeModel("gemini-1.5-pro-preview-0409")
    chat = model.start_chat()

    # Assuming there's a way to send input and receive output, adjust according to actual API
    chat.send(user_input)
    response = chat.receive()

    return response

st.title('Vertex AI API Tester with Streamlit')
user_input = st.text_input("Enter your text prompt:", "Hello, world!")
if st.button('Send Prompt'):
    response = multiturn_generate_content(user_input)
    st.write(response)
