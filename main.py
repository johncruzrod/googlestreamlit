import streamlit as st
from google.oauth2 import service_account
from vertexai.generative_models import GenerativeModel

# Load the service account credentials from Streamlit secrets
service_account_info = {
    "type": st.secrets["gcp_service_account"]["type"],
    "project_id": st.secrets["gcp_service_account"]["project_id"],
    "private_key_id": st.secrets["gcp_service_account"]["private_key_id"],
    "private_key": st.secrets["gcp_service_account"]["private_key"],
    "client_email": st.secrets["gcp_service_account"]["client_email"],
    "client_id": st.secrets["gcp_service_account"]["client_id"],
    "auth_uri": st.secrets["gcp_service_account"]["auth_uri"],
    "token_uri": st.secrets["gcp_service_account"]["token_uri"],
    "auth_provider_x509_cert_url": st.secrets["gcp_service_account"]["auth_provider_x509_cert_url"],
    "client_x509_cert_url": st.secrets["gcp_service_account"]["client_x509_cert_url"]
}

# Create credentials object from the service account info
credentials = service_account.Credentials.from_service_account_info(service_account_info)

# Set up the Streamlit app
st.title("Vertex AI Text Generation")

# Get the user input
text_input = st.text_input("Enter your text:")

# Generate text when the user clicks the button
if st.button("Generate Text"):
    if text_input:
        try:
            # Initialize the Vertex AI client with the credentials
            GenerativeModel.init(project=service_account_info["project_id"], credentials=credentials)

            # Load the model
            model = GenerativeModel("gemini-1.5-pro")

            # Query the model with the user input
            response = model.predict(text_input)

            # Display the generated text
            st.success("Generated Text:")
            st.write(response.text)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter some text.")
