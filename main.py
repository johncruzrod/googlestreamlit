import streamlit as st
from google.oauth2 import service_account
import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models

# Load the service account credentials from Streamlit secrets
service_account_info = {
    "type": st.secrets["gcp"]["type"],
    "project_id": st.secrets["gcp"]["project_id"],
    "private_key_id": st.secrets["gcp"]["private_key_id"],
    "private_key": st.secrets["gcp"]["private_key"],
    "client_email": st.secrets["gcp"]["client_email"],
    "client_id": st.secrets["gcp"]["client_id"],
    "auth_uri": st.secrets["gcp"]["auth_uri"],
    "token_uri": st.secrets["gcp"]["token_uri"],
    "auth_provider_x509_cert_url": st.secrets["gcp"]["auth_provider_x509_cert_url"],
    "client_x509_cert_url": st.secrets["gcp"]["client_x509_cert_url"]
}

# Create credentials object from the service account info
credentials = service_account.Credentials.from_service_account_info(service_account_info) 

# Set up Streamlit app
st.title("Vertex AI Multimodal Generation with Gemini 1.5 Pro")

# File uploader
uploaded_file = st.file_uploader("Upload your file", type=["png", "jpg", "jpeg", "mp3", "wav", "mp4", "pdf"])

# Text prompt input
text_input = st.text_area("Enter your text prompt:")

# Generate button
if st.button("Generate"):
    if uploaded_file and text_input:
        try:
            # Initialize the Vertex AI SDK with the credentials
            vertexai.init(project=service_account_info["project_id"], location="us-central1", credentials=credentials)

            # Load the model
            model = GenerativeModel("gemini-1.5-pro-preview-0409")

            # Determine file type and create Part object
          file_type = uploaded_file.type
          if file_type.startswith("image"):
            part = Part.create_from_file(uploaded_file, mime_type=file_type)  # Use create_from_file
          elif file_type.startswith("audio"):
            part = Part.create_from_file(uploaded_file, mime_type=file_type)  # Use create_from_file
          elif file_type.startswith("video"):
            part = Part.create_from_file(uploaded_file, mime_type=file_type)  # Use create_from_file 
          elif file_type == "application/pdf":
            part = Part.create_from_file(uploaded_file, mime_type=file_type)  # Use create_from_file
          else:
            st.error("Unsupported file type. Please upload an image, audio, video, or PDF file.")
            
            # Generate content
            responses = []
            for response in model.generate_content(
                [part, Part.from_text(text_input)],
                generation_config={"max_output_tokens": 8192, "temperature": 1, "top_p": 0.95},
                safety_settings={
                    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                },
                stream=True,
            ):
                responses.append(response.text)

            # Display generated text
            st.success("Generated Text:")
            st.write("".join(responses))

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please upload a file and provide a text prompt.")
