import streamlit as st
from google.auth import default
from google.api_core.client_options import ClientOptions
import vertexai
from vertexai.generative_models import GenerativeModel, Part

# Get the API key from Streamlit secrets
api_key = st.secrets["gcp_api_key"]

# Create client options with the API key
client_options = ClientOptions(api_endpoint="us-central1-aiplatform.googleapis.com",
                               credentials=default()[0],  # Get Application Default Credentials
                               api_key=api_key)

# Initialize Vertex AI (replace with your project ID)
project_id = "test-document-ai-api-411410"  # Replace with your actual project ID
vertexai.init(project=project_id, location="us-central1", client_options=client_options)

# Load the Gemini 1.5 Pro model
model = GenerativeModel("gemini-1.5-pro-preview-0409")

# Function to process text input
def process_text(prompt):
    response = model.generate_content([prompt])
    st.write(response.text)

# Function to process file input (image, video, PDF)
def process_file(file_uploader, mime_type):
    if file_uploader is not None:
        file_content = Part.from_file_uploader(file_uploader, mime_type)
        prompt = st.text_input("Enter your prompt:")
        if prompt:
            contents = [file_content, prompt]
            response = model.generate_content(contents)
            st.write(response.text)

# Streamlit App
st.title("Gemini 1.5 Pro Interaction")

# Text input section
st.header("Text Input")
text_prompt = st.text_area("Enter your text prompt:", height=100)
if st.button("Submit Text"):
    process_text(text_prompt)

# File input sections
st.header("File Input")

# Image input
st.subheader("Image")
image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
process_file(image_file, "image/jpeg")

# Video input
st.subheader("Video")
video_file = st.file_uploader("Upload a video", type=["mp4"])
process_file(video_file, "video/mp4")

# PDF input
st.subheader("PDF")
pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])
process_file(pdf_file, "application/pdf")
