import streamlit as st
from google.oauth2 import service_account
import vertexai
from vertexai.generative_models import GenerativeModel, Part
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

# Authenticate with Vertex AI
credentials = service_account.Credentials.from_service_account_info(service_account_info)
vertexai.init(project=service_account_info["project_id"], location="us-central1", credentials=credentials)

# Load the Gemini model
model = GenerativeModel("gemini-1.5-pro-preview-0409")  # Replace with your model name

# System prompt and generation configurations
system_prompt = Part.from_text("You are a helpful and informative AI assistant.")
generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}
safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

# Streamlit app title
st.title("Chat with Gemini and Analyze Files")

# Initialize chat and conversation history
if "gemini_chat" not in st.session_state:
    st.session_state.gemini_chat = model.start_chat()
if "gemini_messages" not in st.session_state:
    st.session_state.gemini_messages = []
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# File uploader
uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png", "mp4", "pdf", "mp3", "wav"])

# Chat interface
user_input = st.text_input("What would you like to ask?")

# Process user input and file upload
if user_input or uploaded_file:
    # Add user input to conversation history
    st.session_state.gemini_messages.append({"role": "user", "content": user_input})
    st.session_state.conversation_history.append(f"user: {user_input}")

    # Handle file upload
    file_parts = []
    if uploaded_file:
        mime_type = None
        if uploaded_file.name.lower().endswith((".jpg", ".jpeg", ".png")):
            mime_type = "image/jpeg"
        elif uploaded_file.name.lower().endswith(".mp4"):
            mime_type = "video/mp4"
        # ... (Add more MIME types as needed)
        if mime_type is None:
            st.error("Unsupported file type")
        else:
            file_parts.append(Part.from_data(mime_type=mime_type, data=uploaded_file.read()))

    # Generate response from Gemini
    with st.spinner("Waiting for the assistant to respond..."):
        response = st.session_state.gemini_chat.send_message(
            [system_prompt, *file_parts, Part.from_text(user_input)],
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        # Display response
        if isinstance(response, str):
            st.error(response)
        else:
            response_text = response.text
            st.session_state.gemini_messages.append({"role": "assistant", "content": response_text})
            st.session_state.conversation_history.append(f"assistant: {response_text}")

# Display the conversation history
for message in st.session_state.gemini_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
