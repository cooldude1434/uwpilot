import os
from datetime import datetime

import streamlit as st
import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)

PROJECT_ID = os.environ.get("GCP_PROJECT")  # Your Google Cloud Project ID
LOCATION = os.environ.get("GCP_REGION")  # Your Google Cloud Project Region
vertexai.init(project=PROJECT_ID, location=LOCATION)


@st.cache_resource
def load_models():
    text_model_pro = GenerativeModel("gemini-1.5-pro")
    multimodal_model_pro = GenerativeModel("gemini-1.0-pro-vision")
    return text_model_pro, multimodal_model_pro


def get_gemini_pro_text_response(
    model: GenerativeModel,
    contents: str,
    generation_config: GenerationConfig,
    stream: bool = True,
):
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }

    responses = model.generate_content(
        contents,
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=stream,
    )

    final_response = []
    for response in responses:
        try:
            final_response.append(response.text)
        except IndexError:
            final_response.append("")
            continue
    return " ".join(final_response)


# def get_gemini_pro_vision_response(
#     model, prompt_list, generation_config={}, stream: bool = True
# ):
#     generation_config = {"temperature": 0.1, "max_output_tokens": 2048}
#     responses = model.generate_content(
#         prompt_list, generation_config=generation_config, stream=stream
#     )
#     final_response = []
#     for response in responses:
#         try:
#             final_response.append(response.text)
#         except IndexError:
#             pass
#     return "".join(final_response)


# Set page configuration
st.set_page_config(
    page_title="Gemini 1.5 LLM based Copilot",
    page_icon=":house:",
    # layout="wide",
    
    # initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #E9F1FA;
        padding: 20px;
    }
    .report-box {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
    }
    .header-title {
        color: #1D1D1;
        font-weight: bold;
        text-align: center;
    }
    .section-title {
        color: #333333;
        font-size: 18px;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .stButton>button {
        background-color: #00ABE4;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='header-title'>Gemini 1.5 LLM based Copilot</h1>", unsafe_allow_html=True)

text_model_pro, multimodal_model_pro = load_models()


# st.subheader("Underwriter Copilot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def send_message():
    user_message = st.session_state.user_input
    st.session_state.chat_history.append(f"You: {user_message}")
    config = {"temperature": 0.1, "max_output_tokens": 8192}

    with st.spinner("Thinking..."):
        response = get_gemini_pro_text_response(
            text_model_pro,
            user_message,
            generation_config=config,
        )
        st.session_state.chat_history.append(f"Gemini Pro: {response}")
        st.session_state.user_input = ""

st.text_area("Chat history:", value="\n".join(st.session_state.chat_history), height=300, key="chat_history_display", disabled=True)
st.text_input("Your message:", key="user_input", on_change=send_message)
st.button("Send", on_click=send_message)
