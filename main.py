import streamlit as st
from transformers import pipeline
import torch
import time
import os

# Set the page config at the top
st.set_page_config(page_title="Audio-to-Text Transcription", layout="centered", initial_sidebar_state="auto")

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize the whisper model pipeline
# When a function is decorated with @st.cache_resource, 
# Streamlit will cache the output of that function the first time it's called.
# In subsequent calls, instead of re-running the function, it will return the cached result. 
# This avoids the overhead of loading the model multiple times, speeding up the app.
@st.cache_resource
def load_model():
    return pipeline("automatic-speech-recognition", 
                    "openai/whisper-small", 
                    chunk_length_s=30, 
                    stride_length_s=5, 
                    return_timestamps=True, 
                    device=device)

pipe = load_model()

# Define the app layout
def main():
    st.markdown("<h1 style='color: #00bfff;'>Audio-to-Text Transcription App</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #000000;'>Generate transcription with timestamps and download the result.</p>", unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "ogg"])
    st.audio(uploaded_file)
    
    # Select language and task
    languages = ['English', 'Hindi', 'French'] # Choose the source language
    tasks = ['transcribe', 'translate'] # when you chose translate -> it means translation to english

    language = st.selectbox("Choose the language of the audio", options=languages)
    st.write("**When you choose 'translate', it translates the audio to English**.")
    task = st.selectbox("Choose the task", options=tasks)

    # Transcribe button
    if uploaded_file is not None:
        if st.button(f"{task}"):
            with st.spinner("Transcribing/Translating..."):
                start_time = time.time()

                # Create temporary file path
                temp_dir = "temp_dir"
                os.makedirs(temp_dir, exist_ok=True)
                temp_file_path = os.path.join(temp_dir, uploaded_file.name)

                # Save the uploaded file temporarily
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())


                # Transcribe audio
                transcription = pipe(temp_file_path, generate_kwargs={"language": language, "task": task})
                formatted_transcription = format_transcription(transcription)
                
                st.success(f"{task} completed!")
                st.text_area(f"{task} Output", value=formatted_transcription, height=500)
                
                # Download transcription option
                st.download_button("Download Transcription", formatted_transcription, file_name="transcription.txt")
                
                end_time = time.time()
                st.write(f"Time taken: {round(end_time - start_time, 2)} seconds")
                
                # Clean up the temporary file
                os.remove(temp_file_path)

# Helper function to format the transcription with timestamps
def format_transcription(transcription):
    formatted_text = ""
    for line in transcription['chunks']:
        text = line["text"]
        ts = line["timestamp"]
        formatted_text += f"[{ts[0]}:{ts[1]}] {text}\n"
    return formatted_text.strip()

if __name__ == "__main__":
    main()