import streamlit as st
from transformers import pipeline
import torch
import time
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Set the page config at the top
st.set_page_config(page_title="Audio-to-Text Transcription",
                   layout="centered", initial_sidebar_state="auto")

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info("Using device: %s", device)

# Initialize the whisper model pipeline
# When a function is decorated with @st.cache_resource,
# Streamlit will cache the output of that function the first time it's called.
# In subsequent calls, instead of re-running the function, it will return the cached result.
# This avoids the overhead of loading the model multiple times, speeding up the app.


@st.cache_resource
def load_model():
    logger.info("Loading Whisper model (openai/whisper-small) on %s", device)
    model = pipeline("automatic-speech-recognition",
                     "openai/whisper-small",
                     device=device)
    logger.info("Model loaded successfully")
    return model


pipe = load_model()

def get_audio_duration(file_path):
    import subprocess
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", file_path],
        capture_output=True, text=True
    )
    return float(result.stdout.strip())


def split_audio_chunks(file_path, chunk_seconds=300):
    """Split audio into chunks of chunk_seconds. Returns list of (chunk_path, offset_s)."""
    duration = get_audio_duration(file_path)
    chunks = []
    base, ext = os.path.splitext(file_path)
    start = 0
    index = 0
    while start < duration:
        chunk_path = f"{base}_chunk{index}{ext}"
        os.system(
            f'ffmpeg -y -ss {start} -t {chunk_seconds} -i "{file_path}" -c copy "{chunk_path}" -loglevel error'
        )
        chunks.append((chunk_path, start))
        start += chunk_seconds
        index += 1
    return chunks


# Define the app layout


def main():
    st.markdown("<h1 style='color: #00bfff;'>Audio-to-Text Transcription App</h1>",
                unsafe_allow_html=True)
    st.markdown("<p style='color: #000000;'>Generate transcription with timestamps and download the result.</p>",
                unsafe_allow_html=True)

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload an audio file", type=["mp3", "wav", "ogg", "m4a"])
    st.audio(uploaded_file)

    # Select language and task
    languages = {'English': 'en', 'Spanish': 'es'}  # Choose the source language
    task = 'transcribe'  # when you chose translate -> it means translation to english

    language_label = st.selectbox(
        "Choose the language of the audio", options=list(languages.keys()))
    language = languages[language_label]
    st.write("**When you choose 'translate', it translates the audio to English**.")
    # task = st.selectbox("Choose the task", options=tasks)

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
                logger.info("Saved uploaded file to %s", temp_file_path)

                # Convert m4a to wav if needed
                if uploaded_file.name.lower().endswith(".m4a"):
                    wav_path = temp_file_path.replace(".m4a", ".wav")
                    logger.info("Converting m4a to wav: %s -> %s", temp_file_path, wav_path)
                    os.system(f'ffmpeg -y -i "{temp_file_path}" "{wav_path}"')
                    os.remove(temp_file_path)
                    temp_file_path = wav_path

                # Split into 5-minute chunks and transcribe
                chunks = split_audio_chunks(temp_file_path, chunk_seconds=300)
                logger.info("Split audio into %d chunk(s)", len(chunks))

                all_text = ""
                progress_bar = st.progress(0, text="Transcribing chunk 1...")
                status_area = st.empty()

                for i, (chunk_path, offset_s) in enumerate(chunks):
                    status_area.info(f"Transcribing chunk {i+1} of {len(chunks)} (offset {int(offset_s//60)}m{int(offset_s%60)}s)...")
                    logger.info("Transcribing chunk %d/%d: %s (offset=%ss)", i+1, len(chunks), chunk_path, offset_s)
                    result = pipe(chunk_path, return_timestamps=True, generate_kwargs={"language": language, "task": task})
                    all_text += format_transcription(result) + "\n"
                    os.remove(chunk_path)
                    progress_bar.progress((i + 1) / len(chunks), text=f"Chunk {i+1}/{len(chunks)} done")

                formatted_transcription = all_text.strip()
                status_area.empty()

                elapsed = round(time.time() - start_time, 2)
                logger.info("Transcription completed in %ss (%d chars)", elapsed, len(formatted_transcription))

                st.success(f"{task} completed!")
                st.text_area(f"{task} Output",
                             value=formatted_transcription, height=500)

                # Download transcription option
                st.download_button(
                    "Download Transcription", formatted_transcription, file_name="transcription.txt")

                end_time = time.time()
                st.write(
                    f"Time taken: {round(end_time - start_time, 2)} seconds")

                # Clean up the temporary file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                logger.info("Cleaned up temp file: %s", temp_file_path)

# Helper function to format the transcription with timestamps


def format_transcription(transcription):
    if 'chunks' in transcription and transcription['chunks']:
        return "\n".join(line["text"] for line in transcription['chunks']).strip()
    return transcription.get('text', '').strip()


if __name__ == "__main__":
    main()
