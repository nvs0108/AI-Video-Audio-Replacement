import os
import requests
import streamlit as st
from moviepy.editor import VideoFileClip, AudioFileClip
from google.cloud import speech
from google.cloud import texttospeech
from google.cloud import storage
from dotenv import load_dotenv
from pydub import AudioSegment
import tempfile
import openai

# Load environment variables from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure your .env contains OPENAI_API_KEY
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")  # Path to your Google Cloud credentials

# Streamlit Interface
st.title("AI Video Audio Replacement")

# File uploader
video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

# Initialize Google Cloud Storage client
storage_client = storage.Client()

# Function to upload audio to Google Cloud Storage
def upload_to_gcs(bucket_name, source_file_name):
    blob_name = os.path.basename(source_file_name)  # Get just the file name
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(source_file_name)  # Upload the file
    return f"gs://aiaudiotovideo/"  # Return the valid GCS path including the filename

# Function to extract audio from video
def extract_audio(video_path):
    with VideoFileClip(video_path) as video_clip:
        audio_path = "extracted_audio.wav"
        video_clip.audio.write_audiofile(audio_path, codec='pcm_s16le')  # Ensure correct encoding
    return audio_path

# Function to convert audio to mono
def convert_to_mono(audio_path):
    audio = AudioSegment.from_wav(audio_path)
    mono_audio_path = "mono_audio.wav"
    mono_audio = audio.set_channels(1)  # Set to mono
    mono_audio.export(mono_audio_path, format="wav")
    return mono_audio_path

# Function to transcribe audio using Google Speech-to-Text
def transcribe_audio(audio_path):
    client = speech.SpeechClient()

    # Check audio length
    audio_length_seconds = AudioSegment.from_wav(audio_path).duration_seconds

    if audio_length_seconds <= 60:
        # Synchronous recognition for short audio
        with open(audio_path, "rb") as audio_file:
            content = audio_file.read()

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=44100,  # Match this with your audio file
            language_code="en-US",
            enable_automatic_punctuation=True,
        )

        response = client.recognize(config=config, audio=audio)
        transcript = ""
        for result in response.results:
            transcript += result.alternatives[0].transcript + " "

        return transcript.strip()
    else:
        # For longer audio, process synchronously for simplicity (or implement chunk processing if needed)
        with open(audio_path, "rb") as audio_file:
            content = audio_file.read()

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=44100,
            language_code="en-US",
            enable_automatic_punctuation=True,
        )

        response = client.recognize(config=config, audio=audio)
        transcript = ""
        for result in response.results:
            transcript += result.alternatives[0].transcript + " "

        return transcript.strip()

# Workflow Execution
if video_file is not None:
    generated_audio_path = None  # Initialize to None

    # Save the uploaded video file locally
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
        temp_video.write(video_file.read())
        temp_video_path = temp_video.name

    st.success("Video uploaded successfully!")

    try:
        # Step 1: Extract audio from video
        audio_path = extract_audio(temp_video_path)
        st.audio(audio_path, format="audio/wav")
        st.success("Audio extracted successfully!")

        # Step 2: Convert extracted audio to mono
        mono_audio_path = convert_to_mono(audio_path)
        st.success("Audio converted to mono successfully!")

        # Step 3: Transcribe the extracted audio
        transcript = transcribe_audio(mono_audio_path)
        st.text_area("Transcription:", transcript, height=200)

        # Step 4: Correct the transcription
        corrected_transcript = correct_transcription(transcript)
        st.text_area("Corrected Transcription:", corrected_transcript, height=200)

        # Step 5: Generate audio from corrected transcription
        generated_audio_path = generate_speech(corrected_transcript)
        st.audio(generated_audio_path, format="audio/wav")
        st.success("Audio generated successfully!")

        # Step 6: Replace original audio in the video
        replace_audio_in_video(temp_video_path, generated_audio_path)
        st.video("final_video.mp4")
        st.success("Final video with replaced audio is ready!")

    except Exception as e:
        st.error(f"An error occurred: {e}")

    finally:
        # Cleanup temporary files
        for path in [temp_video_path, audio_path, mono_audio_path, generated_audio_path if generated_audio_path else None, "final_video.mp4"]:
            if path and os.path.exists(path):  # Check if path is not None before trying to remove
                os.remove(path)

# Function to correct transcription using Azure OpenAI GPT-4
def correct_transcription(transcript):
    url = "https://internshala.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",  # Use your Azure OpenAI API key here
    }
    data = {
        "messages": [{"role": "user", "content": f"Correct this transcription:\n\n{transcript}"}],
        "model": "gpt-4o",
        "max_tokens": 500,
        "temperature": 0.7,
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()  # Raise an error for bad responses

    corrected_transcript = response.json()['choices'][0]['message']['content'].strip()
    return corrected_transcript

# Function to generate audio using Google Text-to-Speech
def generate_speech(text, output_audio_path="generated_audio.wav"):
    client = texttospeech.TextToSpeechClient()

    input_text = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US", name="en-US-JennyNeural", ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16)

    response = client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)

    with open(output_audio_path, "wb") as out:
        out.write(response.audio_content)
    
    return output_audio_path

# Function to replace original audio in the video
def replace_audio_in_video(video_path, new_audio_path, output_video_path="final_video.mp4"):
    video_clip = VideoFileClip(video_path)
    new_audio_clip = AudioFileClip(new_audio_path)
    final_video = video_clip.set_audio(new_audio_clip)
    final_video.write_videofile(output_video_path, codec='libx264')  # Specify codec for better compatibility

# Workflow Execution
if video_file is not None:
    # Save the uploaded video file locally
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
        temp_video.write(video_file.read())
        temp_video_path = temp_video.name

    st.success("Video uploaded successfully!")
    
    try:
        # Step 1: Extract audio from video
        audio_path = extract_audio(temp_video_path)
        st.audio(audio_path, format="audio/wav")
        st.success("Audio extracted successfully!")

        # Step 2: Convert extracted audio to mono
        mono_audio_path = convert_to_mono(audio_path)
        st.success("Audio converted to mono successfully!")

        # Step 3: Transcribe the extracted audio
        transcript = transcribe_audio(mono_audio_path)
        st.text_area("Transcription:", transcript, height=200)

        # Step 4: Correct the transcription
        corrected_transcript = correct_transcription(transcript)
        st.text_area("Corrected Transcription:", corrected_transcript, height=200)

        # Step 5: Generate audio from corrected transcription
        generated_audio_path = generate_speech(corrected_transcript)
        st.audio(generated_audio_path, format="audio/wav")
        st.success("Audio generated successfully!")

        # Step 6: Replace original audio in the video
        replace_audio_in_video(temp_video_path, generated_audio_path)
        st.video("final_video.mp4")
        st.success("Final video with replaced audio is ready!")

    except Exception as e:
        st.error(f"An error occurred: {e}")

    finally:
        # Cleanup temporary files
        for path in [temp_video_path, audio_path, mono_audio_path, generated_audio_path, "final_video.mp4"]:
            if os.path.exists(path):
                os.remove(path)