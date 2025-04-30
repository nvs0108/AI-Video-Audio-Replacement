# AI Video Audio Replacement

AI Video Audio Replacement is a Python-based web application that replaces the original audio in a video with AI-generated voice, using a pipeline of speech-to-text, text correction, and text-to-speech. It uses Google's Speech-to-Text API to transcribe video audio, Azure OpenAI's GPT-4 to correct and enhance the transcription, and Google's Text-to-Speech API to synthesize new audio, which is then merged into the original video. The application provides a user-friendly interface built with Streamlit, making it easy to upload a video, process the audio, and download the final output with replaced audio.

## Features

- Upload a video file and automatically transcribe its audio using Google Speech-to-Text  
- Use GPT-4 to correct grammar and improve the quality of the transcript  
- Convert the refined text to speech using Google Text-to-Speech  
- Replace the original video’s audio track with the AI-generated audio using MoviePy  
- Streamlit-powered frontend for seamless user interaction

## Tech Stack

- Python for backend logic and integrations  
- Streamlit for the user interface  
- Google Cloud Speech-to-Text for transcription  
- Azure OpenAI GPT-4 for text correction  
- Google Cloud Text-to-Speech for AI voice generation  
- MoviePy for video and audio editing

## Installation

1. Clone the repository:  
   `git clone https://github.com/nvs0108/AI-Video-Audio-Replacement.git && cd AI-Video-Audio-Replacement`  
2. Install the dependencies:  
   `pip install -r requirements.txt`  
3. Set up API credentials in a `.env` file:
   GOOGLE_APPLICATION_CREDENTIALS="path_to_your_google_credentials.json"
   AZURE_API_KEY="your_azure_openai_key"
   AZURE_API_URL="your_azure_openai_endpoint"
4. Run the Streamlit app:  
`streamlit run app.py`

## Usage

1. Open the Streamlit app in your browser at `http://localhost:8501`  
2. Upload a video file  
3. The app will transcribe the audio, refine the text with GPT-4, and generate a new audio file  
4. The new audio replaces the original video’s audio track  
5. Download the final video with AI-enhanced audio

## Project Structure

- `app.py`: Main Streamlit application  
- `requirements.txt`: Python dependencies  
- `.env`: Stores API credentials (excluded from Git for security)

## Future Enhancements

- Add support for additional languages and voices  
- Provide user control over generated voice type or accent  
- Enable transcript editing before synthesis  
- Improve video processing speed and file size handling

## License

This project is licensed under the MIT License.

## Acknowledgments

Thanks to Google Cloud, Microsoft Azure, and OpenAI for the APIs and tools used in building this application.
