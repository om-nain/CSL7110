import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

def transcribe_audio(file_path="output.wav", model="openai/whisper-large-v3-turbo", output_text_file="transcription.txt"):
    api_key = os.getenv("HUGGING_FACE_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Make sure it's set in the .env file.")
    
    client = InferenceClient(api_key=api_key)
    
    with open(file_path, "rb") as audio_file:
        audio_bytes = audio_file.read()
        output = client.automatic_speech_recognition(audio_bytes, model=model)
    
    transcription = output['text']
    
    with open(output_text_file, "w") as text_file:
        text_file.write(transcription)
    
    print(f"Transcription saved to {output_text_file}")
    print("Transcription:", transcription)
    
    return transcription

if __name__ == "__main__":
    transcribe_audio()
