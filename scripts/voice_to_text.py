import whisper

def transcribe_audio(file_path="output.wav", output_text_file="data/user/transcriptions/transcription.txt"):

    model = whisper.load_model("base")
    audio = "data/user/audio/output.wav"
    result = model.transcribe(audio)
    
    transcription = result["text"]
    
    with open(output_text_file, "w") as text_file:
        text_file.write(transcription)
    
    print(f"Transcription saved to {output_text_file}")
    print("Transcription:", transcription)
    
    return transcription

if __name__ == "__main__":
    transcribe_audio()
