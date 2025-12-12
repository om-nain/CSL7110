import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np

def record_audio(duration=10, sample_rate=44100, output_filename='data/user/audio/output.wav'):
    print(f"Recording for {duration} seconds...")
    
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=2, dtype='float32')
    sd.wait()
    print("Recording complete!")
    
    audio_data = (audio_data * 32767).astype(np.int16)
    write(output_filename, sample_rate, audio_data)
    print(f"Audio saved as {output_filename}")

if __name__ == "__main__":
    record_audio()
