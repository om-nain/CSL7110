## Installation

Before running the project, ensure you have installed all the required dependencies and downloaded the necessary models.

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## Steps to Run

Follow these steps to use the project:

### 1. Record Your Audio
Run the following script to record your voice:

```bash
python record_audio.py
```

- Speak into the microphone for **5 seconds**.
- The recording will be saved as `output.wav`.

---

### 2. Transcribe Into Text
Convert your recorded audio into text using the following command:

```bash
python voice_to_text.py
```

- The transcription will be saved in a file named `transcription.txt`.

---

### 3. Extract your tags
Extract tags from the transcription by running the script:

```bash
python preprocessing.py
```

- The extracted tags will be saved in a file named `extracted_tags.txt`.

---

## Colab Notebook for Generating Embeddings

Find the colab notebook here : [Colab Notebook : b22ai002@iitj.ac.in](https://colab.research.google.com/drive/1X6qDHoGEqQzxzz1wXFxEZaNliE5qCMyt?usp=sharing)

### 4. Get the top recommendation

- Set the user_input in the notebook as the extracted tags. Run the cells and get the preview link for the top recommendation.