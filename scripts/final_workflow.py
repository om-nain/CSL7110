from inference import load_models, precompute_song_embeddings, recommend_songs
import pandas as pd
import webbrowser
from record_audio import record_audio
from voice_to_text import transcribe_audio


def top_indices(top_k=5):
    df = pd.read_csv("data/processed/MusicInfo_tagged.csv")
    df.fillna(0, inplace=True)
    input_dim = len(df.columns)

    # Load trained models
    query_encoder, song_encoder, device = load_models(
        query_encoder_path="models/v1/query_encoder_finetuned.pth",
        song_encoder_path="models/v1/song_encoder.pth",
        input_dim=input_dim,
        embed_dim=128
    )

    # Precompute song embeddings
    song_embs = precompute_song_embeddings(df, song_encoder, device=device)

    # User query
    query_file = "data/user/transcriptions/transcription.txt"
    try:
        with open(query_file, "r") as file:
            user_query = file.read().strip()
        print(f"User query: {user_query}")
    except FileNotFoundError:
        print(f"Error: Query file '{query_file}' not found.")
        exit(1)

    # Get top song indices
    top_idxs = recommend_songs(
        user_query,
        query_encoder,
        song_embs,
        df,
        device=device,
        top_k=top_k
    )

    return top_idxs


def play_songs_one_by_one(indices, csv_file="data/raw/MusicInfo.csv", link_column="spotify_link"):
    """
    Plays songs one by one based on the given indices using Spotify links.
    Waits for user confirmation before playing the next song.
    """
    df = pd.read_csv(csv_file)

    if link_column not in df.columns:
        raise ValueError(f"Column '{link_column}' not found in the CSV file.")

    for idx in indices:
        if idx < 0 or idx >= len(df):
            print(f"Index {idx} is out of bounds. Skipping...")
            continue

        link = df.iloc[idx][link_column]
        if pd.isna(link):
            print(f"No Spotify link found for index {idx}. Skipping...")
            continue

        print(f"Playing song at index {idx}: {link}")
        webbrowser.open(link)

        # Wait for user confirmation to play the next song
        input("Press Enter to play the next song...")


if __name__ == "__main__":

    record_audio()
    transcribe_audio()
    top_song_indices = top_indices(top_k=5)

    csv_file_path = "data/raw/MusicInfo.csv"
    
    spotify_link_column = "spotify_preview_url"
    play_songs_one_by_one(top_song_indices, csv_file=csv_file_path, link_column=spotify_link_column)