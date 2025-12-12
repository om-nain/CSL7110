"""
Uses the trained SongEncoder (song_encoder.pth) and the fine-tuned DistilBert-based
QueryEncoder (query_encoder_finetuned.pth) to recommend songs based on
a user query string.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import numpy as np
import pandas as pd
from modules.query_encoder import DistilBertQueryEncoder
from modules.song_encoder import SongEncoder


def load_models(
    query_encoder_path="query_encoder_finetuned.pth",
    song_encoder_path="song_encoder.pth",
    input_dim=128,
    embed_dim=128
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading models on device:", device)

    query_encoder = DistilBertQueryEncoder(embed_dim=embed_dim).to(device)
    query_encoder.load_state_dict(torch.load(query_encoder_path, map_location=device))
    query_encoder.eval()

    song_encoder = SongEncoder(input_dim=input_dim, embed_dim=embed_dim).to(device)
    song_encoder.load_state_dict(torch.load(song_encoder_path, map_location=device))
    song_encoder.eval()

    return query_encoder, song_encoder, device


def precompute_song_embeddings(df, song_encoder, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    song_encoder.eval()
    embeddings = []
    for idx in df.index:
        row_values = df.iloc[idx].values.astype(np.float32)
        row_t = torch.tensor(row_values, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = song_encoder(row_t)
        embeddings.append(emb.cpu().numpy().flatten())
    return np.array(embeddings)


def recommend_songs(
    user_query,
    query_encoder,
    song_embeddings,
    df,
    device=None,
    top_k=5
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    query_encoder.eval()
    with torch.no_grad():
        emb = query_encoder([user_query])
    query_vec = emb.cpu().numpy()[0]

    norms = np.linalg.norm(song_embeddings, axis=1)
    q_norm = np.linalg.norm(query_vec)
    dot_products = song_embeddings.dot(query_vec)
    sims = dot_products / (norms * q_norm + 1e-10)

    sorted_idxs = np.argsort(-sims)[:top_k]
    return sorted_idxs
