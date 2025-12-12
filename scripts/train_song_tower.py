"""
Trains a two-tower system:
  - Query Tower: DistilBERT-based (fine-tuneable)
  - Song Tower: MLP on numeric columns (including multi-hot tags, etc.)
"""

import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel


class DistilBertQueryEncoder(nn.Module):
    def __init__(self, embed_dim=128):
        super(DistilBertQueryEncoder, self).__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.projection = nn.Linear(768, embed_dim)

    def forward(self, text_list):
        encoded = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.projection.weight.device)  # Ensure tensors are on the same device
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        outputs = self.distilbert(input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]  # shape [batch_size, 768]
        query_emb = self.projection(cls_token)          # shape [batch_size, embed_dim]
        return query_emb


class SongEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=128):
        super(SongEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )

    def forward(self, x):
        return self.net(x)  # shape [batch_size, embed_dim]


def contrastive_loss(query_emb, pos_emb, neg_emb, margin=1.0):
    """
    Margin-based contrastive approach using cosine similarity
    """
    cos_sim = nn.CosineSimilarity(dim=-1)
    pos_sim = cos_sim(query_emb, pos_emb)  # shape [batch_size]
    neg_sim = cos_sim(query_emb, neg_emb)
    loss_val = torch.relu(margin - pos_sim + neg_sim).mean()
    return loss_val


class TripletDataset(Dataset):
    def __init__(self, df, triplets):
        super().__init__()
        self.df = df
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        query_text, pos_i, neg_i = self.triplets[idx]
        pos_feats = self.get_song_features(pos_i)
        neg_feats = self.get_song_features(neg_i)
        return query_text, pos_feats, neg_feats

    def get_song_features(self, row_idx):
        row = self.df.iloc[row_idx]
        feats = row.values.astype(np.float32)
        return feats


def collate_fn(batch):
    query_list = []
    pos_list = []
    neg_list = []
    for (q, p, n) in batch:
        query_list.append(q)
        pos_list.append(p)
        neg_list.append(n)

    pos_arr = np.array(pos_list, dtype=np.float32)
    neg_arr = np.array(neg_list, dtype=np.float32)
    pos_tensor = torch.tensor(pos_arr, dtype=torch.float32)
    neg_tensor = torch.tensor(neg_arr, dtype=torch.float32)
    return query_list, pos_tensor, neg_tensor


def train_two_tower(
    df_csv="data_processed/MusicInfo_tagged.csv",
    triplets_pkl="triplets.pkl",
    embed_dim=128,
    margin=1.0,
    batch_size=8,
    lr=1e-4,
    num_epochs=3,
    output_query_path="query_encoder_finetuned.pth",
    output_song_path="song_encoder.pth"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    df = pd.read_csv(df_csv)
    df.fillna(0, inplace=True)

    input_dim = len(df.columns)
    print(f"Loaded numeric DataFrame with shape: {df.shape}, input_dim={input_dim}")

    with open(triplets_pkl, "rb") as f:
        triplets = pickle.load(f)
    print(f"Loaded {len(triplets)} triplets from {triplets_pkl}")

    query_encoder = DistilBertQueryEncoder(embed_dim=embed_dim).to(device)
    song_encoder = SongEncoder(input_dim=input_dim, embed_dim=embed_dim).to(device)

    dataset = TripletDataset(df, triplets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = optim.Adam(
        list(query_encoder.parameters()) + list(song_encoder.parameters()),
        lr=lr
    )

    for epoch in range(num_epochs):
        query_encoder.train()
        song_encoder.train()

        total_loss = 0.0
        print(f"Starting epoch {epoch + 1}/{num_epochs}...")
        for i, (query_list, pos_feats, neg_feats) in enumerate(loader):
            pos_feats = pos_feats.to(device)
            neg_feats = neg_feats.to(device)

            optimizer.zero_grad()

            query_emb = query_encoder(query_list)  # shape [batch_size, embed_dim]
            query_emb = query_emb.to(device)

            pos_emb = song_encoder(pos_feats)  # shape [batch_size, embed_dim]
            neg_emb = song_encoder(neg_feats)  # shape [batch_size, embed_dim]

            loss_val = contrastive_loss(query_emb, pos_emb, neg_emb, margin=margin)
            loss_val.backward()
            optimizer.step()

            total_loss += loss_val.item()

            if (i + 1) % 10 == 0:  # Print every 10 batches
                print(f"Batch {i + 1}/{len(loader)}, Loss: {loss_val.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1}/{num_epochs} completed. Average Loss: {avg_loss:.4f}")

    torch.save(query_encoder.state_dict(), output_query_path)
    torch.save(song_encoder.state_dict(), output_song_path)
    print(f"Training complete. Saved query encoder -> {output_query_path}")
    print(f"Training complete. Saved song encoder -> {output_song_path}")


if __name__ == "__main__":
    train_two_tower(
        df_csv="https://raw.githubusercontent.com/introspective321/SonicSynapse/refs/heads/main/data_processed/MusicInfo_tagged.csv",
        triplets_pkl="/kaggle/input/triplet/triplets.pkl",
        embed_dim=128,
        margin=1.0,
        batch_size=8,
        lr=1e-4,
        num_epochs=6,
        output_query_path="query_encoder_finetuned.pth",
        output_song_path="song_encoder.pth"
    )