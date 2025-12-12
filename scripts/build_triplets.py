"""
Builds a DataFrame of purely numeric features + multi-hot tags.
Generates synthetic triplets (query_text, pos_song_idx, neg_song_idx).
Saves them to a pickle file for training.
"""

import pandas as pd
import random
import pickle

def load_numeric_dataframe(csv_path="MusicInfo_tagged.csv"):
    """
    Loads the CSV file and ensures all columns are numeric or boolean.
    Drops non-numeric columns dynamically based on the data.
    """
    df = pd.read_csv(csv_path)
    
    #
    non_numeric_cols = df.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    if non_numeric_cols:
        print(f"Dropping non-numeric columns: {non_numeric_cols}")
        df.drop(columns=non_numeric_cols, inplace=True)

    
    df.fillna(0, inplace=True)
    return df


def build_synthetic_triplets(df, output_pkl="triplets.pkl", num_triplets=1000):
    """
    Generates synthetic triplets (query_text, pos_song_idx, neg_song_idx).
    Dynamically creates a query_text based on the multi-hot tag columns.
    """
    # Identify multi-hot tag columns (columns with binary values)
    tag_columns = [col for col in df.columns if df[col].dropna().isin([0, 1]).all()]
    if not tag_columns:
        raise ValueError("No multi-hot tag columns found in the dataset.")

    print(f"Using tag columns for queries: {tag_columns}")

    # Remove prefixes from column names for cleaner query text
    clean_column_names = {col: col.replace("genre_", "").replace("tag_", "") for col in tag_columns}

    triplets = []
    all_indices = df.index.tolist()


    for _ in range(num_triplets):
        pos_idx = random.choice(all_indices)

        # Dynamically generate query_text from tag columns where value is 1
        query_tags = [clean_column_names[col] for col in tag_columns if df.at[pos_idx, col] == 1]
        query_text = " ".join(query_tags)

        if not query_text.strip():
            continue  # Skip if no tags are found for the query

        neg_idx = random.choice(all_indices)
        # Ensure the negative index is different from the positive index
        tries = 0
        while neg_idx == pos_idx and tries < 5:
            neg_idx = random.choice(all_indices)
            tries += 1

        triplets.append((query_text, pos_idx, neg_idx))

    # Save triplets to disk
    with open(output_pkl, "wb") as f:
        pickle.dump(triplets, f)

    print(f"Saved {len(triplets)} triplets to {output_pkl}")

if __name__ == "__main__":
    df = load_numeric_dataframe("data/processed/MusicInfo_tagged.csv")
    build_synthetic_triplets(df, output_pkl="data/processed/triplets.pkl", num_triplets=8000)
