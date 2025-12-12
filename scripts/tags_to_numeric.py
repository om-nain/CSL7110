import pandas as pd
import ast

def parse_tag_list(tag_str):
    try:
        return ast.literal_eval(tag_str)
    except:
        return []

def build_tag_columns(input_csv, output_csv="MusicInfo_tagged.csv", tag_col="tags"):
    
    df = pd.read_csv(input_csv)
    df.fillna("", inplace=True)
    
    # 1) Collect all tags
    all_tags = set()
    for i, row in df.iterrows():
        tag_list = parse_tag_list(row.get(tag_col, "[]"))
        all_tags.update(tag_list)
    all_tags = sorted(all_tags)
    print("Number of unique tags found:", len(all_tags))

    # 2) Make a column for each tag
    for tag in all_tags:
        col_name = f"tag_{tag}"
        df[col_name] = 0  # initialize with zeros

    # 3) Set 1 where track has that tag
    for i, row in df.iterrows():
        tag_list = parse_tag_list(row.get(tag_col, "[]"))
        for tg in tag_list:
            col_name = f"tag_{tg}"
            if col_name in df.columns:
                df.at[i, col_name] = 1

    # 4) Drop the original 'tags' column and other specified columns
    columns_to_drop = [
        tag_col, "year", "danceability", "energy", "loudness", "speechiness", 
        "acousticness", "instrumentalness", "liveness", "valence", 
        "tempo", "duration_ms", "key", "mode"
    ]
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

    df.to_csv(output_csv, index=False)
    print(f"Saved multi-hot CSV -> {output_csv}")


if __name__ == "__main__":
    input_csv = "data_processed/Processed_Music_Info.csv"
    output_csv = "data_processed/MusicInfo_tagged.csv"
    build_tag_columns(input_csv, output_csv, tag_col="tags")
