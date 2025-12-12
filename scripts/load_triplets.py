import pickle

def load_triplets(pkl_file="data/processed/triplets.pkl"):
    """
    Loads the triplets from the pickle file and prints a sample.
    """
    with open(pkl_file, "rb") as f:
        triplets = pickle.load(f)

    print(f"Loaded {len(triplets)} triplets from {pkl_file}")
    
    # Print a few sample triplets
    for i, triplet in enumerate(triplets[:10]):  # Adjust the number of samples as needed
        print(f"Triplet {i + 1}: Query: {triplet[0]}, Positive Index: {triplet[1]}, Negative Index: {triplet[2]}")

if __name__ == "__main__":
    load_triplets("data/processed/triplets.pkl")