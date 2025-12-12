import itertools

def extract_tags_from_transcription(input_file="transcription.txt", output_file="extracted_tags.txt", all_tags=None):
    """
    Extracts relevant tags from transcription based on known tags.

    - Finds single-word matches from `all_tags`
    - Finds two-word combinations (e.g., "classic_rock") from `all_tags`
    - Saves extracted tags to a file
    """

    if all_tags is None:
        raise ValueError("Tag list (all_tags) cannot be None!")

    with open(input_file, "r") as file:
        words = file.read().lower().split()

    found_tags = set()

    for word in words:
        if word in all_tags:
            found_tags.add(word)

    for word1, word2 in itertools.pairwise(words):
        combined_tag = f"{word1}_{word2}"
        if combined_tag in all_tags:
            found_tags.add(combined_tag)

    with open(output_file, "w") as file:
        file.write(" ".join(sorted(found_tags)))

    print("Extracted Tags:", sorted(found_tags))
    print(f"Tags saved to {output_file}")

    return sorted(found_tags)

all_tags = {
    '00s', '60s', '70s', '80s', '90s', 'acoustic', 'alternative', 'alternative_rock',
    'ambient', 'american', 'avant_garde', 'beautiful', 'black_metal', 'blues',
    'blues_rock', 'british', 'britpop', 'chill', 'chillout', 'classic_rock',
    'classical', 'country', 'cover', 'dance', 'dark_ambient', 'death_metal',
    'doom_metal', 'downtempo', 'drum_and_bass', 'electro', 'electronic', 'emo',
    'experimental', 'female_vocalists', 'folk', 'french', 'funk', 'german', 'gothic',
    'gothic_metal', 'grindcore', 'grunge', 'guitar', 'hard_rock', 'hardcore',
    'heavy_metal', 'hip_hop', 'house', 'idm', 'indie', 'indie_pop', 'indie_rock',
    'industrial', 'instrumental', 'j_pop', 'japanese', 'jazz', 'lounge', 'love',
    'male_vocalists', 'mellow', 'melodic_death_metal', 'metal', 'metalcore',
    'new_age', 'new_wave', 'noise', 'nu_metal', 'oldies', 'piano', 'polish', 'pop',
    'pop_rock', 'post_hardcore', 'post_punk', 'post_rock', 'power_metal',
    'progressive_metal', 'progressive_rock', 'psychedelic', 'psychedelic_rock',
    'punk', 'punk_rock', 'rap', 'reggae', 'rnb', 'rock', 'russian', 'screamo',
    'singer_songwriter', 'ska', 'soul', 'soundtrack', 'swedish', 'symphonic_metal',
    'synthpop', 'techno', 'thrash_metal', 'trance', 'trip_hop'
}

extracted_tags = extract_tags_from_transcription(all_tags=all_tags)
