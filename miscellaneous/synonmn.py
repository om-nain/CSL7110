import itertools
from nltk.corpus import wordnet
import nltk

# Ensure WordNet resources are available
nltk.download('wordnet', quiet=True)

def get_synonyms(word):
    """Generate music-aware synonyms with phrase handling"""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            # Handle multi-word expressions and preserve music terminology
            synonym = lemma.name().lower().replace(' ', '_')
            synonyms.add(synonym)
    
    # Music-specific synonym overrides
    music_mappings = {
        'upbeat': ['energetic', 'lively', 'fast-paced'],
        'mellow': ['chill', 'relaxing', 'laid-back'],
        'heavy': ['intense', 'powerful', 'aggressive'],
        'electronic': ['synth', 'techno', 'edm']
    }
    return list(synonyms.union(music_mappings.get(word, [])))

def expand_with_synonyms(words):
    """Create expanded word list with priority for original terms"""
    expanded = []
    for word in words:
        expanded.append(word)  # Preserve original position
        expanded.extend(get_synonyms(word))
    return expanded

def extract_tags_from_transcription(input_file="transcription.txt", 
                                   output_file="extracted_tags.txt", 
                                   all_tags=None):
    """
    Enhanced tag extraction with multi-layer synonym matching:
    1. Direct matches
    2. Single-word synonym matches
    3. Original pairwise matches
    4. Synonym-expanded pairwise matches
    """
    if all_tags is None:
        raise ValueError("Tag list (all_tags) cannot be None!")

    with open(input_file, "r") as file:
        original_words = file.read().lower().split()
    
    expanded_words = expand_with_synonyms(original_words)
    found_tags = set()

    # Single-word matching phase
    unique_terms = set(expanded_words)  # Deduplicate while preserving order
    for term in unique_terms:
        if term in all_tags:
            found_tags.add(term)

    # Pairwise matching with synonym combinations
    for i in range(len(original_words)-1):
        word1, word2 = original_words[i], original_words[i+1]
        
        # Base combination
        base_pair = f"{word1}_{word2}"
        if base_pair in all_tags:
            found_tags.add(base_pair)
        
        # Generate all possible synonym pairs
        for syn1 in get_synonyms(word1):
            for syn2 in get_synonyms(word2):
                combined = f"{syn1}_{syn2}"
                if combined in all_tags:
                    found_tags.add(combined)

    # Save results with relevance scoring
    sorted_tags = sorted(found_tags, 
                        key=lambda x: (x in original_words, x.count('_')), 
                        reverse=True)

    with open(output_file, "w") as file:
        file.write(" ".join(sorted_tags))

    print(f"Identified {len(sorted_tags)} tags:")
    print(", ".join(sorted_tags))
    return sorted_tags

# Example usage remains the same
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
