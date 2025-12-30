from sentence_transformers import SentenceTransformer
from pathlib import Path
import pickle
from utils import getDataFromJson


def create_embeddings():
    embeddings_dict = dict()
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    data = getDataFromJson()
    for chapter, verses in data.items():
        if chapter not in embeddings_dict:
            embeddings_dict[chapter] = {}
        for verse, text in verses.items():
            embedding = model.encode(text, show_progress_bar=True)
            embeddings_dict[chapter][verse] = embedding
    return embeddings_dict


def get_or_create_embeddings():
    cache_file = Path('esv_embeddings.pkl')
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            print("Got from pickle...")
            return pickle.load(f)

    embeddings_dict = create_embeddings()
    with open(cache_file, 'wb') as f:
        pickle.dump(embeddings_dict, f)

    return embeddings_dict


def embeddings_dict_to_array(data):
    # Convert embeddings dictionary to numpy array with verse references.
    embeddings = []
    verse_refs = []
    for chapter, verses in data.items():
        for verse, embed in verses.items():
            embeddings.append(embed)
            verse_refs.append((chapter, verse))

    return embeddings, verse_refs
