import logging
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pickle
from utils import getDataFromJson

logger = logging.getLogger(__name__)

# Define project root relative to this module's location
PROJECT_ROOT = Path(__file__).parent
CACHE_FILE = PROJECT_ROOT / "esv_embeddings.pkl"


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
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'rb') as f:
                logger.info("Got from pickle...")
                return pickle.load(f)
        except PermissionError:
            raise PermissionError(
                f"Permission denied reading {CACHE_FILE}. Check file permissions."
            )
        except pickle.UnpicklingError as e:
            raise ValueError(
                f"Error unpickling {CACHE_FILE}. File may be corrupted: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error reading {CACHE_FILE}: {e}")

    embeddings_dict = create_embeddings()
    try:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(embeddings_dict, f)
    except PermissionError:
        raise PermissionError(
            f"Permission denied writing to {CACHE_FILE}. Check directory permissions."
        )
    except OSError as e:
        raise OSError(f"Error writing to {CACHE_FILE}: {e}")

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
