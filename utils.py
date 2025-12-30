import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Define project root relative to this module's location
PROJECT_ROOT = Path(__file__).parent
DATA_FILE = PROJECT_ROOT / "cleaned_esv.json"
VISUALIZATIONS_DIR = PROJECT_ROOT / "visualizations"


def setup_logging(level=logging.INFO, format_string=None):
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    logging.basicConfig(
        level=level,
        format=format_string,
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def getAllVerses():
    data = getDataFromJson()
    return [text for verses in data.values() for text in verses.values()]


def getDataFromJson():
    with open(DATA_FILE, "r", encoding='utf-8') as json_file:
        return json.load(json_file)


def saveGraph(title: str, plt):
    outdir = VISUALIZATIONS_DIR
    try:
        outdir.mkdir(parents=True, exist_ok=True)
        plt.savefig(outdir / title)
    except PermissionError:
        raise PermissionError(
            f"Permission denied writing to {outdir / title}. Check directory permissions."
        )
    except OSError as e:
        raise OSError(f"Error saving graph to {outdir / title}: {e}")


def organize_clusters_data(
    labels: np.ndarray,
    verse_refs: List[Tuple[str, str]],
    verses_dict: Dict,
    k: int
) -> Dict:
    clusters_data = {}

    for cluster_id in range(k):
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_verses = []

        for idx in cluster_indices:
            chapter, verse = verse_refs[idx]
            text = verses_dict[chapter][verse]
            cluster_verses.append({
                "chapter": chapter,
                "verse": verse,
                "text": text,
                "index": int(idx)
            })

        clusters_data[f"cluster_{cluster_id}"] = {
            "cluster_id": cluster_id,
            "num_verses": len(cluster_indices),
            "verses": cluster_verses
        }

    return clusters_data
