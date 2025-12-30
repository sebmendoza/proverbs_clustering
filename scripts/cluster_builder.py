import json
from typing import TypedDict


class Cluster(TypedDict):
    cluster_id: int
    verses: list[str]
    num_verses: int


__all__ = ['Cluster', 'ClusterBuilder']


class ClusterBuilder:
    def __init__(self, json_filepath: str):
        self.json_filepath = json_filepath
        self.clusters: list[Cluster] = []
        self._load_clusters()

    def _load_clusters(self):
        with open(self.json_filepath, 'r') as f:
            data = json.load(f)

        clusters_dict = data.get('clusters', {})

        for _, cluster_data in clusters_dict.items():
            cluster_id = cluster_data['cluster_id']
            verses = [verse['text'] for verse in cluster_data['verses']]

            self.clusters.append({
                'cluster_id': cluster_id,
                'verses': verses,
                'num_verses': len(verses)
            })

        print(
            f"Loaded {len(self.clusters)} clusters from {self.json_filepath}")

    def get_clusters(self) -> list[Cluster]:
        return self.clusters
