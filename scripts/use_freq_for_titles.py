from cluster_builder import ClusterBuilder, Cluster
from collections import Counter
import re


class FrequencyTitleGenerator:
    def __init__(self, cluster_of_verses: list[Cluster]):
        self.verses = cluster_of_verses

    def get_top_n__words_per_cluster(self, n: int):
        for cluster in self.verses:
            id = cluster["cluster_id"]
            verses = cluster["verses"]
            num_verses = cluster["num_verses"]

            all_text = " ".join(verses)
            words = all_text.split()
            cleaned_words = [word.lower() for word in words]
            cleaned_words = [re.sub(r'[.,;:!?\(\)]', '', word)
                             for word in cleaned_words]
            counter = Counter(cleaned_words)
            filtered_counter = counter.most_common(n)
            print("Cluster " + str(id) + " has " + str(num_verses) + " verses")
            print("Words: " + str(filtered_counter))
            print("\n")


if __name__ == "__main__":
    json_file = "experiments/experiment_20251223_114054/k_20/clusters.json"
    cluster_builder = ClusterBuilder(json_file)
    clusters = cluster_builder.get_clusters()
    generator = FrequencyTitleGenerator(clusters)
    generator.get_top_n__words_per_cluster(30)
