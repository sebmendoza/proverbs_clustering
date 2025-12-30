import json


def main():
    out_file = "kmean_19_clusters_for_summarizing.txt"

    with open("./outputs/esv_kmeans_19_clusters.json", "r") as f:
        data = json.load(f)

    with open(out_file, "w") as f:
        for cluster in data.values():
            f.write(f"\n\nCluster {cluster['cluster_id']}\n")
            for verse in cluster["verses"]:
                f.write(f"{verse['text']}\n")


if __name__ == "__main__":
    main()
