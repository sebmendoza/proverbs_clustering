import json


with open("experiments/experiment_20251223_114054/k_19/clusters.json", "r") as f:
    data = json.load(f)

counter = 0
for cluster in data["clusters"].values():
    print("Cluster " + str(counter) + ": " + cluster["titles"]["title_4words"])
    counter += 1

print("--------------------------------")
with open("kmean_19_clusters_for_summarizing.txt", "r") as f:
    text = f.read()

for line in text.split("\n"):
    line = line.strip()
    if line.startswith("Cluster "):
        print(line)
