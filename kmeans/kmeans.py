"""
K-means Clustering Module (Legacy)

This module contains legacy K-means functions. For new clustering experiments,
use the clustering_pipeline module instead.
"""

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

from utils import saveGraph, getDataFromJson
from kmeans.cluster_quality import compute_cluster_metrics, flag_quality_issues, get_quality_summary
from kmeans.cluster_viz import create_2d_visualization, create_3d_visualization, create_cluster_distribution_plot


def print_clusters(labels, verse_refs, k, verses_dict, output_file="esv_kmeans_18_clusters.json"):
    # Save all verses organized by their cluster assignments to JSON file.
    clusters_data = {}

    for cluster_id in range(k):
        cluster_indices = np.where(labels == cluster_id)[0]

        # Print summary to console
        print(f"\n{'='*60}")
        print(f"CLUSTER {cluster_id} ({len(cluster_indices)} verses)")
        print(f"{'='*60}")

        # Build cluster data for JSON
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

            # Print abbreviated version to console
            print(f"  [{idx}] Proverbs {chapter}:{verse} - {text[:80]}...")

        clusters_data[f"cluster_{cluster_id}"] = {
            "cluster_id": cluster_id,
            "num_verses": len(cluster_indices),
            "verses": cluster_verses
        }

    # Save to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(clusters_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Saved {k} clusters to {output_file}")
    print(f"{'='*60}")


# plot_cluster_centers_2d, plot_cluster_centers_3d, and print_quality_report
# have been moved to cluster_viz.py and cluster_quality.py


def run_kmeans_experiments(data, verse_refs):
    # Legacy function for running K-means experiments. For new experiments, use clustering_pipeline.run_clustering_pipeline() instead.

    # K-means clustering: partitions data into k clusters by minimizing within-cluster variance
    # Key limitation: you must specify k beforehand (unlike hierarchical clustering)
    # We test multiple k values to find the "best" number of clusters

    K = range(2, 30)
    fits = []
    scores = []
    inertias = []
    d_scores = []

    print("Running K-means experiments (k=2 to 29)...")
    for k in K:
        # random_state=42 ensures reproducibility (same initialization each run)
        # n_init='auto' lets sklearn choose number of random initializations
        model = KMeans(n_clusters=k, random_state=42,
                       n_init='auto').fit(data)
        fits.append(model)

        # Silhouette score: measures how similar a point is to its own cluster vs other clusters
        # Range: -1 to 1, higher is better. Measures cluster cohesion and separation.
        scores.append(silhouette_score(
            data, model.labels_, metric='euclidean'))

        # Inertia: sum of squared distances from points to their cluster center (within-cluster variance)
        # Lower is better, but decreases as k increases (more clusters = tighter fit)
        inertias.append(model.inertia_)

        # Davies-Bouldin score: ratio of within-cluster to between-cluster distances
        # Lower is better (want tight clusters that are far apart)
        d_scores.append(davies_bouldin_score(data, model.labels_))

    # Plot the scores
    plt.figure()
    plt.plot(K, scores, marker='o', linewidth=2, color='steelblue')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.title('Silhouette score vs Number of clusters')
    saveGraph("kmeans_silhouette_score_vs_number_of_clusters.png", plt)
    plt.close()

    # Plot the inertias
    plt.figure()
    plt.plot(K, inertias)
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Inertia vs Number of clusters')
    saveGraph("kmeans_inertia_vs_number_of_clusters.png", plt)
    plt.close()

    # Plot the d_scores
    plt.figure()
    plt.plot(K, d_scores)
    plt.xlabel('Number of clusters')
    plt.ylabel('D_score')
    plt.title('D_score vs Number of clusters')
    saveGraph("kmeans_d_scores_vs_number_of_clusters.png", plt)
    plt.close()

    # Hardcoded best_k_idx = 17 means k=19 was chosen manually based on the plots
    # In practice, you'd use elbow method (inertia plot) or silhouette analysis
    best_k_idx = 17
    best_k = K[best_k_idx]
    best_model = fits[best_k_idx]
    verses_dict = getDataFromJson()

    if verse_refs:
        output_filename = f"esv_kmeans_{best_k}_clusters.json"
        print_clusters(best_model.labels_, verse_refs,
                       best_k, verses_dict, output_filename)

    # Analyze cluster quality using new module
    print("\nAnalyzing cluster quality...")
    metrics = compute_cluster_metrics(data, best_model.labels_, best_k)
    warnings = flag_quality_issues(metrics, best_k)
    summary = get_quality_summary(metrics, warnings)
    print(summary)

    # Create visualizations using new module
    print("\nCreating visualizations...")
    viz_dir = Path("visualizations")
    create_cluster_distribution_plot(best_model.labels_, best_k,
                                     viz_dir / f"kmeans_{best_k}_cluster_distribution.png")
    create_2d_visualization(data, best_model.labels_, best_k,
                            viz_dir / f"kmeans_{best_k}_clusters_2d.png")
    create_3d_visualization(data, best_model.labels_, best_k,
                            viz_dir / f"kmeans_{best_k}_clusters_3d.html")

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print("\nFor more advanced experiments with multiple k values and LLM titles,")
    print("use: clustering_pipeline.run_clustering_pipeline()")
