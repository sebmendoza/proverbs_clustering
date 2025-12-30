import collections
from sklearn.decomposition import PCA
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.metrics import (
    silhouette_score, homogeneity_score, completeness_score,
    v_measure_score, adjusted_rand_score, adjusted_mutual_info_score
)
from numpy.typing import NDArray
import seaborn as sns
from sklearn.metrics.pairwise import cosine_distances


out_dir = Path("visualizations")


def initialSimiliarity(data: NDArray):
    output_dir = Path("visualizations")
    out = output_dir / "semantic_embedding_similarity_matrix.png"
    similarities = np.dot(data, data.T)
    print("\n Similarity analysis:")
    print(f"   • Average similarity: {similarities.mean():.3f}")
    print(
        f"   • Min/Max similarity: {similarities.min():.3f} / {similarities.max():.3f}")
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarities, annot=False, cmap='viridis', square=True)
    plt.title("Semantic Embedding Similarity Matrix")
    plt.xlabel("Verses")
    plt.ylabel("Verses")
    plt.tight_layout()

    plt.savefig(out)


def distance_metrics(data: NDArray):
    out = out_dir / "semantic_embedding_similarity_matrix.png"
    # Compute pairwise cosine distances
    dist_matrix = cosine_distances(data)
    mask = ~np.eye(dist_matrix.shape[0], dtype=bool)
    distances_flat = dist_matrix[mask]

    # Print statistics
    print(f" Cosine distance statistics:")
    print(f"   • Minimum: {distances_flat.min():.4f}")
    print(f"   • 5th percentile: {np.percentile(distances_flat, 5):.4f}")
    print(f"   • 10th percentile: {np.percentile(distances_flat, 10):.4f}")
    print(f"   • Median: {np.percentile(distances_flat, 50):.4f}")
    print(f"   • Maximum: {distances_flat.max():.4f}")

    # Plot distance distribution
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(distances_flat, bins=30, alpha=0.7, edgecolor='black')
    plt.title("Cosine Distance Distribution")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(range(len(distances_flat)), np.sort(
        distances_flat), 'b-', linewidth=2)
    plt.title("Sorted Distance Curve")
    plt.xlabel("Sentence Pairs")
    plt.ylabel("Distance")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    out = out_dir / "distance_metrics.png"
    plt.savefig(out)


def evaluate_clustering(embeddings, labels, labels_true=None, method_name="Clustering"):
    # Evaluate clustering quality with multiple metrics.

    # Basic statistics
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    n_points = len(labels)
    noise_ratio = n_noise / n_points if n_points > 0 else 0

    # Handle edge cases
    if n_clusters <= 1:
        result = {
            'method': method_name,
            'silhouette': 0,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'n_points': n_points,
            'noise_ratio': noise_ratio
        }
        if labels_true is not None:
            result.update({
                'homogeneity': 0, 'completeness': 0, 'v_measure': 0,
                'adjusted_rand': 0, 'adjusted_mutual_info': 0
            })
        else:
            result.update({
                'homogeneity': np.nan, 'completeness': np.nan, 'v_measure': np.nan,
                'adjusted_rand': np.nan, 'adjusted_mutual_info': np.nan
            })
        return result

    # Filter noise points
    valid_mask = labels != -1
    if valid_mask.sum() < 2:
        result = {
            'method': method_name,
            'silhouette': 0,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'n_points': n_points,
            'noise_ratio': noise_ratio
        }
        if labels_true is not None:
            result.update({
                'homogeneity': 0, 'completeness': 0, 'v_measure': 0,
                'adjusted_rand': 0, 'adjusted_mutual_info': 0
            })
        else:
            result.update({
                'homogeneity': np.nan, 'completeness': np.nan, 'v_measure': np.nan,
                'adjusted_rand': np.nan, 'adjusted_mutual_info': np.nan
            })
        return result

    valid_embeddings = embeddings[valid_mask]
    valid_labels = labels[valid_mask]

    # Calculate unsupervised metrics
    try:
        silhouette = silhouette_score(valid_embeddings, valid_labels)
    except:
        silhouette = 0

    result = {
        'method': method_name,
        'silhouette': silhouette,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'n_points': n_points,
        'noise_ratio': noise_ratio
    }

    # Calculate supervised metrics if true labels available
    if labels_true is not None:
        try:
            valid_labels_true = np.array(labels_true)[valid_mask]

            homogeneity = homogeneity_score(valid_labels_true, valid_labels)
            completeness = completeness_score(valid_labels_true, valid_labels)
            v_measure = v_measure_score(valid_labels_true, valid_labels)
            adjusted_rand = adjusted_rand_score(
                valid_labels_true, valid_labels)
            adjusted_mutual_info = adjusted_mutual_info_score(
                valid_labels_true, valid_labels)

            result.update({
                'homogeneity': homogeneity,
                'completeness': completeness,
                'v_measure': v_measure,
                'adjusted_rand': adjusted_rand,
                'adjusted_mutual_info': adjusted_mutual_info
            })
        except Exception as e:
            print(f"Error calculating supervised metrics: {e}")
            result.update({
                'homogeneity': 0, 'completeness': 0, 'v_measure': 0,
                'adjusted_rand': 0, 'adjusted_mutual_info': 0
            })
    else:
        result.update({
            'homogeneity': np.nan, 'completeness': np.nan, 'v_measure': np.nan,
            'adjusted_rand': np.nan, 'adjusted_mutual_info': np.nan
        })

    return result


def visualizing_results(eps_values, df_results):
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Silhouette Score
    axes[0, 0].plot(eps_values, df_results['silhouette'],
                    'o-', linewidth=2, markersize=8)
    axes[0, 0].set_title('Silhouette Score (higher = better)',
                         fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Eps')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)

    # Number of clusters
    axes[0, 1].plot(eps_values, df_results['n_clusters'], 'o-',
                    color='orange', linewidth=2, markersize=8)
    axes[0, 1].set_title('Number of Clusters', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Eps')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].grid(True, alpha=0.3)

    # Homogeneity
    axes[1, 0].plot(eps_values, df_results['homogeneity'], 'o-',
                    color='green', linewidth=2, markersize=8)
    axes[1, 0].set_title('Homogeneity Score (higher = better)',
                         fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Eps')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].grid(True, alpha=0.3)

    # V-measure
    axes[1, 1].plot(eps_values, df_results['v_measure'], 'o-',
                    color='purple', linewidth=2, markersize=8)
    axes[1, 1].set_title('V-measure Score (higher = better)',
                         fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Eps')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = out_dir / "dbscan_parameters.png"
    plt.savefig(out)

    # Find best configuration
    best_idx = df_results['v_measure'].idxmax()
    best_eps = eps_values[best_idx]
    print(f"Best configuration found:")
    print(f"   • eps = {best_eps}")
    print(f"   • V-measure = {df_results.iloc[best_idx]['v_measure']:.3f}")
    print(f"   • Homogeneity = {df_results.iloc[best_idx]['homogeneity']:.3f}")
    print(f"   • Clusters = {df_results.iloc[best_idx]['n_clusters']}")
    print(
        f"   • Noise = {df_results.iloc[best_idx]['n_noise']} ({df_results.iloc[best_idx]['noise_ratio']:.1%})")


def test_dbscan_params(data, labels_true=None):
    # Test different eps values
    eps_values = [0.30, 0.35, 0.40, 0.45, 0.50, 0.51, 0.52, 0.53, 0.54,
                  0.55, 0.60]
    results = []

    print(" Testing in progress...")
    for eps in eps_values:
        # Clustering with this eps
        db_test = DBSCAN(eps=eps, min_samples=2, metric="cosine")
        labels_test = db_test.fit_predict(data)

        # Evaluate with supervised metrics
        metrics = evaluate_clustering(
            data, labels_test, labels_true, f"eps={eps}")
        results.append(metrics)
        print(
            f"eps={eps:.2f} → {metrics['n_clusters']} clusters, {metrics['n_noise']} noise")
        print(
            f"   Silhouette: {metrics['silhouette']:.3f}, Homogeneity: {metrics['homogeneity']:.3f}")

    # Create DataFrame for analysis
    df_results = pd.DataFrame(results)
    print(f" Results (unsupervised metrics):")
    print(df_results[['method', 'silhouette', 'n_clusters',
          'n_noise', 'noise_ratio']].round(4))

    print(f" Results (homogeneity metrics):")
    print(df_results[['method', 'homogeneity', 'completeness',
          'v_measure', 'adjusted_rand']].round(4))

    visualizing_results(eps_values=eps_values, df_results=df_results)


def final_dbscan(best_eps=0.50, embeddings=[], sentences=[]):

    # Apply final clustering
    db_final = DBSCAN(eps=best_eps, min_samples=2, metric="cosine")
    labels_final = db_final.fit_predict(embeddings)

    # Dimensionality reduction for visualization
    pca = PCA(n_components=2, random_state=53)
    reduced = pca.fit_transform(embeddings)

    # Visualize clusters
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        reduced[:, 0], reduced[:, 1], c=labels_final, s=100, alpha=0.8, cmap='tab10')
    plt.title(
        f"Final Semantic Clustering (eps={best_eps})", fontsize=16, fontweight='bold')
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")

    # Annotate sentences
    # for i, txt in enumerate(sentences):
    #     short_txt = shorten(txt, width=30, placeholder="…")
    #     plt.annotate(short_txt, (reduced[i, 0] + 0.02, reduced[i, 1] + 0.02),
    #                  fontsize=8, alpha=0.8)

    plt.tight_layout()
    # plt.show()

    # Analyze found clusters
    groups_final = collections.defaultdict(list)
    for i, lab in enumerate(labels_final):
        groups_final[lab].append(sentences[i])

    print(f"Final clusters found:", len(groups_final))
    for lab, items in sorted(groups_final.items(), key=lambda x: (x[0] == -1, x[0])):
        header = f"Cluster {lab}" if lab != -1 else "Noise (-1)"
        print(f"\n{header} ({len(items)} items):")
        for item in items:
            print(f"  • {item}")
