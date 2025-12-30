"""
Cluster Quality Metrics Module

Computes comprehensive quality metrics for K-means clustering results
and flags potential issues like imbalanced clusters, noise, and mixed topics.
"""

import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, silhouette_samples
from collections import Counter
from typing import Dict, List, Tuple


def compute_cluster_metrics(embeddings: np.ndarray, labels: np.ndarray, k: int) -> Dict:
    # Compute comprehensive quality metrics for clustering results.
    metrics = {}
    
    # Cluster size distribution
    cluster_counts = Counter(labels)
    sizes = list(cluster_counts.values())
    
    metrics['k'] = k
    metrics['total_points'] = len(labels)
    metrics['num_clusters'] = len(cluster_counts)
    
    # Size statistics
    metrics['size_min'] = min(sizes)
    metrics['size_max'] = max(sizes)
    metrics['size_mean'] = np.mean(sizes)
    metrics['size_std'] = np.std(sizes)
    metrics['size_median'] = np.median(sizes)
    metrics['size_ratio'] = max(sizes) / min(sizes) if min(sizes) > 0 else float('inf')
    metrics['coefficient_of_variation'] = metrics['size_std'] / metrics['size_mean'] if metrics['size_mean'] > 0 else 0
    
    # Count small clusters (potential noise)
    metrics['num_tiny_clusters'] = sum(1 for s in sizes if s < 3)
    metrics['num_small_clusters'] = sum(1 for s in sizes if s < 5)
    
    # Global quality scores
    if len(np.unique(labels)) > 1:
        metrics['silhouette_global'] = silhouette_score(embeddings, labels, metric='euclidean')
        metrics['davies_bouldin'] = davies_bouldin_score(embeddings, labels)
        
        # Per-cluster silhouette scores
        silhouette_vals = silhouette_samples(embeddings, labels, metric='euclidean')
        per_cluster_silhouette = {}
        for cluster_id in range(k):
            mask = labels == cluster_id
            if np.sum(mask) > 0:
                per_cluster_silhouette[cluster_id] = float(np.mean(silhouette_vals[mask]))
        
        metrics['silhouette_per_cluster'] = per_cluster_silhouette
        metrics['silhouette_min'] = min(per_cluster_silhouette.values())
        metrics['silhouette_max'] = max(per_cluster_silhouette.values())
        
        # Count poorly separated clusters
        metrics['num_poor_silhouette'] = sum(1 for s in per_cluster_silhouette.values() if s < 0.2)
    else:
        metrics['silhouette_global'] = 0.0
        metrics['davies_bouldin'] = float('inf')
        metrics['silhouette_per_cluster'] = {}
        metrics['silhouette_min'] = 0.0
        metrics['silhouette_max'] = 0.0
        metrics['num_poor_silhouette'] = 0
    
    # Intra-cluster variance (compactness)
    intra_cluster_variance = {}
    for cluster_id in range(k):
        mask = labels == cluster_id
        cluster_points = embeddings[mask]
        if len(cluster_points) > 1:
            center = cluster_points.mean(axis=0)
            variance = np.mean(np.sum((cluster_points - center) ** 2, axis=1))
            intra_cluster_variance[cluster_id] = float(variance)
        else:
            intra_cluster_variance[cluster_id] = 0.0
    
    metrics['intra_cluster_variance'] = intra_cluster_variance
    metrics['variance_mean'] = float(np.mean(list(intra_cluster_variance.values())))
    metrics['variance_std'] = float(np.std(list(intra_cluster_variance.values())))
    
    return metrics


def flag_quality_issues(metrics: Dict, k: int) -> List[str]:
    # Flag quality issues based on metrics and thresholds.
    warnings = []
    
    # Check for small/noise clusters
    if metrics['num_tiny_clusters'] >= 2 and k > 20:
        warnings.append(
            f"HIGH_K_NOISE: {metrics['num_tiny_clusters']} clusters with <3 verses detected (k={k})"
        )
    
    if metrics['num_small_clusters'] >= 3:
        warnings.append(
            f"SMALL_CLUSTERS: {metrics['num_small_clusters']} clusters with <5 verses"
        )
    
    # Check for imbalanced clusters
    if metrics['size_ratio'] > 10:
        warnings.append(
            f"IMBALANCED: Size ratio {metrics['size_ratio']:.1f} "
            f"(min={metrics['size_min']}, max={metrics['size_max']})"
        )
    
    # Check global separation
    if metrics['silhouette_global'] < 0.3:
        warnings.append(
            f"POOR_SEPARATION: Global silhouette {metrics['silhouette_global']:.3f} < 0.3"
        )
    
    # Check for mixed-topic clusters
    if metrics['num_poor_silhouette'] > 0:
        poor_clusters = [
            cid for cid, score in metrics['silhouette_per_cluster'].items() 
            if score < 0.2
        ]
        warnings.append(
            f"MIXED_TOPICS: {metrics['num_poor_silhouette']} cluster(s) with silhouette <0.2: {poor_clusters}"
        )
    
    # Check coefficient of variation (cluster size consistency)
    if metrics['coefficient_of_variation'] > 1.0:
        warnings.append(
            f"HIGH_VARIANCE: Coefficient of variation {metrics['coefficient_of_variation']:.2f} "
            "suggests inconsistent cluster sizes"
        )
    
    return warnings


def get_quality_summary(metrics: Dict, warnings: List[str]) -> str:
    # Generate a human-readable quality summary.
    summary = []
    summary.append(f"Clustering Quality Summary (k={metrics['k']})")
    summary.append("=" * 60)
    summary.append(f"\nCluster Sizes:")
    summary.append(f"  Min: {metrics['size_min']}, Max: {metrics['size_max']}, Mean: {metrics['size_mean']:.1f}")
    summary.append(f"  Std: {metrics['size_std']:.1f}, Ratio: {metrics['size_ratio']:.2f}")
    summary.append(f"\nGlobal Quality:")
    summary.append(f"  Silhouette Score: {metrics['silhouette_global']:.4f}")
    summary.append(f"  Davies-Bouldin Score: {metrics['davies_bouldin']:.4f}")
    summary.append(f"\nPer-Cluster Silhouette:")
    summary.append(f"  Min: {metrics['silhouette_min']:.4f}, Max: {metrics['silhouette_max']:.4f}")
    summary.append(f"  Poorly separated clusters: {metrics['num_poor_silhouette']}")
    
    if warnings:
        summary.append(f"\n⚠️  Quality Warnings ({len(warnings)}):")
        for warning in warnings:
            summary.append(f"  - {warning}")
    else:
        summary.append("\n✓ No quality issues detected")
    
    summary.append("=" * 60)
    return "\n".join(summary)


def compare_k_metrics(metrics_list: List[Dict]) -> Dict:
    # Compare metrics across different k values.
    comparison = {
        'k_values': [m['k'] for m in metrics_list],
        'silhouette_scores': [m['silhouette_global'] for m in metrics_list],
        'davies_bouldin_scores': [m['davies_bouldin'] for m in metrics_list],
        'size_ratios': [m['size_ratio'] for m in metrics_list],
        'num_warnings': []
    }
    
    # Count warnings for each k
    for metrics in metrics_list:
        warnings = flag_quality_issues(metrics, metrics['k'])
        comparison['num_warnings'].append(len(warnings))
    
    # Find best k by silhouette score
    best_idx = np.argmax(comparison['silhouette_scores'])
    comparison['best_k_silhouette'] = comparison['k_values'][best_idx]
    comparison['best_silhouette_score'] = comparison['silhouette_scores'][best_idx]
    
    # Find k with fewest warnings
    min_warnings_idx = np.argmin(comparison['num_warnings'])
    comparison['best_k_warnings'] = comparison['k_values'][min_warnings_idx]
    
    return comparison


