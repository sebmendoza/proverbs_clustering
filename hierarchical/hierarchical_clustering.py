"""
Hierarchical Clustering Module

Performs agglomerative hierarchical clustering on Proverbs verse embeddings
to reveal thematic structure at multiple levels of granularity.
"""

import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict, Optional
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist

from utils import getDataFromJson, organize_clusters_data

logger = logging.getLogger(__name__)


def compute_linkage(
    embeddings: np.ndarray,
    method: str = 'average',
    metric: str = 'cosine'
) -> Tuple[np.ndarray, np.ndarray]:
    # Compute hierarchical clustering linkage matrix.
    # Hierarchical clustering builds a tree (dendrogram) by iteratively merging closest clusters.
    # The linkage matrix Z encodes this tree structure: each row represents one merge.

    # Compute pairwise distances between all data points
    # pdist returns a condensed distance matrix (only upper triangle, saves memory)
    if method == 'ward':
        # Ward linkage minimizes variance within clusters, but requires euclidean distance
        # This is important: Ward is mathematically defined only for euclidean space
        distance_matrix = pdist(embeddings, metric='euclidean')
    else:
        # Other linkage methods (average, complete, single) can use any distance metric
        distance_matrix = pdist(embeddings, metric=metric)

    # Compute linkage matrix Z: shape (n-1, 4) where n = number of samples
    # Each row: [cluster_i, cluster_j, distance, num_points_in_new_cluster]
    Z = linkage(distance_matrix, method=method)

    return Z, distance_matrix


def compute_hierarchical_metrics(
    embeddings: np.ndarray,
    linkage_matrix: np.ndarray,
    distance_matrix: np.ndarray
) -> Dict:
    # Compute quality metrics for hierarchical clustering.

    # Cophenetic correlation: measures how well the dendrogram preserves original pairwise distances
    # Range: -1 to 1, closer to 1 is better. High values mean the tree structure accurately
    # represents the original data relationships. This is the key metric for hierarchical clustering quality.
    cophenetic_corr, _ = cophenet(linkage_matrix, distance_matrix)

    # Merge heights: the distance at which clusters were merged (column 2 of linkage matrix)
    # Lower heights = clusters merged early (very similar), higher = merged late (dissimilar)
    merge_heights = linkage_matrix[:, 2]

    metrics = {
        'cophenetic_correlation': float(cophenetic_corr),
        'num_samples': len(embeddings),
        'num_merges': len(linkage_matrix),
        'min_merge_height': float(merge_heights.min()),
        'max_merge_height': float(merge_heights.max()),
        'mean_merge_height': float(merge_heights.mean()),
        'std_merge_height': float(merge_heights.std())
    }

    return metrics


def extract_clusters_at_level(
    linkage_matrix: np.ndarray,
    n_clusters: int = None,
    height: float = None
) -> np.ndarray:
    # Extract flat clusters by cutting the dendrogram at a specific level.
    # A dendrogram is a tree - "cutting" it horizontally gives you a flat partition.
    # This converts the hierarchical structure into discrete clusters (like K-means output).

    if n_clusters is not None:
        # Cut to get exactly n_clusters (e.g., "give me 10 clusters")
        labels = fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')
    elif height is not None:
        # Cut at a specific distance threshold (e.g., "merge everything closer than 0.5")
        labels = fcluster(linkage_matrix, t=height, criterion='distance')
    else:
        raise ValueError("Either n_clusters or height must be specified")

    # fcluster returns 1-indexed labels, convert to 0-indexed for consistency
    return labels - 1


def create_dendrogram_visualization(
    linkage_matrix: np.ndarray,
    output_path: Path,
    verse_refs: List[Tuple[str, str]] = None,
    color_threshold: float = None,
    truncate_mode: str = 'lastp',
    p: int = 30,
    title: str = 'Proverbs Hierarchical Clustering Dendrogram'
) -> Dict:
    # Create static dendrogram visualization using matplotlib.
    fig, ax = plt.subplots(figsize=(16, 10))

    # Prepare labels if provided
    labels = None
    if verse_refs and truncate_mode is None:
        labels = [f"{ch}:{v}" for ch, v in verse_refs]

    # Set default color threshold based on dendrogram heights
    # Color threshold determines which branches get colored differently in the visualization
    # Branches merged below this threshold get distinct colors (showing major clusters)
    if color_threshold is None:
        # Use median merge height for coloring - splits the tree roughly in half
        color_threshold = np.median(linkage_matrix[:, 2])

    # Create dendrogram
    dendro_data = dendrogram(
        linkage_matrix,
        ax=ax,
        labels=labels,
        truncate_mode=truncate_mode,
        p=p,
        color_threshold=color_threshold,
        leaf_rotation=90,
        leaf_font_size=8,
        above_threshold_color='gray'
    )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Verse Clusters' if truncate_mode else 'Verses', fontsize=12)
    ax.set_ylabel('Distance', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    # Add horizontal line at color threshold
    ax.axhline(y=color_threshold, color='red', linestyle='--',
               alpha=0.5, label=f'Color threshold: {color_threshold:.3f}')
    ax.legend(loc='upper right')

    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return dendro_data


def create_interactive_dendrogram(
    linkage_matrix: np.ndarray,
    output_path: Path,
    verse_refs: List[Tuple[str, str]] = None,
    verses_dict: Dict = None,
    color_threshold: float = None,
    title: str = 'Proverbs Hierarchical Clustering (Interactive)'
) -> None:
    # Create interactive dendrogram visualization using Plotly.
    # Interactive dendrograms let you zoom, pan, and hover for details - crucial for large datasets.

    # Get dendrogram coordinates from scipy (no_plot=True means compute but don't render)
    # This gives us the tree structure data to manually plot with Plotly
    dendro_data = dendrogram(
        linkage_matrix,
        no_plot=True,
        color_threshold=color_threshold or np.median(linkage_matrix[:, 2])
    )

    # Extract coordinates: scipy provides these for drawing the tree
    icoord = np.array(dendro_data['icoord'])  # x-coordinates of tree branches
    # y-coordinates (heights/distances)
    dcoord = np.array(dendro_data['dcoord'])
    colors = dendro_data['color_list']  # Color assigned to each branch
    # Order of leaf nodes (data points) at bottom
    leaves = dendro_data['leaves']

    # Create traces for dendrogram lines
    traces = []

    # Color mapping for branches
    color_map = {
        'C0': '#1f77b4', 'C1': '#ff7f0e', 'C2': '#2ca02c', 'C3': '#d62728',
        'C4': '#9467bd', 'C5': '#8c564b', 'C6': '#e377c2', 'C7': '#7f7f7f',
        'C8': '#bcbd22', 'C9': '#17becf', 'g': '#808080'
    }

    # Add each branch as a trace
    for i, (xs, ys, color) in enumerate(zip(icoord, dcoord, colors)):
        plot_color = color_map.get(color, '#808080')
        traces.append(go.Scatter(
            x=xs,
            y=ys,
            mode='lines',
            line=dict(color=plot_color, width=1.5),
            hoverinfo='skip',
            showlegend=False
        ))

    # Add leaf nodes with hover information
    # Leaf nodes are the actual data points at the bottom of the dendrogram
    # Spacing them evenly: each leaf gets 10 units of space, starting at x=5
    leaf_x = list(range(5, 10 * len(leaves), 10))
    leaf_y = [0] * len(leaves)  # All leaves at y=0 (bottom of tree)

    # Prepare hover text
    hover_texts = []
    leaf_labels = []
    for leaf_idx in leaves:
        if verse_refs and leaf_idx < len(verse_refs):
            chapter, verse = verse_refs[leaf_idx]
            label = f"Prov {chapter}:{verse}"
            hover_text = f"<b>{label}</b>"
            if verses_dict and chapter in verses_dict and verse in verses_dict[chapter]:
                text = verses_dict[chapter][verse]
                # Truncate long text
                if len(text) > 100:
                    text = text[:100] + "..."
                hover_text += f"<br><br>{text}"
            hover_texts.append(hover_text)
            leaf_labels.append(label)
        else:
            hover_texts.append(f"Sample {leaf_idx}")
            leaf_labels.append(str(leaf_idx))

    # Create colorful leaf markers using HSL color space
    # HSL (Hue, Saturation, Lightness) makes it easy to generate evenly-spaced colors
    # We cycle through the hue wheel (0-360 degrees) to get distinct colors
    n_leaves = len(leaves)
    leaf_colors = [
        f'hsl({i * 360 / n_leaves}, 70%, 50%)' for i in range(n_leaves)]

    traces.append(go.Scatter(
        x=leaf_x,
        y=leaf_y,
        mode='markers',
        marker=dict(
            size=8,
            color=leaf_colors,
            line=dict(color='black', width=0.5)
        ),
        text=hover_texts,
        hoverinfo='text',
        name='Verses'
    ))

    # Create figure
    fig = go.Figure(data=traces)

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        xaxis=dict(
            title='Verses',
            showticklabels=False,
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            title='Distance',
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=True
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=1400,
        height=800,
        hovermode='closest'
    )

    # Add annotation for threshold
    if color_threshold:
        fig.add_hline(
            y=color_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Color threshold: {color_threshold:.3f}",
            annotation_position="top right"
        )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)


def run_hierarchical_clustering(
    embeddings: np.ndarray,
    verse_refs: List[Tuple[str, str]],
    output_dir: str = "experiments",
    method: str = 'average',
    metric: str = 'cosine',
    k_values: List[int] = None,
    generate_titles: bool = False,
    title_backend: str = 'ollama',
    title_model: str = 'llama3'
) -> Path:
    # Run complete hierarchical clustering pipeline.
    if k_values is None:
        k_values = [5, 10, 15, 20, 25]

    # Create experiment directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path(output_dir) / f"hierarchical_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*70)
    logger.info("Starting Hierarchical Clustering Pipeline")
    logger.info("="*70)
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Linkage method: {method}")
    logger.info(f"Distance metric: {metric}")
    logger.info(f"Total verses: {len(verse_refs)}")
    logger.info(f"K values for flat clustering: {k_values}")
    logger.info(f"Output directory: {experiment_dir}")
    logger.info("="*70)

    # Load verse data
    verses_dict = getDataFromJson()

    # Compute linkage
    logger.info("Computing hierarchical clustering linkage...")
    linkage_matrix, distance_matrix = compute_linkage(
        embeddings, method, metric)
    logger.debug(f"Linkage matrix shape: {linkage_matrix.shape}")

    # Compute metrics
    logger.info("Computing quality metrics...")
    metrics = compute_hierarchical_metrics(
        embeddings, linkage_matrix, distance_matrix)
    logger.info(
        f"Cophenetic correlation: {metrics['cophenetic_correlation']:.4f}")

    # Save metrics
    metrics_file = experiment_dir / "metrics.json"
    metrics['method'] = method
    metrics['metric'] = metric
    metrics['timestamp'] = timestamp
    try:
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to: {metrics_file}")
    except PermissionError:
        raise PermissionError(
            f"Permission denied writing to {metrics_file}. Check directory permissions."
        )
    except OSError as e:
        raise OSError(f"Error writing to {metrics_file}: {e}")

    # Create visualizations
    logger.info("Creating dendrogram visualizations...")

    # Static dendrogram (truncated for readability)
    dendro_path = experiment_dir / "dendrogram.png"
    create_dendrogram_visualization(
        linkage_matrix,
        dendro_path,
        verse_refs=verse_refs,
        truncate_mode='lastp',
        p=50,
        title=f'Proverbs Hierarchical Clustering\n(Method: {method}, Metric: {metric})'
    )
    logger.info(f"Saved static dendrogram to: {dendro_path}")

    # Full dendrogram (may be large)
    full_dendro_path = experiment_dir / "dendrogram_full.png"
    create_dendrogram_visualization(
        linkage_matrix,
        full_dendro_path,
        verse_refs=verse_refs,
        truncate_mode=None,
        title=f'Proverbs Full Dendrogram\n(Method: {method}, Metric: {metric})'
    )
    logger.info(f"Saved full dendrogram to: {full_dendro_path}")

    # Interactive dendrogram
    interactive_path = experiment_dir / "dendrogram_interactive.html"
    create_interactive_dendrogram(
        linkage_matrix,
        interactive_path,
        verse_refs=verse_refs,
        verses_dict=verses_dict,
        title=f'Proverbs Hierarchical Clustering (Interactive)<br>Method: {method}, Metric: {metric}'
    )
    logger.info(f"Saved interactive dendrogram to: {interactive_path}")

    # Extract flat clusters at different k values
    # The beauty of hierarchical clustering: you can extract any number of clusters
    # from the same tree without recomputing. K-means requires separate runs for each k.
    logger.info(f"Extracting flat clusters at k = {k_values}...")

    all_clusters_results = {}

    for k in k_values:
        logger.info(f"Processing k={k}...")
        # Cut the dendrogram to get exactly k clusters
        labels = extract_clusters_at_level(linkage_matrix, n_clusters=k)

        # Organize cluster data
        clusters_data = organize_clusters_data(
            labels, verse_refs, verses_dict, k)

        # Generate titles if requested
        if generate_titles:
            logger.info(f"Generating cluster titles...")
            from kmeans.cluster_titles import generate_titles_for_cluster, initialize_title_generator
            initialize_title_generator(
                backend=title_backend, model=title_model)

            for cluster_id in range(k):
                cluster_key = f"cluster_{cluster_id}"
                verses_texts = [v["text"]
                                for v in clusters_data[cluster_key]["verses"]]

                if len(verses_texts) > 0:
                    titles = generate_titles_for_cluster(verses_texts)
                    clusters_data[cluster_key]["titles"] = titles
                    logger.debug(
                        f"      Cluster {cluster_id}: {titles.get('title_3words', 'N/A')}")

        # Compute cluster sizes
        cluster_sizes = [
            clusters_data[f"cluster_{i}"]["num_verses"] for i in range(k)]

        result = {
            "k": k,
            "method": method,
            "metric": metric,
            "timestamp": timestamp,
            "cluster_sizes": cluster_sizes,
            "clusters": clusters_data
        }

        # Save clusters JSON
        clusters_file = experiment_dir / f"clusters_k{k}.json"
        try:
            with open(clusters_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved clusters to: {clusters_file}")
        except PermissionError:
            raise PermissionError(
                f"Permission denied writing to {clusters_file}. Check directory permissions."
            )
        except OSError as e:
            raise OSError(f"Error writing to {clusters_file}: {e}")

        # Size ratio: max/min cluster sizes. High ratio = very unbalanced clusters (some huge, some tiny)
        # This is a quality indicator - balanced clusters are often better
        all_clusters_results[f"k_{k}"] = {
            "num_clusters": k,
            "cluster_sizes": cluster_sizes,
            "size_ratio": max(cluster_sizes) / min(cluster_sizes) if min(cluster_sizes) > 0 else float('inf')
        }

    # Save summary
    summary = {
        "experiment_type": "hierarchical_clustering",
        "timestamp": timestamp,
        "method": method,
        "metric": metric,
        "total_verses": len(verse_refs),
        "metrics": metrics,
        "k_results": all_clusters_results
    }

    summary_file = experiment_dir / "summary.json"
    try:
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
    except PermissionError:
        raise PermissionError(
            f"Permission denied writing to {summary_file}. Check directory permissions."
        )
    except OSError as e:
        raise OSError(f"Error writing to {summary_file}: {e}")

    logger.info("="*70)
    logger.info("HIERARCHICAL CLUSTERING COMPLETE")
    logger.info("="*70)
    logger.info(f"Output directory: {experiment_dir}")
    logger.info("Files created:")
    logger.info("  - dendrogram.png (truncated view)")
    logger.info("  - dendrogram_full.png (all verses)")
    logger.info("  - dendrogram_interactive.html (Plotly)")
    logger.info("  - metrics.json")
    logger.info("  - summary.json")
    for k in k_values:
        logger.info(f"  - clusters_k{k}.json")
    logger.info(
        f"Cophenetic correlation: {metrics['cophenetic_correlation']:.4f}")
    logger.info(
        "  (values closer to 1.0 indicate the dendrogram well represents distances)")
    logger.info("="*70)

    return experiment_dir


def compare_linkage_methods(
    embeddings: np.ndarray,
    verse_refs: List[Tuple[str, str]],
    output_dir: str = "experiments"
) -> Dict:
    # Compare different linkage methods and metrics.
    # Linkage methods determine how to measure distance between clusters:
    # - average: mean distance between all pairs (balanced, most common)
    # - complete: maximum distance (tends to create compact clusters)
    # - single: minimum distance (tends to create elongated clusters, chaining problem)
    # - ward: minimizes variance (only works with euclidean, creates spherical clusters)
    methods = ['average', 'complete', 'single']
    # Distance metrics between individual points
    metrics = ['cosine', 'euclidean']

    results = {}

    logger.info("="*70)
    logger.info("Comparing Linkage Methods")
    logger.info("="*70)

    for method in methods:
        for metric in metrics:
            # Ward linkage is mathematically defined only for euclidean distance
            # It minimizes within-cluster variance, which requires euclidean geometry
            if method == 'ward' and metric != 'euclidean':
                continue

            key = f"{method}_{metric}"
            logger.info(f"Testing {method} linkage with {metric} distance...")

            try:
                Z, dist_matrix = compute_linkage(embeddings, method, metric)
                metrics_dict = compute_hierarchical_metrics(
                    embeddings, Z, dist_matrix)
                results[key] = {
                    'method': method,
                    'metric': metric,
                    'cophenetic_correlation': metrics_dict['cophenetic_correlation']
                }
                logger.info(
                    f"  Cophenetic correlation: {metrics_dict['cophenetic_correlation']:.4f}")
            except Exception as e:
                logger.error(f"  Error: {e}")
                results[key] = {'method': method,
                                'metric': metric, 'error': str(e)}

    # Find best method
    valid_results = {k: v for k,
                     v in results.items() if 'cophenetic_correlation' in v}
    if valid_results:
        best_key = max(
            valid_results, key=lambda k: valid_results[k]['cophenetic_correlation'])
        best = valid_results[best_key]
        logger.info(
            f"Best method: {best['method']} with {best['metric']} distance")
        logger.info(
            f"Cophenetic correlation: {best['cophenetic_correlation']:.4f}")

    # Save comparison results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_dir = Path(output_dir) / f"hierarchical_comparison_{timestamp}"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    try:
        with open(comparison_dir / "comparison_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
    except PermissionError:
        raise PermissionError(
            f"Permission denied writing to {comparison_dir / 'comparison_results.json'}. Check directory permissions."
        )
    except OSError as e:
        raise OSError(
            f"Error writing to {comparison_dir / 'comparison_results.json'}: {e}")

    logger.info(f"Saved comparison to: {comparison_dir}")

    return results


if __name__ == "__main__":
    # Standalone execution for testing.
    from embeddings import get_or_create_embeddings
    from main import embeddings_dict_to_array

    logger.info("Loading embeddings...")
    embeddings_dict = get_or_create_embeddings()
    embeddings, verse_refs = embeddings_dict_to_array(embeddings_dict)
    arr = np.array(embeddings)
    logger.info(
        f"Loaded {len(arr)} verses with {arr.shape[1]}-dimensional embeddings")

    # Run hierarchical clustering
    run_hierarchical_clustering(
        arr,
        verse_refs,
        method='average',
        metric='cosine',
        k_values=[5, 10, 15, 20, 25],
        generate_titles=False
    )
