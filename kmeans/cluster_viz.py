"""
Cluster Visualization Module

Creates 2D and 3D visualizations of clustering results using UMAP
for dimensionality reduction.
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import umap
from pathlib import Path
from typing import Optional
from collections import Counter


def create_2d_visualization(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int,
    output_path: Path,
    n_neighbors: int = 15,
    metric: str = 'cosine',
    min_dist: float = 0.1
) -> None:
    # Create 2D UMAP visualization of clusters.
    # Fit UMAP
    reducer = umap.UMAP(
        n_components=2,
        random_state=42,
        n_neighbors=n_neighbors,
        metric=metric,
        min_dist=min_dist
    )
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Choose colormap based on k
    if k <= 10:
        cmap = 'tab10'
    elif k <= 20:
        cmap = 'tab20'
    else:
        cmap = 'viridis'
    
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=labels,
        cmap=cmap,
        alpha=0.6,
        s=30,
        edgecolors='black',
        linewidth=0.5
    )
    
    plt.colorbar(scatter, label='Cluster ID')
    plt.title(f'Proverbs Clusters 2D (k={k})\nUMAP: n_neighbors={n_neighbors}, metric={metric}, min_dist={min_dist}')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_3d_visualization(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int,
    output_path: Path,
    n_neighbors: int = 15,
    metric: str = 'cosine',
    min_dist: float = 0.1
) -> None:
    # Create interactive 3D UMAP visualization of clusters using Plotly.
    # Fit UMAP
    reducer = umap.UMAP(
        n_components=3,
        random_state=42,
        n_neighbors=n_neighbors,
        metric=metric,
        min_dist=min_dist
    )
    embeddings_3d = reducer.fit_transform(embeddings)
    
    # Create Plotly 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=embeddings_3d[:, 0],
        y=embeddings_3d[:, 1],
        z=embeddings_3d[:, 2],
        mode='markers',
        marker=dict(
            size=4,
            color=labels,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Cluster ID"),
            opacity=0.7,
            line=dict(color='black', width=0.5)
        ),
        text=[f'Cluster {label}' for label in labels],
        hovertemplate='<b>%{text}</b><br>UMAP1: %{x:.2f}<br>UMAP2: %{y:.2f}<br>UMAP3: %{z:.2f}<extra></extra>'
    )])
    
    fig.update_layout(
        title=f'Proverbs Clusters 3D (k={k})<br>UMAP: n_neighbors={n_neighbors}, metric={metric}, min_dist={min_dist}',
        scene=dict(
            xaxis_title='UMAP 1',
            yaxis_title='UMAP 2',
            zaxis_title='UMAP 3'
        ),
        width=1200,
        height=900
    )
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)


def create_cluster_distribution_plot(
    labels: np.ndarray,
    k: int,
    output_path: Path
) -> None:
    # Create histogram showing the distribution of cluster sizes.
    # Count points in each cluster
    cluster_counts = Counter(labels)
    cluster_ids = sorted(cluster_counts.keys())
    sizes = [cluster_counts[cid] for cid in cluster_ids]
    
    # Calculate percentages
    total_points = len(labels)
    percentages = [size / total_points * 100 for size in sizes]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Bar chart with counts
    bars = ax1.bar(
        range(len(cluster_ids)),
        sizes,
        color='steelblue',
        alpha=0.7,
        edgecolor='black'
    )
    ax1.set_xlabel('Cluster ID')
    ax1.set_ylabel('Number of Verses')
    ax1.set_title('Cluster Size Distribution (Counts)')
    ax1.set_xticks(range(len(cluster_ids)))
    ax1.set_xticklabels(cluster_ids, rotation=45 if k > 20 else 0)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars (only if k is small enough)
    if k <= 25:
        for i, (bar, size) in enumerate(zip(bars, sizes)):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{size}\n({percentages[i]:.1f}%)',
                ha='center',
                va='bottom',
                fontsize=8
            )
    
    # Right plot: Histogram showing distribution
    ax2.hist(
        sizes,
        bins=min(20, len(set(sizes))),
        color='coral',
        alpha=0.7,
        edgecolor='black'
    )
    ax2.set_xlabel('Cluster Size')
    ax2.set_ylabel('Frequency (Number of Clusters)')
    ax2.set_title('Distribution of Cluster Sizes')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add statistics text
    stats_text = f'Total verses: {total_points}\n'
    stats_text += f'Number of clusters: {len(cluster_ids)}\n'
    stats_text += f'Mean size: {np.mean(sizes):.1f}\n'
    stats_text += f'Std size: {np.std(sizes):.1f}\n'
    stats_text += f'Min size: {min(sizes)}\n'
    stats_text += f'Max size: {max(sizes)}\n'
    stats_text += f'Size ratio (max/min): {max(sizes) / min(sizes):.2f}'
    
    fig.text(
        0.5, 0.02,
        stats_text,
        ha='center',
        fontsize=9,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    # Overall title
    fig.suptitle(f'Cluster Size Distribution (k={k})', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_all_visualizations(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int,
    output_dir: Path,
    create_2d: bool = True,
    create_3d: bool = True,
    create_distribution: bool = True
) -> None:
    # Create all visualization types for a clustering result.
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if create_2d:
        create_2d_visualization(
            embeddings, labels, k,
            output_dir / 'clusters_2d.png'
        )
    
    if create_3d:
        create_3d_visualization(
            embeddings, labels, k,
            output_dir / 'clusters_3d.html'
        )
    
    if create_distribution:
        create_cluster_distribution_plot(
            labels, k,
            output_dir / 'cluster_distribution.png'
        )


