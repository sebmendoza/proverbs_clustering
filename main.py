"""
Proverbs Analysis - Main Entry Point

This script provides entry points for various analysis methods:
- K-means clustering (legacy and new pipeline)
- Community graph analysis
- DBSCAN clustering
- Hierarchical clustering (dendrograms)
"""

import logging
import numpy as np
from embeddings import get_or_create_embeddings, embeddings_dict_to_array
from kmeans.clustering_pipeline import run_clustering_pipeline
from hierarchical.hierarchical_clustering import run_hierarchical_clustering, compare_linkage_methods
from utils import setup_logging
from community_graph.community_graph import create_community_graph
logger = logging.getLogger(__name__)


def run_new_clustering_pipeline(embeddings, verse_refs, k_range=(10, 41),
                                title_backend='ollama', title_model='llama3'):
    logger.info("="*80)
    logger.info("RUNNING NEW CLUSTERING PIPELINE")
    logger.info("="*80)
    logger.info(f"This will run K-means for k={k_range[0]} to {k_range[1]-1}")
    logger.info(f"Title generation: {title_backend} ({title_model})")
    logger.info("Each clustering will:")
    logger.info("  - Compute quality metrics")
    logger.info("  - Generate 2D and 3D visualizations")
    logger.info("  - Generate 1, 3, and 5-word cluster titles using LLM")
    logger.info("  - Save results to experiments/ directory")
    logger.info("This may take several minutes depending on your hardware...")
    logger.info("="*80)

    experiment_dir = run_clustering_pipeline(
        embeddings=embeddings,
        verse_refs=verse_refs,
        k_range=k_range,
        output_dir="experiments",
        generate_titles=True,
        create_visualizations=True,
        title_backend=title_backend,
        title_model=title_model
    )

    logger.info(f"Experiment complete! Results saved to: {experiment_dir}")
    logger.info("To analyze results, run:")
    logger.info(f"  python cluster_analysis.py {experiment_dir}")
    logger.info("Or in Python:")
    logger.info("  from cluster_analysis import compare_k_values")
    logger.info(f"  compare_k_values('{experiment_dir}')")

    return experiment_dir


def run_hierarchical_pipeline(embeddings, verse_refs, method='average', metric='cosine',
                              k_values=None, generate_titles=False,
                              title_backend='ollama', title_model='llama3'):
    # Run hierarchical clustering pipeline with dendrogram visualizations.
    logger.info("="*80)
    logger.info("RUNNING HIERARCHICAL CLUSTERING PIPELINE")
    logger.info("="*80)
    logger.info("This will:")
    logger.info("  - Compute agglomerative hierarchical clustering")
    logger.info("  - Create static and interactive dendrogram visualizations")
    logger.info(
        f"  - Extract flat clusters at k = {k_values or [5, 10, 15, 20, 25]}")
    logger.info(
        "  - Compute cophenetic correlation to assess dendrogram quality")
    if generate_titles:
        logger.info(
            f"  - Generate cluster titles using {title_backend} ({title_model})")
    logger.info("="*80)

    experiment_dir = run_hierarchical_clustering(
        embeddings=embeddings,
        verse_refs=verse_refs,
        output_dir="experiments",
        method=method,
        metric=metric,
        k_values=k_values,
        generate_titles=generate_titles,
        title_backend=title_backend,
        title_model=title_model
    )

    logger.info(f"Experiment complete! Results saved to: {experiment_dir}")
    logger.info("Key files:")
    logger.info("  - dendrogram.png: Truncated dendrogram overview")
    logger.info("  - dendrogram_full.png: Full dendrogram with all verses")
    logger.info(
        "  - dendrogram_interactive.html: Interactive Plotly visualization")
    logger.info("  - clusters_k{N}.json: Flat clusters at various levels")

    return experiment_dir


if __name__ == "__main__":
    setup_logging()

    logger.info("Loading embeddings...")
    embeddings_dict = get_or_create_embeddings()
    embeddings, verse_refs = embeddings_dict_to_array(embeddings_dict)
    arr = np.array(embeddings)
    logger.info(
        f"Loaded {len(arr)} verses with {arr.shape[1]}-dimensional embeddings")

    # Choose which analysis to run:

    # Option 1: NEW CLUSTERING PIPELINE (K-means)
    # Runs k=10 to k=40 with quality metrics, visualizations, and LLM-generated titles
    # run_new_clustering_pipeline(arr, verse_refs, k_range=(10, 41))

    # Option 2: HIERARCHICAL CLUSTERING (Recommended for theme exploration)
    # Creates dendrograms showing thematic hierarchy at multiple levels
    # run_hierarchical_pipeline(
    #     arr, verse_refs,
    #     method='ward',
    #     metric='euclidean',
    #     k_values=[5, 10, 15, 20, 25, 40, 50, 60],
    #     generate_titles=False
    # )

    # Option 3: LEGACY KMEANS EXPERIMENTS
    # Runs k=2 to k=29 with basic metrics and visualizations
    # run_kmeans_experiments(arr, verse_refs)

    # Option 4: COMMUNITY GRAPH ANALYSIS
    create_community_graph(arr, verse_refs)

    # Option 5: COMPARE LINKAGE METHODS
    # Finds the best linkage method/metric combination
    # compare_linkage_methods(arr, verse_refs)
