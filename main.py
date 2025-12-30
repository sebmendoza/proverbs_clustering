"""
Proverbs Analysis - Main Entry Point

This script provides entry points for various analysis methods:
- K-means clustering (legacy and new pipeline)
- Community graph analysis
- DBSCAN clustering
- Hierarchical clustering (dendrograms)
"""

import numpy as np
from embeddings import get_or_create_embeddings, embeddings_dict_to_array
from kmeans.kmeans import run_kmeans_experiments
from community_graph.community_graph import create_community_graph
from kmeans.clustering_pipeline import run_clustering_pipeline
from hierarchical.hierarchical_clustering import run_hierarchical_clustering, compare_linkage_methods


def run_new_clustering_pipeline(embeddings, verse_refs, k_range=(10, 41),
                                title_backend='ollama', title_model='llama3'):
    print("\n" + "="*80)
    print("RUNNING NEW CLUSTERING PIPELINE")
    print("="*80)
    print(f"This will run K-means for k={k_range[0]} to {k_range[1]-1}")
    print(f"Title generation: {title_backend} ({title_model})")
    print("Each clustering will:")
    print("  - Compute quality metrics")
    print("  - Generate 2D and 3D visualizations")
    print("  - Generate 1, 3, and 5-word cluster titles using LLM")
    print("  - Save results to experiments/ directory")
    print("\nThis may take several minutes depending on your hardware...")
    print("="*80 + "\n")

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

    print(f"\nExperiment complete! Results saved to: {experiment_dir}")
    print(f"\nTo analyze results, run:")
    print(f"  python cluster_analysis.py {experiment_dir}")
    print(f"\nOr in Python:")
    print(f"  from cluster_analysis import compare_k_values")
    print(f"  compare_k_values('{experiment_dir}')")

    return experiment_dir


def run_hierarchical_pipeline(embeddings, verse_refs, method='average', metric='cosine',
                              k_values=None, generate_titles=False,
                              title_backend='ollama', title_model='llama3'):
    # Run hierarchical clustering pipeline with dendrogram visualizations.
    print("\n" + "="*80)
    print("RUNNING HIERARCHICAL CLUSTERING PIPELINE")
    print("="*80)
    print("This will:")
    print("  - Compute agglomerative hierarchical clustering")
    print("  - Create static and interactive dendrogram visualizations")
    print(
        f"  - Extract flat clusters at k = {k_values or [5, 10, 15, 20, 25]}")
    print("  - Compute cophenetic correlation to assess dendrogram quality")
    if generate_titles:
        print(
            f"  - Generate cluster titles using {title_backend} ({title_model})")
    print("="*80 + "\n")

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

    print(f"\nExperiment complete! Results saved to: {experiment_dir}")
    print(f"\nKey files:")
    print(f"  - dendrogram.png: Truncated dendrogram overview")
    print(f"  - dendrogram_full.png: Full dendrogram with all verses")
    print(f"  - dendrogram_interactive.html: Interactive Plotly visualization")
    print(f"  - clusters_k{{N}}.json: Flat clusters at various levels")

    return experiment_dir


if __name__ == "__main__":
    print("Loading embeddings...")
    embeddings_dict = get_or_create_embeddings()
    embeddings, verse_refs = embeddings_dict_to_array(embeddings_dict)
    arr = np.array(embeddings)
    print(
        f"Loaded {len(arr)} verses with {arr.shape[1]}-dimensional embeddings\n")

    # Choose which analysis to run:

    # Option 1: NEW CLUSTERING PIPELINE (K-means)
    # Runs k=10 to k=40 with quality metrics, visualizations, and LLM-generated titles
    # run_new_clustering_pipeline(arr, verse_refs, k_range=(10, 41))

    # Option 2: HIERARCHICAL CLUSTERING (Recommended for theme exploration)
    # Creates dendrograms showing thematic hierarchy at multiple levels
    run_hierarchical_pipeline(
        arr, verse_refs,
        method='ward',
        metric='euclidean',
        k_values=[5, 10, 15, 20, 25, 40, 50, 60],
        generate_titles=False
    )

    # Option 3: LEGACY KMEANS EXPERIMENTS
    # Runs k=2 to k=29 with basic metrics and visualizations
    # run_kmeans_experiments(arr, verse_refs)

    # Option 4: COMMUNITY GRAPH ANALYSIS
    # create_community_graph(arr, verse_refs)

    # Option 5: COMPARE LINKAGE METHODS
    # Finds the best linkage method/metric combination
    # compare_linkage_methods(arr, verse_refs)
