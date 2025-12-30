"""
Clustering Pipeline Module

Main orchestrator for running K-means clustering experiments across multiple k values.
Coordinates quality analysis, visualization, and title generation.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict
from sklearn.cluster import KMeans

from cluster_quality import compute_cluster_metrics, flag_quality_issues, get_quality_summary, compare_k_metrics
from cluster_viz import create_all_visualizations
from cluster_titles import initialize_title_generator
from utils import getDataFromJson


def run_clustering_pipeline(
    embeddings: np.ndarray,
    verse_refs: List[Tuple[str, str]],
    k_range: Tuple[int, int] = (10, 41),
    output_dir: str = "experiments",
    generate_titles: bool = True,
    create_visualizations: bool = True,
    title_backend: str = 'ollama',
    title_model: str = 'llama3'
) -> Path:
    # Run complete clustering pipeline across multiple k values.
    # Create experiment directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path(output_dir) / f"experiment_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Starting Clustering Pipeline Experiment")
    print(f"Timestamp: {timestamp}")
    print(f"K range: {k_range[0]} to {k_range[1]-1}")
    print(f"Total verses: {len(verse_refs)}")
    print(f"Output directory: {experiment_dir}")
    print(f"{'='*70}\n")

    # Load verse data
    verses_dict = getDataFromJson()

    # Initialize title generator if needed (do this once at start)
    if generate_titles:
        print(
            f"Initializing LLM for title generation ({title_backend}: {title_model})...")
        initialize_title_generator(backend=title_backend, model=title_model)
        print("LLM initialized.\n")

    # Store results for comparison
    all_metrics = []

    # Loop through k values
    for k in range(k_range[0], k_range[1]):
        print(f"\n{'='*70}")
        print(f"Processing k={k}")
        print(f"{'='*70}")

        # Run K-means
        print(f"Running K-means clustering...")
        model = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = model.fit_predict(embeddings)

        # Create output directory for this k
        k_dir = experiment_dir / f"k_{k}"
        k_dir.mkdir(exist_ok=True)

        # Compute quality metrics
        print(f"Computing quality metrics...")
        metrics = compute_cluster_metrics(embeddings, labels, k)
        warnings = flag_quality_issues(metrics, k)
        all_metrics.append(metrics)

        # Print quality summary
        summary = get_quality_summary(metrics, warnings)
        print(summary)

        # Organize clusters with verses
        clusters_data = {}
        for cluster_id in range(k):
            cluster_indices = np.where(labels == cluster_id)[0]
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

            clusters_data[f"cluster_{cluster_id}"] = {
                "cluster_id": cluster_id,
                "num_verses": len(cluster_indices),
                "verses": cluster_verses
            }

        # Generate titles if requested
        if generate_titles:
            print(f"Generating cluster titles with LLM...")
            from cluster_titles import generate_titles_for_cluster

            for cluster_id in range(k):
                cluster_key = f"cluster_{cluster_id}"
                verses_texts = [v["text"]
                                for v in clusters_data[cluster_key]["verses"]]

                if len(verses_texts) > 0:
                    titles = generate_titles_for_cluster(verses_texts)
                    clusters_data[cluster_key]["titles"] = titles
                    print(
                        f"  Cluster {cluster_id}: {titles.get('title_3words', 'N/A')}")
                else:
                    clusters_data[cluster_key]["titles"] = {
                        "title_1word": "Empty",
                        "title_3words": "Empty Cluster Error",
                        "title_5words": "Empty Cluster Error Occurred"
                    }

        # Add metrics to clusters data
        result = {
            "k": k,
            "timestamp": timestamp,
            "metrics": metrics,
            "warnings": warnings,
            "clusters": clusters_data
        }

        # Save clusters JSON
        clusters_file = k_dir / "clusters.json"
        with open(clusters_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Saved clusters data to: {clusters_file}")

        # Create visualizations if requested
        if create_visualizations:
            print(f"Creating visualizations...")
            viz_dir = k_dir / "visualizations"
            create_all_visualizations(
                embeddings,
                labels,
                k,
                viz_dir,
                create_2d=True,
                create_3d=True,
                create_distribution=True
            )
            print(f"Saved visualizations to: {viz_dir}")

        print(f"Completed k={k}")

    # Generate summary report
    print(f"\n{'='*70}")
    print(f"Generating Summary Report")
    print(f"{'='*70}")

    comparison = compare_k_metrics(all_metrics)

    summary_report = {
        "experiment_timestamp": timestamp,
        "k_range": list(range(k_range[0], k_range[1])),
        "total_verses": len(verse_refs),
        "comparison": comparison,
        "recommendations": {
            "best_k_by_silhouette": comparison['best_k_silhouette'],
            "best_silhouette_score": comparison['best_silhouette_score'],
            "best_k_by_warnings": comparison['best_k_warnings'],
            "notes": []
        }
    }

    # Add recommendations based on patterns
    if comparison['best_k_silhouette'] != comparison['best_k_warnings']:
        summary_report['recommendations']['notes'].append(
            f"Different k values optimal for silhouette ({comparison['best_k_silhouette']}) "
            f"vs warnings ({comparison['best_k_warnings']}). Consider manual inspection."
        )

    # Check if high k values have issues
    high_k_warnings = sum(1 for i, k in enumerate(comparison['k_values'])
                          if k > 25 and comparison['num_warnings'][i] > 3)
    if high_k_warnings > 3:
        summary_report['recommendations']['notes'].append(
            f"High k values (>25) show quality issues. Consider limiting range."
        )

    # Save summary report
    summary_file = experiment_dir / "summary_report.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_report, f, indent=2)

    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"Output directory: {experiment_dir}")
    print(f"Summary report: {summary_file}")
    print(f"\nRecommendations:")
    print(f"  Best k by silhouette score: {comparison['best_k_silhouette']} "
          f"(score: {comparison['best_silhouette_score']:.4f})")
    print(f"  Best k by warnings: {comparison['best_k_warnings']}")

    if summary_report['recommendations']['notes']:
        print(f"\nNotes:")
        for note in summary_report['recommendations']['notes']:
            print(f"  - {note}")

    print(f"\nTo analyze results, use:")
    print(f"  from cluster_analysis import compare_k_values, inspect_cluster")
    print(f"  compare_k_values('{experiment_dir}')")
    print(f"{'='*70}\n")

    return experiment_dir


def run_single_clustering(
    embeddings: np.ndarray,
    verse_refs: List[Tuple[str, str]],
    k: int,
    output_file: str = None,
    generate_titles: bool = True,
    title_backend: str = 'ollama',
    title_model: str = 'llama3'
) -> Dict:
    # Run clustering for a single k value (convenience function).
    print(f"Running K-means with k={k}...")

    # Run K-means
    model = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = model.fit_predict(embeddings)

    # Load verse data
    verses_dict = getDataFromJson()

    # Compute metrics
    metrics = compute_cluster_metrics(embeddings, labels, k)
    warnings = flag_quality_issues(metrics, k)

    # Organize clusters
    clusters_data = {}
    for cluster_id in range(k):
        cluster_indices = np.where(labels == cluster_id)[0]
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

        clusters_data[f"cluster_{cluster_id}"] = {
            "cluster_id": cluster_id,
            "num_verses": len(cluster_indices),
            "verses": cluster_verses
        }

    # Generate titles if requested
    if generate_titles:
        print(f"Generating titles ({title_backend}: {title_model})...")
        from cluster_titles import generate_titles_for_cluster
        initialize_title_generator(backend=title_backend, model=title_model)

        for cluster_id in range(k):
            cluster_key = f"cluster_{cluster_id}"
            verses_texts = [v["text"]
                            for v in clusters_data[cluster_key]["verses"]]

            if len(verses_texts) > 0:
                titles = generate_titles_for_cluster(
                    verses_texts, backend=title_backend, model=title_model)
                clusters_data[cluster_key]["titles"] = titles

    result = {
        "k": k,
        "metrics": metrics,
        "warnings": warnings,
        "clusters": clusters_data
    }

    # Save if requested
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Saved to: {output_file}")

    # Print summary
    summary = get_quality_summary(metrics, warnings)
    print(summary)

    return result
