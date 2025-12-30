"""
Cluster Analysis Module

Tools to review, compare, and analyze clustering experiment results.
Provides interactive inspection of clusters and quality metrics.
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np


def load_experiment(experiment_dir: Union[str, Path]) -> Dict:
    # Load all results from an experiment directory.
    experiment_dir = Path(experiment_dir)
    
    if not experiment_dir.exists():
        raise ValueError(f"Experiment directory not found: {experiment_dir}")
    
    # Load summary report
    summary_file = experiment_dir / "summary_report.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
    else:
        summary = None
    
    # Load all k results
    k_results = {}
    for k_dir in sorted(experiment_dir.glob("k_*")):
        if k_dir.is_dir():
            clusters_file = k_dir / "clusters.json"
            if clusters_file.exists():
                with open(clusters_file, 'r') as f:
                    k_results[k_dir.name] = json.load(f)
    
    return {
        "experiment_dir": str(experiment_dir),
        "summary": summary,
        "k_results": k_results
    }


def compare_k_values(experiment_dir: Union[str, Path], show_plot: bool = True) -> None:
    # Compare quality metrics across all k values in an experiment.
    experiment = load_experiment(experiment_dir)
    
    if not experiment['k_results']:
        print("No clustering results found in experiment.")
        return
    
    # Extract metrics
    k_values = []
    silhouette_scores = []
    davies_bouldin_scores = []
    size_ratios = []
    num_warnings = []
    num_small_clusters = []
    
    for k_name in sorted(experiment['k_results'].keys(), key=lambda x: int(x.split('_')[1])):
        result = experiment['k_results'][k_name]
        metrics = result['metrics']
        
        k_values.append(metrics['k'])
        silhouette_scores.append(metrics['silhouette_global'])
        davies_bouldin_scores.append(metrics['davies_bouldin'])
        size_ratios.append(metrics['size_ratio'])
        num_warnings.append(len(result['warnings']))
        num_small_clusters.append(metrics['num_small_clusters'])
    
    # Print table
    print(f"\n{'='*100}")
    print(f"K-Values Comparison: {experiment['experiment_dir']}")
    print(f"{'='*100}")
    print(f"{'k':>4} | {'Silhouette':>12} | {'Davies-Bouldin':>15} | {'Size Ratio':>11} | "
          f"{'Warnings':>9} | {'Small Clusters':>14}")
    print(f"{'-'*4}-+-{'-'*12}-+-{'-'*15}-+-{'-'*11}-+-{'-'*9}-+-{'-'*14}")
    
    for i, k in enumerate(k_values):
        warning_marker = "⚠️ " if num_warnings[i] > 2 else "  "
        print(f"{warning_marker}{k:>3} | {silhouette_scores[i]:>12.4f} | "
              f"{davies_bouldin_scores[i]:>15.4f} | {size_ratios[i]:>11.2f} | "
              f"{num_warnings[i]:>9} | {num_small_clusters[i]:>14}")
    
    print(f"{'='*100}")
    
    # Print recommendations
    if experiment['summary']:
        recs = experiment['summary'].get('recommendations', {})
        print(f"\nRecommendations:")
        print(f"  Best k by silhouette: {recs.get('best_k_by_silhouette', 'N/A')}")
        print(f"  Best k by warnings: {recs.get('best_k_by_warnings', 'N/A')}")
        
        if recs.get('notes'):
            print(f"\nNotes:")
            for note in recs['notes']:
                print(f"  - {note}")
    
    # Create plots
    if show_plot:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Silhouette scores
        axes[0, 0].plot(k_values, silhouette_scores, marker='o', linewidth=2, color='steelblue')
        axes[0, 0].set_xlabel('Number of Clusters (k)')
        axes[0, 0].set_ylabel('Silhouette Score')
        axes[0, 0].set_title('Silhouette Score vs k (higher is better)')
        axes[0, 0].grid(alpha=0.3)
        axes[0, 0].axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Threshold (0.3)')
        axes[0, 0].legend()
        
        # Plot 2: Davies-Bouldin scores
        axes[0, 1].plot(k_values, davies_bouldin_scores, marker='o', linewidth=2, color='coral')
        axes[0, 1].set_xlabel('Number of Clusters (k)')
        axes[0, 1].set_ylabel('Davies-Bouldin Score')
        axes[0, 1].set_title('Davies-Bouldin Score vs k (lower is better)')
        axes[0, 1].grid(alpha=0.3)
        
        # Plot 3: Size ratios
        axes[1, 0].plot(k_values, size_ratios, marker='o', linewidth=2, color='green')
        axes[1, 0].set_xlabel('Number of Clusters (k)')
        axes[1, 0].set_ylabel('Size Ratio (max/min)')
        axes[1, 0].set_title('Cluster Size Imbalance vs k')
        axes[1, 0].grid(alpha=0.3)
        axes[1, 0].axhline(y=10, color='red', linestyle='--', alpha=0.5, label='Threshold (10)')
        axes[1, 0].legend()
        
        # Plot 4: Number of warnings
        axes[1, 1].bar(k_values, num_warnings, color='orange', alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Number of Clusters (k)')
        axes[1, 1].set_ylabel('Number of Warnings')
        axes[1, 1].set_title('Quality Warnings vs k')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def inspect_cluster(
    experiment_dir: Union[str, Path],
    k: int,
    cluster_id: int,
    show_verses: bool = True,
    max_verses: int = 20
) -> None:
    # Display detailed information about a specific cluster.
    experiment = load_experiment(experiment_dir)
    
    k_key = f"k_{k}"
    if k_key not in experiment['k_results']:
        print(f"No results found for k={k}")
        return
    
    result = experiment['k_results'][k_key]
    cluster_key = f"cluster_{cluster_id}"
    
    if cluster_key not in result['clusters']:
        print(f"Cluster {cluster_id} not found in k={k} results")
        return
    
    cluster = result['clusters'][cluster_key]
    metrics = result['metrics']
    
    print(f"\n{'='*80}")
    print(f"Cluster {cluster_id} (k={k})")
    print(f"{'='*80}")
    
    # Cluster info
    print(f"\nCluster Size: {cluster['num_verses']} verses")
    
    # Titles if available
    if 'titles' in cluster:
        titles = cluster['titles']
        print(f"\nGenerated Titles:")
        print(f"  1-word:  {titles.get('title_1word', 'N/A')}")
        print(f"  3-word:  {titles.get('title_3words', 'N/A')}")
        print(f"  5-word:  {titles.get('title_5words', 'N/A')}")
    
    # Quality metrics
    if 'silhouette_per_cluster' in metrics:
        silhouette = metrics['silhouette_per_cluster'].get(str(cluster_id), 'N/A')
        print(f"\nQuality Metrics:")
        print(f"  Silhouette score: {silhouette}")
        
        if isinstance(silhouette, float):
            if silhouette < 0.2:
                print(f"  ⚠️  Low silhouette - may contain mixed topics")
            elif silhouette > 0.5:
                print(f"  ✓ Good silhouette - well-separated cluster")
    
    # Verses
    if show_verses:
        print(f"\nVerses (showing {min(max_verses, len(cluster['verses']))} of {len(cluster['verses'])}):")
        print(f"{'-'*80}")
        
        for i, verse_data in enumerate(cluster['verses'][:max_verses]):
            chapter = verse_data['chapter']
            verse = verse_data['verse']
            text = verse_data['text']
            print(f"{i+1}. Proverbs {chapter}:{verse}")
            print(f"   {text}")
            print()
        
        if len(cluster['verses']) > max_verses:
            print(f"... and {len(cluster['verses']) - max_verses} more verses")
    
    print(f"{'='*80}\n")


def find_best_k(
    experiment_dir: Union[str, Path],
    criteria: str = 'silhouette'
) -> int:
    # Find the optimal k value based on specified criteria.
    experiment = load_experiment(experiment_dir)
    
    if not experiment['k_results']:
        raise ValueError("No clustering results found in experiment.")
    
    if criteria == 'silhouette':
        # Find k with highest silhouette score
        best_k = None
        best_score = -1
        
        for k_name, result in experiment['k_results'].items():
            score = result['metrics']['silhouette_global']
            if score > best_score:
                best_score = score
                best_k = result['metrics']['k']
        
        print(f"Best k by silhouette score: {best_k} (score: {best_score:.4f})")
        return best_k
    
    elif criteria == 'warnings':
        # Find k with fewest warnings
        best_k = None
        min_warnings = float('inf')
        
        for k_name, result in experiment['k_results'].items():
            num_warnings = len(result['warnings'])
            if num_warnings < min_warnings:
                min_warnings = num_warnings
                best_k = result['metrics']['k']
        
        print(f"Best k by warnings: {best_k} ({min_warnings} warnings)")
        return best_k
    
    elif criteria == 'balanced':
        # Balance between silhouette score and number of warnings
        best_k = None
        best_combined_score = -float('inf')
        
        # Normalize scores
        silhouettes = [r['metrics']['silhouette_global'] for r in experiment['k_results'].values()]
        warnings_counts = [len(r['warnings']) for r in experiment['k_results'].values()]
        
        max_sil = max(silhouettes)
        max_warn = max(warnings_counts) if max(warnings_counts) > 0 else 1
        
        for k_name, result in experiment['k_results'].items():
            sil_norm = result['metrics']['silhouette_global'] / max_sil
            warn_norm = 1 - (len(result['warnings']) / max_warn)
            
            # Combined score: 60% silhouette, 40% warnings
            combined = 0.6 * sil_norm + 0.4 * warn_norm
            
            if combined > best_combined_score:
                best_combined_score = combined
                best_k = result['metrics']['k']
        
        print(f"Best k by balanced criteria: {best_k} (combined score: {best_combined_score:.4f})")
        return best_k
    
    else:
        raise ValueError(f"Unknown criteria: {criteria}. Use 'silhouette', 'warnings', or 'balanced'")


def export_cluster_verses(
    experiment_dir: Union[str, Path],
    k: int,
    cluster_id: int,
    output_file: Union[str, Path]
) -> None:
    # Export all verses from a cluster to a text file.
    experiment = load_experiment(experiment_dir)
    
    k_key = f"k_{k}"
    if k_key not in experiment['k_results']:
        print(f"No results found for k={k}")
        return
    
    result = experiment['k_results'][k_key]
    cluster_key = f"cluster_{cluster_id}"
    
    if cluster_key not in result['clusters']:
        print(f"Cluster {cluster_id} not found in k={k} results")
        return
    
    cluster = result['clusters'][cluster_key]
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Cluster {cluster_id} from k={k}\n")
        f.write(f"{'='*80}\n\n")
        
        if 'titles' in cluster:
            titles = cluster['titles']
            f.write(f"Titles:\n")
            f.write(f"  1-word:  {titles.get('title_1word', 'N/A')}\n")
            f.write(f"  3-word:  {titles.get('title_3words', 'N/A')}\n")
            f.write(f"  5-word:  {titles.get('title_5words', 'N/A')}\n\n")
        
        f.write(f"Total verses: {cluster['num_verses']}\n")
        f.write(f"{'='*80}\n\n")
        
        for i, verse_data in enumerate(cluster['verses']):
            chapter = verse_data['chapter']
            verse = verse_data['verse']
            text = verse_data['text']
            f.write(f"{i+1}. Proverbs {chapter}:{verse}\n")
            f.write(f"   {text}\n\n")
    
    print(f"Exported cluster {cluster_id} to: {output_file}")


if __name__ == "__main__":
    # Simple CLI interface for cluster analysis.
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python cluster_analysis.py <experiment_dir>")
        print("\nExample: python cluster_analysis.py experiments/experiment_20250101_120000")
        sys.exit(1)
    
    experiment_dir = sys.argv[1]
    
    print("\n" + "="*80)
    print("Cluster Analysis Tool")
    print("="*80)
    
    while True:
        print("\nOptions:")
        print("  1. Compare all k values")
        print("  2. Inspect specific cluster")
        print("  3. Find best k")
        print("  4. Export cluster verses")
        print("  5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            compare_k_values(experiment_dir)
        
        elif choice == '2':
            k = int(input("Enter k value: ").strip())
            cluster_id = int(input("Enter cluster ID: ").strip())
            inspect_cluster(experiment_dir, k, cluster_id)
        
        elif choice == '3':
            criteria = input("Enter criteria (silhouette/warnings/balanced): ").strip()
            find_best_k(experiment_dir, criteria)
        
        elif choice == '4':
            k = int(input("Enter k value: ").strip())
            cluster_id = int(input("Enter cluster ID: ").strip())
            output_file = input("Enter output file path: ").strip()
            export_cluster_verses(experiment_dir, k, cluster_id, output_file)
        
        elif choice == '5':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please try again.")


