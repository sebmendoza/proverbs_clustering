"""
Title Regeneration Script

Regenerate cluster titles for existing experiments using improved backends (Ollama).
Updates JSON files in-place with automatic backups.
"""

import json
import argparse
import shutil
from pathlib import Path
from typing import Dict, List
from datetime import datetime

from kmeans.cluster_titles import initialize_title_generator


def backup_json_file(json_path: Path) -> Path:
    # Create a backup of JSON file before modification.
    backup_path = json_path.with_suffix('.json.backup')
    shutil.copy2(json_path, backup_path)
    return backup_path


def regenerate_titles_for_k(
    clusters_json_path: Path,
    backend: str = 'ollama',
    model: str = 'llama3',
    dry_run: bool = False
) -> Dict:
    # Regenerate titles for a single k value (one clusters.json file).
    print(f"\n{'='*80}")
    print(f"Processing: {clusters_json_path}")
    print(f"{'='*80}")

    # Load existing data
    with open(clusters_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    k = data['k']
    clusters = data['clusters']

    print(f"K value: {k}")
    print(f"Number of clusters: {len(clusters)}")
    print(f"Backend: {backend} ({model})")

    # Initialize title generator
    print("\nInitializing title generator...")
    generator = initialize_title_generator(backend=backend, model=model)

    # Track changes
    comparisons = []
    updated_count = 0

    # Regenerate titles for each cluster
    print(f"\nRegenerating titles...")
    for cluster_key, cluster_data in clusters.items():
        cluster_id = cluster_data['cluster_id']
        num_verses = cluster_data['num_verses']
        verses_texts = [v['text'] for v in cluster_data['verses']]

        print(f"\n  Cluster {cluster_id} ({num_verses} verses)...")

        # Get old titles
        old_titles = cluster_data.get('titles', {})

        # Generate new titles
        try:
            new_titles = generator.generate_all_titles(verses_texts)

            # Update in data structure
            cluster_data['titles'] = new_titles
            updated_count += 1

            # Store comparison
            comparison = {
                'cluster_id': cluster_id,
                'old': old_titles,
                'new': new_titles
            }
            comparisons.append(comparison)

            # Print comparison
            print(f"    Old 1-word:  {old_titles.get('title_1word', 'N/A')}")
            print(f"    New 1-word:  {new_titles.get('title_1word', 'N/A')}")
            print(f"    Old 3-words: {old_titles.get('title_3words', 'N/A')}")
            print(f"    New 3-words: {new_titles.get('title_3words', 'N/A')}")

        except Exception as e:
            print(f"    ⚠️  Error generating titles: {e}")
            comparisons.append({
                'cluster_id': cluster_id,
                'error': str(e)
            })

    # Save updated data
    if not dry_run:
        print(f"\n{'='*80}")
        print(f"Saving updated titles...")

        # Create backup
        backup_path = backup_json_file(clusters_json_path)
        print(f"Backup created: {backup_path}")

        # Save updated JSON
        with open(clusters_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✓ Updated: {clusters_json_path}")
        print(f"  {updated_count}/{len(clusters)} clusters updated")
    else:
        print(f"\n{'='*80}")
        print(f"DRY RUN - No files modified")

    return {
        'k': k,
        'total_clusters': len(clusters),
        'updated_count': updated_count,
        'comparisons': comparisons
    }


def regenerate_titles_for_experiment(
    experiment_dir: Path,
    backend: str = 'ollama',
    model: str = 'llama3',
    k_values: List[int] = None,
    dry_run: bool = False
) -> Dict:
    # Regenerate titles for all k values in an experiment.
    experiment_dir = Path(experiment_dir)

    if not experiment_dir.exists():
        raise ValueError(f"Experiment directory not found: {experiment_dir}")

    print(f"\n{'#'*80}")
    print(f"# REGENERATING TITLES FOR EXPERIMENT")
    print(f"# {experiment_dir}")
    print(f"{'#'*80}")

    # Find all k directories
    k_dirs = sorted([d for d in experiment_dir.glob("k_*") if d.is_dir()],
                    key=lambda x: int(x.name.split('_')[1]))

    if not k_dirs:
        print("No k_* directories found in experiment.")
        return {}

    # Filter by specific k values if requested
    if k_values:
        k_dirs = [d for d in k_dirs if int(d.name.split('_')[1]) in k_values]

    print(f"\nFound {len(k_dirs)} k directories to process")
    if dry_run:
        print("⚠️  DRY RUN MODE - No files will be modified")

    # Process each k
    results = []
    start_time = datetime.now()

    for i, k_dir in enumerate(k_dirs, 1):
        clusters_json = k_dir / 'clusters.json'

        if not clusters_json.exists():
            print(f"\n⚠️  Skipping {k_dir.name}: clusters.json not found")
            continue

        print(f"\n[{i}/{len(k_dirs)}] Processing {k_dir.name}...")

        try:
            result = regenerate_titles_for_k(
                clusters_json,
                backend=backend,
                model=model,
                dry_run=dry_run
            )
            results.append(result)
        except Exception as e:
            print(f"⚠️  Error processing {k_dir.name}: {e}")

    # Print summary
    elapsed = datetime.now() - start_time

    print(f"\n{'#'*80}")
    print(f"# SUMMARY")
    print(f"{'#'*80}")
    print(f"Experiment: {experiment_dir.name}")
    print(f"Backend: {backend} ({model})")
    print(f"Processed: {len(results)} k values")
    print(f"Elapsed time: {elapsed}")

    if not dry_run:
        total_updated = sum(r['updated_count'] for r in results)
        total_clusters = sum(r['total_clusters'] for r in results)
        print(f"Total clusters updated: {total_updated}/{total_clusters}")

    return {
        'experiment_dir': str(experiment_dir),
        'backend': backend,
        'model': model,
        'results': results,
        'elapsed_time': str(elapsed)
    }


def regenerate_all_experiments(
    experiments_dir: Path = Path("experiments"),
    backend: str = 'ollama',
    model: str = 'llama3',
    dry_run: bool = False
) -> List[Dict]:
    # Regenerate titles for all experiments in the experiments directory.
    experiments_dir = Path(experiments_dir)

    if not experiments_dir.exists():
        raise ValueError(f"Experiments directory not found: {experiments_dir}")

    # Find all experiment directories
    exp_dirs = sorted([d for d in experiments_dir.glob("experiment_*") if d.is_dir()],
                      key=lambda x: x.name)

    if not exp_dirs:
        print("No experiment directories found.")
        return []

    print(f"\n{'#'*80}")
    print(f"# REGENERATING TITLES FOR ALL EXPERIMENTS")
    print(f"{'#'*80}")
    print(f"Found {len(exp_dirs)} experiments")

    all_results = []

    for i, exp_dir in enumerate(exp_dirs, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(exp_dirs)}] {exp_dir.name}")
        print(f"{'='*80}")

        try:
            result = regenerate_titles_for_experiment(
                exp_dir,
                backend=backend,
                model=model,
                dry_run=dry_run
            )
            all_results.append(result)
        except Exception as e:
            print(f"⚠️  Error processing {exp_dir.name}: {e}")

    print(f"\n{'#'*80}")
    print(f"# COMPLETE")
    print(f"{'#'*80}")
    print(f"Processed {len(all_results)} experiments successfully")

    return all_results


def preview_single_cluster(
    clusters_json_path: Path,
    cluster_id: int,
    backend: str = 'ollama',
    model: str = 'llama3'
) -> None:
    # Preview title regeneration for a single cluster.
    # Load data
    with open(clusters_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cluster_key = f"cluster_{cluster_id}"
    if cluster_key not in data['clusters']:
        print(f"Cluster {cluster_id} not found")
        return

    cluster_data = data['clusters'][cluster_key]
    verses_texts = [v['text'] for v in cluster_data['verses']]
    old_titles = cluster_data.get('titles', {})

    print(f"\n{'='*80}")
    print(f"Cluster {cluster_id} Preview")
    print(f"{'='*80}")
    print(f"Number of verses: {len(verses_texts)}")
    print(f"\nFirst 3 verses:")
    for i, verse in enumerate(verses_texts[:3]):
        print(f"  {i+1}. {verse}")

    print(f"\nOld titles:")
    for key, value in old_titles.items():
        print(f"  {key}: {value}")

    print(f"\nGenerating new titles with {backend} ({model})...")
    generator = initialize_title_generator(backend=backend, model=model)
    new_titles = generator.generate_all_titles(verses_texts)

    print(f"\nNew titles:")
    for key, value in new_titles.items():
        print(f"  {key}: {value}")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Regenerate cluster titles for existing experiments"
    )

    parser.add_argument(
        '--experiment',
        type=str,
        help='Path to specific experiment directory'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all experiments in experiments/ directory'
    )

    parser.add_argument(
        '--backend',
        type=str,
        default='ollama',
        choices=['ollama', 'transformers'],
        help='Backend to use (default: ollama)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='llama3',
        help='Model name (default: llama3 for ollama, flan-t5-small for transformers)'
    )

    parser.add_argument(
        '--k',
        type=int,
        nargs='+',
        help='Specific k values to process (e.g., --k 19 23)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without saving'
    )

    parser.add_argument(
        '--preview',
        type=int,
        metavar='CLUSTER_ID',
        help='Preview single cluster from first k in experiment'
    )

    args = parser.parse_args()

    # Set default model for transformers
    if args.backend == 'transformers' and args.model == 'llama3':
        args.model = 'google/flan-t5-small'

    try:
        if args.preview is not None and args.experiment:
            # Preview mode
            exp_path = Path(args.experiment)
            k_dirs = sorted([d for d in exp_path.glob("k_*") if d.is_dir()])
            if k_dirs:
                clusters_json = k_dirs[0] / 'clusters.json'
                preview_single_cluster(
                    clusters_json,
                    args.preview,
                    args.backend,
                    args.model
                )
            else:
                print("No k directories found in experiment")

        elif args.all:
            # Process all experiments
            regenerate_all_experiments(
                backend=args.backend,
                model=args.model,
                dry_run=args.dry_run
            )

        elif args.experiment:
            # Process single experiment
            regenerate_titles_for_experiment(
                Path(args.experiment),
                backend=args.backend,
                model=args.model,
                k_values=args.k,
                dry_run=args.dry_run
            )

        else:
            parser.print_help()
            print("\nExamples:")
            print("  # Preview single cluster")
            print(
                "  python regenerate_titles.py --experiment experiments/experiment_20251223_114054 --preview 0")
            print("\n  # Regenerate for single experiment")
            print(
                "  python regenerate_titles.py --experiment experiments/experiment_20251223_114054")
            print("\n  # Regenerate specific k values only")
            print(
                "  python regenerate_titles.py --experiment experiments/experiment_20251223_114054 --k 19 23")
            print("\n  # Dry run to preview changes")
            print(
                "  python regenerate_titles.py --experiment experiments/experiment_20251223_114054 --dry-run")
            print("\n  # Process all experiments")
            print("  python regenerate_titles.py --all")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n⚠️  Error: {e}")
        raise
