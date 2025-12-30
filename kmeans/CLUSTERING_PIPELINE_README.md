# Proverbs Clustering Pipeline - User Guide

## Overview

This pipeline provides a comprehensive system for semantic clustering of Proverbs with quality metrics, automated title generation, and visualization.

## Quick Start

### Installation

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

2. **Install Ollama** (Recommended for high-quality titles):

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: Download from https://ollama.com/download
```

3. **Start Ollama and pull a model**:

```bash
# Start Ollama service (keep running)
ollama serve

# In another terminal, pull the model
ollama pull llama3
```

See [OLLAMA_SETUP.md](OLLAMA_SETUP.md) for detailed setup instructions.

### Running the Pipeline

The simplest way to run the full pipeline:

```python
python main.py
```

This will:

- Load existing embeddings (or create them if needed)
- Run K-means clustering for k=10 to k=40
- Compute quality metrics for each k value
- Generate 2D and 3D visualizations
- Generate 1, 3, and 5-word cluster titles using LLM
- Save all results to `experiments/experiment_TIMESTAMP/`

## Pipeline Architecture

### Core Modules

1. **`clustering_pipeline.py`** - Main orchestrator
2. **`cluster_quality.py`** - Quality metrics computation
3. **`cluster_viz.py`** - Visualization generation
4. **`cluster_titles.py`** - LLM-based title generation
5. **`cluster_analysis.py`** - Results analysis and comparison

### Data Flow

```
Embeddings → K-means (k=10-40) → Quality Metrics → Visualizations → Titles → JSON
```

## Using the Pipeline

### Method 1: Run Full Experiment

```python
from clustering_pipeline import run_clustering_pipeline
from embeddings import get_or_create_embeddings
import numpy as np

# Load data
embeddings_dict = get_or_create_embeddings()
embeddings = []
verse_refs = []
for chapter, verses in embeddings_dict.items():
    for verse, embed in verses.items():
        embeddings.append(embed)
        verse_refs.append((chapter, verse))

arr = np.array(embeddings)

# Run full pipeline
experiment_dir = run_clustering_pipeline(
    embeddings=arr,
    verse_refs=verse_refs,
    k_range=(10, 41),  # Test k from 10 to 40
    output_dir="experiments",
    generate_titles=True,
    create_visualizations=True
)
```

### Method 2: Run Single K Value

```python
from clustering_pipeline import run_single_clustering

result = run_single_clustering(
    embeddings=arr,
    verse_refs=verse_refs,
    k=19,
    output_file="my_k19_results.json",
    generate_titles=True
)
```

## Analyzing Results

### Interactive CLI

```bash
python cluster_analysis.py experiments/experiment_20250101_120000
```

This opens an interactive menu where you can:

1. Compare all k values
2. Inspect specific clusters
3. Find optimal k
4. Export cluster verses

### Programmatic Analysis

```python
from cluster_analysis import (
    compare_k_values,
    inspect_cluster,
    find_best_k,
    export_cluster_verses
)

experiment_dir = "experiments/experiment_20250101_120000"

# Compare all k values with visualizations
compare_k_values(experiment_dir, show_plot=True)

# Find optimal k
best_k = find_best_k(experiment_dir, criteria='balanced')

# Inspect a specific cluster
inspect_cluster(experiment_dir, k=19, cluster_id=0)

# Export verses from a cluster
export_cluster_verses(
    experiment_dir,
    k=19,
    cluster_id=0,
    output_file="cluster_0_verses.txt"
)
```

## Output Structure

```
experiments/
└── experiment_20250101_120000/
    ├── metadata.json
    ├── summary_report.json
    ├── k_10/
    │   ├── clusters.json
    │   └── visualizations/
    │       ├── clusters_2d.png
    │       ├── clusters_3d.html
    │       └── cluster_distribution.png
    ├── k_11/
    └── ...
```

### clusters.json Structure

```json
{
  "k": 19,
  "timestamp": "20250101_120000",
  "metrics": {
    "silhouette_global": 0.234,
    "davies_bouldin": 1.456,
    "size_ratio": 5.2,
    "num_small_clusters": 2,
    ...
  },
  "warnings": [
    "MIXED_TOPICS: Cluster 5 has low silhouette (0.15)"
  ],
  "clusters": {
    "cluster_0": {
      "cluster_id": 0,
      "num_verses": 45,
      "titles": {
        "title_1word": "Laziness",
        "title_3words": "Sluggard Leads Ruin",
        "title_5words": "Laziness Brings Poverty And Destruction"
      },
      "verses": [
        {
          "chapter": "20",
          "verse": "4",
          "text": "The sluggard does not plow...",
          "index": 123
        },
        ...
      ]
    }
  }
}
```

## Quality Metrics Explained

### Global Metrics

- **Silhouette Score** (0-1, higher is better): Measures how well-separated clusters are

  - > 0.5: Strong separation
  - 0.3-0.5: Moderate separation
  - <0.3: Poor separation (warning)

- **Davies-Bouldin Score** (lower is better): Measures cluster compactness vs separation

- **Size Ratio** (max/min): Cluster imbalance
  - > 10: High imbalance (warning)

### Per-Cluster Metrics

- **Per-cluster Silhouette**: Identifies mixed-topic clusters

  - <0.2: Likely contains mixed topics (warning)

- **Small Cluster Count**: Number of clusters with <5 verses
  - May indicate noise or over-clustering

## Addressing Common Issues

### Issue 1: High k values create tiny clusters

**Detection**: Pipeline automatically flags with "HIGH_K_NOISE" warning

**Solution**: The quality metrics will show when k is too high. Review summary report to find optimal range.

### Issue 2: Low k values create mixed topics

**Detection**: Per-cluster silhouette scores identify clusters with mixed topics

**Solution**:

- Check "MIXED_TOPICS" warnings in results
- Increase k or manually inspect flagged clusters

### Issue 3: Slow LLM title generation

**Solution**:

- Use smaller model: `initialize_title_generator("google/flan-t5-small")`
- Disable titles temporarily: `generate_titles=False`
- Use GPU if available (automatically detected)

## Customization

### Using Different Title Generation Backends

**Ollama (Recommended)** - High quality, local, free:

```python
experiment_dir = run_clustering_pipeline(
    embeddings=arr,
    verse_refs=verse_refs,
    title_backend='ollama',
    title_model='llama3'  # or 'mistral', 'phi3:mini'
)
```

**Transformers (Fallback)** - If Ollama unavailable:

```python
experiment_dir = run_clustering_pipeline(
    embeddings=arr,
    verse_refs=verse_refs,
    title_backend='transformers',
    title_model='google/flan-t5-small'
)
```

**Note**: Ollama produces **much better** titles than Flan-T5-small. See [OLLAMA_SETUP.md](OLLAMA_SETUP.md) for setup.

### Custom K Range

```python
# Test only k=15 to k=25
experiment_dir = run_clustering_pipeline(
    embeddings=arr,
    verse_refs=verse_refs,
    k_range=(15, 26),  # 26 because range is exclusive
    ...
)
```

### Custom UMAP Parameters

Edit `cluster_viz.py` functions to change UMAP parameters:

- `n_neighbors`: Controls local vs global structure (default: 15)
- `metric`: Distance metric (default: 'cosine')
- `min_dist`: Minimum distance between points (default: 0.1)

## Regenerating Titles for Existing Experiments

If you already ran experiments with poor-quality titles (Flan-T5), you can regenerate them with Ollama:

### Preview Single Cluster

```bash
python regenerate_titles.py \
    --experiment experiments/experiment_20251223_114054 \
    --preview 0
```

### Dry Run (See Changes Without Saving)

```bash
python regenerate_titles.py \
    --experiment experiments/experiment_20251223_114054 \
    --dry-run
```

### Regenerate One Experiment

```bash
python regenerate_titles.py \
    --experiment experiments/experiment_20251223_114054 \
    --model llama3
```

### Regenerate Specific K Values Only

```bash
python regenerate_titles.py \
    --experiment experiments/experiment_20251223_114054 \
    --k 19 23 28 \
    --model llama3
```

### Regenerate All Experiments

```bash
python regenerate_titles.py --all --model llama3
```

**Note**: Original files are automatically backed up as `.json.backup` before modification.

## Best Practices

1. **Start with default range** (k=10-40) to get a broad view
2. **Review summary report** to identify optimal k range
3. **Narrow range if needed** based on quality metrics
4. **Manually inspect** clusters at recommended k values
5. **Compare titles** across different k values for same topics
6. **Export verses** for clusters of interest for deeper analysis

## Troubleshooting

### Out of Memory

- Reduce k range
- Disable visualizations: `create_visualizations=False`
- Use smaller LLM model

### Slow Performance

- Disable title generation for initial exploration
- Use GPU if available
- Reduce k range
- Close other applications

### Poor Cluster Quality Across All K

- Check embedding quality
- Review original verse data
- Try different distance metrics in K-means
- Consider data preprocessing

## Legacy Functions

The original `kmeans.py` functions are still available but marked as legacy:

```python
from kmeans import run_kmeans_experiments

# This runs k=2-29 with basic metrics
run_kmeans_experiments(arr, verse_refs)
```

For new work, use `clustering_pipeline` instead.
