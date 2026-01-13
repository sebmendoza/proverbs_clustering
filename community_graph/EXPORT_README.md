# Community Graph Export for Web Projects

This standalone script runs community detection and exports results to JSON format for use in TypeScript/React projects.

## Features

- **Standalone execution**: Run independently without modifying other code
- **Self-documenting JSON**: Includes metadata about algorithm, parameters, and creation date
- **Complete node data**: Verse text, positions (2D & 3D), community assignments, and statistics
- **Community summaries**: Size, color, representative verses, and connectivity metrics
- **Edge data**: Similarity scores with intra/inter-community classification
- **TypeScript types**: Auto-generated type definitions for type-safe development

## Quick Start

### Run the Export

```bash
cd /Users/sebastianmendoza/Waterloo/Projects/Proverbs/community_graph
python export_to_project.py
```

This creates:
- `proverbs_communities_threshold_0.5_export.json` - Complete graph data
- `proverbs_graph_types.d.ts` - TypeScript type definitions

### Custom Threshold

```bash
python export_to_project.py --threshold 0.6
```

### Custom Output Directory

```bash
python export_to_project.py --output-dir ../exports
```

## Output Files

### 1. JSON Data Export
- **Filename**: `proverbs_communities_threshold_{threshold}_export.json`
- **Size**: ~500-800 KB (depending on threshold)
- **Content**: Complete graph data with metadata

### 2. TypeScript Types
- **Filename**: `proverbs_graph_types.d.ts`
- **Content**: Type definitions for the exported data structure

## Data Structure

```typescript
{
  "metadata": {
    "algorithm": "Leiden",
    "threshold": 0.5,
    "modularity": 0.723,
    "created_at": "2026-01-09T...",
    // ...
  },
  
  "graph_stats": {
    "total_nodes": 915,
    "total_edges": 12847,
    "num_communities": 15,
    // ...
  },
  
  "nodes": [
    {
      "id": "Proverbs 1:1",
      "text": "The proverbs of Solomon...",
      "community_id": 3,
      "position_2d": {"x": 0.123, "y": -0.456},
      "position_3d": {"x": 0.123, "y": -0.456, "z": 0.789},
      "degree": 24,
      "chapter": 1,
      "verse": 1
    }
    // ... more nodes
  ],
  
  "communities": {
    "0": {
      "id": 0,
      "size": 87,
      "color": "#e6194b",
      "members": ["Proverbs 1:1", ...],
      "representative_verses": ["Proverbs 15:3", ...],
      "stats": {
        "density": 0.234,
        "avg_degree": 12.5,
        // ...
      }
    }
    // ... more communities
  },
  
  "edges": [
    {
      "source": "Proverbs 1:1",
      "target": "Proverbs 1:2",
      "weight": 0.623,
      "same_community": true,
      "community_id": 3,
      "type": "intra"
    }
    // ... more edges
  ]
}
```

## Using in TypeScript/React

### 1. Copy Files to Your Project

```bash
# Copy the JSON data
cp proverbs_communities_threshold_0.5_export.json your-project/src/data/

# Copy the type definitions
cp proverbs_graph_types.d.ts your-project/src/types/
```

### 2. Import and Use

```typescript
import graphData from './data/proverbs_communities_threshold_0.5_export.json';
import type { ProverbsGraphData, Node, Community } from './types/proverbs_graph_types';

// Type-safe access to the data
const data: ProverbsGraphData = graphData;

// Get all nodes
const nodes: Node[] = data.nodes;

// Get a specific community
const community: Community = data.communities["0"];

// Filter nodes by community
const communityNodes = nodes.filter(n => n.community_id === 0);

// Find high-similarity edges
const strongEdges = data.edges.filter(e => e.weight > 0.6);

// Get metadata
console.log(`Created: ${data.metadata.created_at}`);
console.log(`Modularity: ${data.metadata.modularity}`);
```

### 3. Visualization Example

```typescript
// Using D3.js or similar
import * as d3 from 'd3';

function renderGraph(data: ProverbsGraphData) {
  const svg = d3.select('#graph-container')
    .append('svg');
  
  // Create nodes
  const nodes = svg.selectAll('circle')
    .data(data.nodes)
    .enter()
    .append('circle')
    .attr('cx', d => scale(d.position_2d.x))
    .attr('cy', d => scale(d.position_2d.y))
    .attr('fill', d => data.communities[d.community_id].color)
    .attr('r', 5);
  
  // Create edges
  const edges = svg.selectAll('line')
    .data(data.edges.filter(e => e.same_community))
    .enter()
    .append('line')
    .attr('stroke', '#999')
    .attr('stroke-opacity', d => d.weight);
}
```

## Metadata Reference

### Algorithm Information
- **algorithm**: Name of the community detection algorithm (e.g., "Leiden")
- **threshold**: Similarity threshold used for edge creation
- **modularity**: Quality score (higher = better community structure)
- **seed**: Random seed for reproducibility

### Graph Statistics
- **total_nodes**: Number of verses in the graph
- **total_edges**: Number of similarity connections
- **num_communities**: Number of detected communities
- **density**: Graph density (0-1, measures how connected the graph is)
- **avg_clustering**: Average clustering coefficient

### Community Statistics
- **density**: How interconnected the community is internally
- **avg_degree**: Average number of connections per verse
- **internal_edges**: Edges within the community
- **external_edges**: Edges connecting to other communities

## Filtering and Querying

### Filter by Chapter
```typescript
const chapter10 = data.nodes.filter(n => n.chapter === 10);
```

### Find Most Connected Verses
```typescript
const hubs = data.nodes
  .sort((a, b) => b.degree - a.degree)
  .slice(0, 10);
```

### Get Community Statistics
```typescript
const communitySizes = Object.values(data.communities)
  .map(c => ({ id: c.id, size: c.size }))
  .sort((a, b) => b.size - a.size);
```

### Find Inter-Community Bridges
```typescript
const bridges = data.edges.filter(e => !e.same_community);
```

## Advanced Usage

### Calculating Additional Metrics

The exported data includes everything you need to calculate additional metrics in your web app:

```typescript
// Calculate verse importance (PageRank-like)
function calculateImportance(nodeId: string, data: ProverbsGraphData): number {
  const node = data.nodes.find(n => n.id === nodeId);
  const incomingEdges = data.edges.filter(e => e.target === nodeId);
  const weightedDegree = incomingEdges.reduce((sum, e) => sum + e.weight, 0);
  return weightedDegree / data.graph_stats.total_edges;
}

// Find similar verses
function findSimilar(nodeId: string, data: ProverbsGraphData, topN: number = 5) {
  return data.edges
    .filter(e => e.source === nodeId || e.target === nodeId)
    .sort((a, b) => b.weight - a.weight)
    .slice(0, topN);
}
```

## Command Line Options

```bash
python export_to_project.py --help
```

### Available Options

- `--threshold THRESHOLD` - Similarity threshold for edge creation (default: 0.5)
- `--output-dir DIR` - Output directory for export files (default: current directory)

### Examples

```bash
# Export with threshold 0.6
python export_to_project.py --threshold 0.6

# Export to specific directory
python export_to_project.py --output-dir ~/Desktop/exports

# Both options together
python export_to_project.py --threshold 0.55 --output-dir ../data
```

## What Happens During Export

1. **Load embeddings** - Reads or creates verse embeddings
2. **Create similarity graph** - Builds graph with edges above threshold
3. **Extract main component** - Focuses on largest connected component
4. **Run Leiden algorithm** - Detects communities
5. **Generate positions** - Creates 2D (spring layout) and 3D (PCA) coordinates
6. **Calculate statistics** - Computes all graph and community metrics
7. **Export JSON** - Writes complete data file
8. **Generate types** - Creates TypeScript definitions

## Troubleshooting

### JSON File Too Large
If the export file is too large for your needs:
- Increase the threshold to reduce edges: `--threshold 0.6`
- Filter out low-weight edges in post-processing
- Consider exporting multiple threshold levels

### Missing Embeddings
If embeddings haven't been created yet, the script will:
1. Download the sentence-transformers model (if needed)
2. Generate embeddings for all verses
3. Cache them in `esv_embeddings.pkl`

This may take a few minutes on first run.

### TypeScript Type Errors
Make sure to import types correctly:
```typescript
import type { ProverbsGraphData } from './types/proverbs_graph_types';
```

## Next Steps

1. **Visualization**: Use D3.js, Three.js, or React Flow for interactive graphs
2. **Search**: Implement full-text search across verse texts
3. **Filtering**: Add UI controls to filter by community, chapter, or similarity
4. **Analysis**: Display statistics and insights about communities
5. **Comparison**: Export multiple threshold levels and compare results

## Questions?

The export includes everything needed for rich web visualization. The self-documenting structure ensures you always know the context and parameters of the analysis.
