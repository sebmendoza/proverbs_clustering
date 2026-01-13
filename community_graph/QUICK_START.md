# Quick Start: Export to Web Project

## TL;DR

Run standalone export script, get web-ready JSON + TypeScript types.

## One Command

```bash
cd /Users/sebastianmendoza/Waterloo/Projects/Proverbs/community_graph
python export_to_project.py
```

This will:
1. ✅ Load embeddings
2. ✅ Run community detection (Leiden algorithm)
3. ✅ Generate 2D and 3D positions
4. ✅ **Export JSON** → `proverbs_communities_threshold_0.5_export.json`
5. ✅ **Generate TypeScript types** → `proverbs_graph_types.d.ts`

## Options

```bash
# Use different threshold
python export_to_project.py --threshold 0.6

# Specify output directory
python export_to_project.py --output-dir ../exports

# See all options
python export_to_project.py --help
```

## Copy to Your React Project

```bash
# From the community_graph directory
cp proverbs_communities_threshold_0.5_export.json ~/your-react-app/src/data/
cp proverbs_graph_types.d.ts ~/your-react-app/src/types/
```

## Use in React (TypeScript)

```typescript
import graphData from './data/proverbs_communities_threshold_0.5_export.json';
import type { ProverbsGraphData } from './types/proverbs_graph_types';

function App() {
  const data: ProverbsGraphData = graphData;
  
  return (
    <div>
      <h1>{data.metadata.algorithm} Communities</h1>
      <p>Found {data.graph_stats.num_communities} communities</p>
      <p>Modularity: {data.metadata.modularity?.toFixed(3)}</p>
      
      {data.nodes.map(node => (
        <div key={node.id}>
          <strong>{node.id}</strong>: {node.text}
        </div>
      ))}
    </div>
  );
}
```

## What's in the JSON?

```json
{
  "metadata": {
    "algorithm": "Leiden",
    "threshold": 0.5,
    "modularity": 0.723,
    "created_at": "2026-01-09T..."
  },
  "graph_stats": {
    "total_nodes": 915,
    "num_communities": 15
  },
  "nodes": [
    {
      "id": "Proverbs 1:1",
      "text": "The proverbs of Solomon...",
      "community_id": 3,
      "position_2d": {"x": 0.123, "y": -0.456},
      "position_3d": {"x": 0.123, "y": -0.456, "z": 0.789}
    }
  ],
  "communities": {
    "0": {
      "id": 0,
      "size": 87,
      "color": "#e6194b",
      "members": ["Proverbs 1:1", ...]
    }
  },
  "edges": [
    {
      "source": "Proverbs 1:1",
      "target": "Proverbs 1:2",
      "weight": 0.623,
      "type": "intra"
    }
  ]
}
```

## Common Queries

```typescript
// Get all verses in community 5
const comm5 = data.nodes.filter(n => n.community_id === 5);

// Find high-similarity connections
const strong = data.edges.filter(e => e.weight > 0.7);

// Search verse text
const results = data.nodes.filter(n => 
  n.text.toLowerCase().includes('wisdom')
);

// Get community color
const color = data.communities["5"].color;

// Find most connected verses
const hubs = data.nodes
  .sort((a, b) => b.degree - a.degree)
  .slice(0, 10);
```

## Visualization Libraries

Works great with:
- **D3.js** - Custom SVG graphs
- **Three.js** - 3D WebGL visualization  
- **React Flow** - Node graph UI
- **Cytoscape.js** - Network analysis

## Need More Info?

- **EXPORT_README.md** - Complete documentation
- **IMPLEMENTATION_SUMMARY.md** - Technical details
- **example_export.py** - Code examples

## Troubleshooting

**"Module not found"**
```bash
# Make sure you're in the right directory
cd /Users/sebastianmendoza/Waterloo/Projects/Proverbs/community_graph
python export_to_project.py
```

**"File not found: cleaned_esv.json"**
- Script looks for `../cleaned_esv.json` relative to `community_graph/`
- Make sure it's in the project root

**Want different threshold?**
```bash
python export_to_project.py --threshold 0.6
```

## That's It! 🎉

You now have web-ready, type-safe, self-documenting community detection data.
