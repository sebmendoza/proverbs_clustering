# Community Graph Export

**Standalone script to export Proverbs community detection results for TypeScript/React projects.**

## Quick Start

```bash
cd /Users/sebastianmendoza/Waterloo/Projects/Proverbs/community_graph
python export_to_project.py
```

**Output:**
- `proverbs_communities_threshold_0.5_export.json` - Complete graph data
- `proverbs_graph_types.d.ts` - TypeScript type definitions

## What This Does

1. ✅ Loads verse embeddings
2. ✅ Runs Leiden community detection
3. ✅ Generates 2D and 3D positions
4. ✅ Calculates statistics (modularity, density, clustering)
5. ✅ Exports JSON with full metadata
6. ✅ Creates TypeScript type definitions

## Options

```bash
# Custom threshold
python export_to_project.py --threshold 0.6

# Custom output directory
python export_to_project.py --output-dir ../exports

# See all options
python export_to_project.py --help
```

## Copy to React Project

```bash
cp proverbs_communities_threshold_0.5_export.json ~/your-react-app/src/data/
cp proverbs_graph_types.d.ts ~/your-react-app/src/types/
```

## Use in TypeScript

```typescript
import graphData from './data/proverbs_communities_threshold_0.5_export.json';
import type { ProverbsGraphData } from './types/proverbs_graph_types';

const data: ProverbsGraphData = graphData;

// Type-safe access!
console.log(`Found ${data.graph_stats.num_communities} communities`);
console.log(`Modularity: ${data.metadata.modularity}`);
```

## Documentation

- **QUICK_START.md** - Quick reference guide
- **EXPORT_README.md** - Complete usage documentation
- **IMPLEMENTATION_SUMMARY.md** - Technical details

## Features

- **Self-documenting**: All metadata included
- **Type-safe**: TypeScript definitions
- **Complete**: Nodes, communities, edges, and statistics
- **Flexible**: Configurable threshold and output
- **Standalone**: Runs independently

## Example Output

```json
{
  "metadata": {
    "algorithm": "Leiden",
    "threshold": 0.5,
    "modularity": 0.723,
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
      "size": 87,
      "color": "#e6194b",
      "members": ["Proverbs 1:1", ...]
    }
  },
  "edges": [...]
}
```

## Requirements

- Python 3.7+
- Dependencies from project `requirements.txt`
- Virtual environment activated

## Notes

- First run creates embeddings (~2-5 minutes)
- Subsequent runs use cached embeddings (~15-25 seconds)
- Output size: ~600 KB (varies with threshold)
- Higher threshold = fewer edges = smaller file

## Integration

Works great with:
- **D3.js** - Custom visualizations
- **Three.js** - 3D graphics
- **React Flow** - Node graphs
- **Cytoscape.js** - Network analysis

## Support

See documentation files for:
- Usage examples
- TypeScript integration patterns
- Filtering and querying
- Troubleshooting

---

**Ready to export?** Just run `python export_to_project.py` and copy the files to your React project!
