"""
Standalone Export Script for Community Detection Results

This script runs the complete community detection pipeline and exports
the results to JSON for TypeScript/React projects.

Usage:
    python export_to_project.py [--threshold THRESHOLD]

Output:
    - proverbs_communities_threshold_{threshold}_export.json
    - proverbs_graph_types.d.ts
"""

import json
import sys
import argparse
from pathlib import Path
import networkx as nx
from datetime import datetime
from typing import Dict, List, Any, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import igraph as ig
import leidenalg

# Add parent directory to path to import from project root
sys.path.append(str(Path(__file__).parent.parent))
from utils import getDataFromJson
from embeddings import get_or_create_embeddings, embeddings_dict_to_array


def verse_tuple_to_string(chapter: str, verse: str) -> str:
    """Convert (chapter, verse) tuple to 'Proverbs X:Y' string."""
    return f"Proverbs {chapter}:{verse}"


def get_verse_text(verses_dict: Dict, chapter: str, verse: str) -> str:
    """Get verse text from the verses dictionary."""
    return verses_dict.get(chapter, {}).get(verse, "")


def get_community_color(comm_id: int) -> str:
    """Get the color hex code for a community ID (matches visualization colors)."""
    colors = [
        '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
        '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
        '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
        '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080'
    ]
    return colors[comm_id % len(colors)]


def create_similarity_graph(embeddings_matrix: np.ndarray, 
                            verse_refs: List[Tuple[str, str]], 
                            threshold: float) -> nx.Graph:
    """
    Create a NetworkX graph from embeddings using cosine similarity.
    
    Args:
        embeddings_matrix: Matrix of verse embeddings
        verse_refs: List of (chapter, verse) tuples
        threshold: Similarity threshold for edge creation
        
    Returns:
        NetworkX graph with edges above threshold
    """
    print(f"\nCreating similarity graph with threshold {threshold}...")
    similarity_matrix = cosine_similarity(embeddings_matrix)
    
    G = nx.Graph()
    G.add_nodes_from(verse_refs)
    
    # Add edges between verses with similarity above threshold
    edge_count = 0
    for i in range(len(verse_refs)):
        for j in range(i+1, len(verse_refs)):
            if similarity_matrix[i][j] > threshold:
                G.add_edge(verse_refs[i], verse_refs[j], 
                          weight=similarity_matrix[i][j])
                edge_count += 1
    
    print(f"Created graph with {G.number_of_nodes()} nodes and {edge_count} edges")
    return G


def run_leiden_algorithm(graph: nx.Graph) -> Dict[Tuple[str, str], int]:
    """
    Run Leiden community detection algorithm.
    
    Args:
        graph: NetworkX graph
        
    Returns:
        Dictionary mapping node to community ID
    """
    print("\nRunning Leiden community detection...")
    seed = 42
    
    # Convert to igraph
    g_igraph = ig.Graph.from_networkx(graph)
    
    # Run Leiden algorithm
    partition = leidenalg.find_partition(
        g_igraph,
        leidenalg.ModularityVertexPartition,
        seed=seed
    )
    
    # Convert to dictionary
    communities = {}
    for node_idx, comm_id in enumerate(partition.membership):
        vertex_name = g_igraph.vs[node_idx]['_nx_name']
        communities[vertex_name] = comm_id
    
    num_communities = len(set(partition.membership))
    print(f"Found {num_communities} communities")
    
    return communities


def get_representative_nodes(graph: nx.Graph, 
                            community_nodes: List[Tuple[str, str]], 
                            top_n: int = 5) -> List[Tuple[str, str]]:
    """Find the most representative (central) nodes in a community."""
    if not community_nodes:
        return []
    
    subgraph = graph.subgraph(community_nodes)
    node_degrees = [(node, subgraph.degree(node)) for node in community_nodes]
    node_degrees.sort(key=lambda x: x[1], reverse=True)
    
    return [node for node, _ in node_degrees[:top_n]]


def export_to_json(
    graph: nx.Graph,
    node_to_community: Dict[Tuple[str, str], int],
    verse_refs: List[Tuple[str, str]],
    embeddings_matrix: np.ndarray,
    threshold: float,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Export community detection results to JSON.
    
    Args:
        graph: NetworkX graph with communities
        node_to_community: Mapping of nodes to community IDs
        verse_refs: List of verse references
        embeddings_matrix: Verse embeddings for PCA
        threshold: Similarity threshold used
        output_dir: Directory to save output files
        
    Returns:
        Dictionary containing all exported data
    """
    print("\n" + "="*80)
    print("EXPORTING DATA FOR WEB PROJECT")
    print("="*80)
    
    # Load verse texts
    verses_dict = getDataFromJson()
    
    # Generate 2D positions
    print("\nGenerating 2D layout positions...")
    positions_2d = nx.spring_layout(graph, k=2, iterations=50, seed=42)
    
    # Generate 3D positions using PCA
    print("Generating 3D positions using PCA...")
    pca = PCA(n_components=3, random_state=42)
    
    # Get embeddings for nodes in graph
    node_to_idx = {verse_refs[i]: i for i in range(len(verse_refs))}
    graph_indices = [node_to_idx[node] for node in graph.nodes()]
    graph_embeddings = embeddings_matrix[graph_indices]
    positions_3d_array = pca.fit_transform(graph_embeddings)
    
    # Create positions_3d dict
    positions_3d = {}
    for i, node in enumerate(graph.nodes()):
        positions_3d[node] = tuple(positions_3d_array[i])
    
    # Calculate modularity
    try:
        modularity = nx.algorithms.community.modularity(
            graph, 
            [set([n for n in graph.nodes() if node_to_community[n] == comm]) 
             for comm in set(node_to_community.values())]
        )
    except:
        modularity = None
    
    # Initialize export structure
    export_data = {
        "metadata": {
            "algorithm": "Leiden",
            "algorithm_version": "leidenalg",
            "partition_type": "ModularityVertexPartition",
            "threshold": float(threshold),
            "seed": 42,
            "modularity": float(modularity) if modularity is not None else None,
            "created_at": datetime.now().isoformat(),
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "source_text": "ESV Bible - Book of Proverbs",
            "description": f"Community detection using Leiden algorithm with similarity threshold {threshold}"
        },
        
        "graph_stats": {
            "total_nodes": int(graph.number_of_nodes()),
            "total_edges": int(graph.number_of_edges()),
            "num_communities": int(len(set(node_to_community.values()))),
            "density": float(nx.density(graph)),
            "avg_clustering": float(nx.average_clustering(graph)),
            "avg_degree": float(sum(dict(graph.degree()).values()) / graph.number_of_nodes()) if graph.number_of_nodes() > 0 else 0.0
        },
        
        "nodes": [],
        "communities": {},
        "edges": []
    }
    
    print(f"\nExporting {export_data['graph_stats']['total_nodes']} nodes...")
    
    # Export nodes
    for node in graph.nodes():
        chapter, verse = node
        verse_id = verse_tuple_to_string(chapter, verse)
        verse_text = get_verse_text(verses_dict, chapter, verse)
        
        node_data = {
            "id": verse_id,
            "text": verse_text,
            "community_id": int(node_to_community[node]),
            "position_2d": {
                "x": float(positions_2d[node][0]),
                "y": float(positions_2d[node][1])
            },
            "position_3d": {
                "x": float(positions_3d[node][0]),
                "y": float(positions_3d[node][1]),
                "z": float(positions_3d[node][2])
            },
            "degree": int(graph.degree(node)),
            "chapter": int(chapter),
            "verse": int(verse)
        }
        
        export_data["nodes"].append(node_data)
    
    print(f"Exporting {export_data['graph_stats']['num_communities']} communities...")
    
    # Export communities with statistics
    community_groups = {}
    for node, comm_id in node_to_community.items():
        if comm_id not in community_groups:
            community_groups[comm_id] = []
        community_groups[comm_id].append(node)
    
    for comm_id, members in community_groups.items():
        subgraph = graph.subgraph(members)
        
        # Get representative verses
        representative_nodes = get_representative_nodes(graph, members, top_n=5)
        representative_verse_ids = [verse_tuple_to_string(ch, v) for ch, v in representative_nodes]
        
        # Convert member tuples to verse ID strings
        member_verse_ids = [verse_tuple_to_string(ch, v) for ch, v in members]
        
        # Calculate internal vs external edges
        internal_edges = subgraph.number_of_edges()
        external_edges = sum(1 for node in members 
                           for neighbor in graph.neighbors(node)
                           if node_to_community.get(neighbor) != comm_id) / 2
        
        export_data["communities"][str(comm_id)] = {
            "id": int(comm_id),
            "size": int(len(members)),
            "color": get_community_color(comm_id),
            "members": member_verse_ids,
            "representative_verses": representative_verse_ids,
            "stats": {
                "density": float(nx.density(subgraph)) if len(members) > 1 else 0.0,
                "avg_degree": float(sum(dict(subgraph.degree()).values()) / len(members)) if members else 0.0,
                "internal_edges": int(internal_edges),
                "external_edges": int(external_edges),
                "avg_clustering": float(nx.average_clustering(subgraph)) if len(members) > 2 else 0.0
            }
        }
    
    print(f"Exporting {export_data['graph_stats']['total_edges']} edges...")
    
    # Export edges
    for edge in graph.edges(data=True):
        source, target = edge[0], edge[1]
        source_id = verse_tuple_to_string(source[0], source[1])
        target_id = verse_tuple_to_string(target[0], target[1])
        weight = edge[2].get('weight', 0)
        same_community = node_to_community[source] == node_to_community[target]
        
        export_data["edges"].append({
            "source": source_id,
            "target": target_id,
            "weight": float(weight),  # Convert numpy float to Python float
            "same_community": bool(same_community),  # Ensure it's a Python bool
            "community_id": int(node_to_community[source]) if same_community else None,  # Convert to Python int
            "type": "intra" if same_community else "inter"
        })
    
    # Write to JSON
    output_path = output_dir / f"proverbs_communities_threshold_{threshold}_export.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"✓ Export complete!")
    print(f"{'='*80}")
    print(f"Output file: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")
    print(f"\nSummary:")
    print(f"  - {export_data['graph_stats']['total_nodes']} verses")
    print(f"  - {export_data['graph_stats']['num_communities']} communities")
    print(f"  - {export_data['graph_stats']['total_edges']} connections")
    print(f"  - Modularity: {modularity:.3f}" if modularity else "  - Modularity: N/A")
    print(f"  - Threshold: {threshold}")
    print(f"\nCommunity sizes:")
    for comm_id in sorted(community_groups.keys()):
        size = len(community_groups[comm_id])
        print(f"  Community {comm_id}: {size} verses")
    print(f"{'='*80}\n")
    
    return export_data


def generate_typescript_types(output_dir: Path):
    """Generate TypeScript type definitions for the exported data."""
    
    typescript_types = '''/**
 * Type definitions for Proverbs Community Detection Export
 * Auto-generated from Python export script
 */

export interface ProverbsGraphData {
  metadata: Metadata;
  graph_stats: GraphStats;
  nodes: Node[];
  communities: Record<string, Community>;
  edges: Edge[];
}

export interface Metadata {
  algorithm: string;
  algorithm_version: string;
  partition_type: string;
  threshold: number;
  seed: number;
  modularity: number | null;
  created_at: string;
  embedding_model: string;
  source_text: string;
  description: string;
}

export interface GraphStats {
  total_nodes: number;
  total_edges: number;
  num_communities: number;
  density: number;
  avg_clustering: number;
  avg_degree: number;
}

export interface Node {
  id: string;
  text: string;
  community_id: number;
  position_2d: Position2D;
  position_3d: Position3D;
  degree: number;
  chapter: number;
  verse: number;
}

export interface Position2D {
  x: number;
  y: number;
}

export interface Position3D {
  x: number;
  y: number;
  z: number;
}

export interface Community {
  id: number;
  size: number;
  color: string;
  members: string[];
  representative_verses: string[];
  stats: CommunityStats;
}

export interface CommunityStats {
  density: number;
  avg_degree: number;
  internal_edges: number;
  external_edges: number;
  avg_clustering: number;
}

export interface Edge {
  source: string;
  target: string;
  weight: number;
  same_community: boolean;
  community_id: number | null;
  type: "intra" | "inter";
}
'''
    
    output_path = output_dir / "proverbs_graph_types.d.ts"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(typescript_types)
    
    print(f"✓ TypeScript definitions written to: {output_path}\n")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Export Proverbs community detection results for TypeScript/React"
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Similarity threshold for edge creation (default: 0.5)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='.',
        help='Output directory for export files (default: current directory)'
    )
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("PROVERBS COMMUNITY DETECTION EXPORT")
    print("="*80)
    print(f"Threshold: {args.threshold}")
    print(f"Output directory: {output_dir.absolute()}")
    print("="*80)
    
    # Load embeddings
    print("\nLoading embeddings...")
    embeddings_dict = get_or_create_embeddings()
    embeddings, verse_refs = embeddings_dict_to_array(embeddings_dict)
    embeddings_matrix = np.array(embeddings)
    print(f"Loaded {len(embeddings_matrix)} verses with {embeddings_matrix.shape[1]}-dimensional embeddings")
    
    # Create similarity graph
    graph = create_similarity_graph(embeddings_matrix, verse_refs, args.threshold)
    
    # Extract largest connected component
    print("\nExtracting largest connected component...")
    largest_cc = max(nx.connected_components(graph), key=len)
    G_main = graph.subgraph(largest_cc).copy()
    print(f"Largest component has {G_main.number_of_nodes()} nodes")
    
    # Get embeddings for main graph
    node_to_idx = {verse_refs[i]: i for i in range(len(verse_refs))}
    G_main_indices = [node_to_idx[node] for node in G_main.nodes()]
    G_main_embeddings = embeddings_matrix[G_main_indices]
    G_main_verse_refs = [verse_refs[i] for i in G_main_indices]
    
    # Run community detection
    node_to_community = run_leiden_algorithm(G_main)
    
    # Export to JSON
    export_to_json(
        graph=G_main,
        node_to_community=node_to_community,
        verse_refs=verse_refs,
        embeddings_matrix=embeddings_matrix,
        threshold=args.threshold,
        output_dir=output_dir
    )
    
    # Generate TypeScript types
    generate_typescript_types(output_dir)
    
    print("="*80)
    print("EXPORT COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("1. Copy the JSON file to your React project:")
    print(f"   cp {output_dir / f'proverbs_communities_threshold_{args.threshold}_export.json'} your-react-app/src/data/")
    print("\n2. Copy the TypeScript types:")
    print(f"   cp {output_dir / 'proverbs_graph_types.d.ts'} your-react-app/src/types/")
    print("\n3. Import and use in your React components!")
    print("="*80)


if __name__ == "__main__":
    main()
