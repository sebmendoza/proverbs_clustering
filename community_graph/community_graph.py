import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import igraph as ig
import leidenalg
from pyvis.network import Network
import plotly.graph_objects as go


def cosine_similarity_histogram(embeddings_matrix):
    # Cosine similarity: measures angle between vectors, range [-1, 1]
    # 1 = identical direction, 0 = orthogonal, -1 = opposite
    # For embeddings, typically 0.3-0.7 means "somewhat similar"
    similarity_matrix = cosine_similarity(embeddings_matrix)
    plt.hist(similarity_matrix, bins=100)
    plt.show()


def create_similarity_score_histogram_with_buckets(similarity_matrix):
    # Create a similarity score histogram with buckets.
    plt.hist(similarity_matrix, bins=100)
    plt.show()


def evaluate_threshold_edge_counts(similarity_matrix, total_possible):
    # Graph construction: we create edges between verses if their similarity exceeds a threshold
    # This function helps choose the threshold by showing how many edges you'd get
    # Too low threshold = too many edges (graph is dense, hard to find communities)
    # Too high threshold = too few edges (graph is sparse, many isolated nodes)
    thresholds = [0.3, 0.35, 0.4, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5]
    for t in thresholds:
        # Similarity matrix is symmetric (similarity[i,j] = similarity[j,i])
        # So we count edges twice, divide by 2 to get actual edge count
        num_edges = (similarity_matrix > t).sum() / 2

        pct = (num_edges / total_possible) * 100
        print(f"Threshold {t}: {num_edges} edges ({pct:.2f}%)")


def create_community_graph_with_threshold(verse_refs, similarity_matrix, threshold_value):
    # Create a community graph: verses are nodes, edges connect similar verses
    # This is the foundation for community detection - we need a graph structure first
    G = nx.Graph()  # Undirected graph (if A is similar to B, B is similar to A)
    G.add_nodes_from(verse_refs)  # Each verse is a node

    # Add edges between verses with similarity above threshold
    # Only check upper triangle (i < j) since matrix is symmetric
    for i in range(len(verse_refs)):
        for j in range(i+1, len(verse_refs)):
            if similarity_matrix[i][j] > threshold_value:
                # Edge weight = similarity score (higher = stronger connection)
                G.add_edge(verse_refs[i], verse_refs[j],
                           weight=similarity_matrix[i][j])
    return G


def analyze_community_graph(G):
    # Graph analysis: understand the structure before community detection
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")

    # Connected components: groups of nodes that can reach each other via edges
    # If graph has multiple components, they're completely disconnected (no path between them)
    # This is important: community detection works within each component separately
    components = list(nx.connected_components(G))
    num_components = len(components)
    sizes = sorted([len(c) for c in components], reverse=True)

    print(f"Connected components: {num_components}")
    if sizes:
        print(
            f"  Largest: {sizes[0]}, Smallest: {sizes[-1]}, Avg: {np.mean(sizes):.1f}")
        print(f"  Top 5 sizes: {sizes[:5]}")
        # Isolated nodes: verses with no similar verses (singleton components)
        # These are outliers or very unique verses
        print(f"  Isolated nodes: {sum(1 for s in sizes if s == 1)}")


def evaluate_threshold_clustering(verse_refs, similarity_matrix):
    print("=" * 80)
    print("THRESHOLD CLUSTERING ANALYSIS")
    print("=" * 80)

    thresholds = [0.35, 0.40, 0.42, 0.44, 0.46, 0.48,
                  0.50, 0.56, 0.58, 0.6, 0.62, 0.66, 0.68, 0.7]

    for t in thresholds:
        print(f"\n--- THRESHOLD {t} ---")
        G = create_community_graph_with_threshold(
            verse_refs, similarity_matrix, t)
        analyze_community_graph(G)

    print("\n" + "=" * 80)


def run_leiden_algo(nxgraph):
    # Leiden algorithm: state-of-the-art community detection algorithm
    # Finds communities (clusters) by optimizing modularity: measures how much more
    # edges exist within communities vs what you'd expect by random chance
    # Higher modularity = better community structure (dense within, sparse between)

    seed = 42  # Random seed for reproducibility
    # Convert NetworkX graph to igraph format (Leiden algorithm uses igraph)
    g_igraph = ig.Graph.from_networkx(nxgraph)

    # ModularityVertexPartition: optimize for modularity metric
    # This is the most common objective for community detection
    partition = leidenalg.find_partition(g_igraph,
                                         leidenalg.ModularityVertexPartition,
                                         seed=seed)

    # Convert partition result back to dictionary mapping node -> community_id
    communities = {}
    for node_idx, comm_id in enumerate(partition.membership):
        vertex_name = g_igraph.vs[node_idx]['_nx_name']
        communities[vertex_name] = comm_id
    print(f"Found {len(set(partition.membership))} communities")

    return communities


def visualize_initial_graph_2d(nxgraph):
    net_before = Network(height='900px', width='100%',
                         bgcolor='#222222', font_color='white',
                         )
    for node in nxgraph.nodes():
        node_id = str(node)
        net_before.add_node(node_id,
                            label=node_id,
                            color='#97c2fc',  # Single blue color
                            title=f"Verse: {node_id}",
                            size=10)

    # Note: edges are commented out to reduce visual clutter in initial graph
    # Uncomment the lines below if you want to see all connections
    # for edge in nxgraph.edges(data=True):
    #     weight = float(edge[2].get('weight', 1))
    #     net_before.add_edge(str(edge[0]), str(edge[1]), value=weight)

    # Barnes-Hut force-directed layout: simulates physics to position nodes
    # Nodes repel each other, edges act like springs pulling connected nodes together
    # This creates visual clusters: similar verses (connected) appear close together
    net_before.barnes_hut(gravity=-5000,  # Negative = repulsion between all nodes
                          central_gravity=0.3,  # Slight pull toward center
                          spring_length=150,  # Preferred edge length
                          spring_strength=0.001)  # How strong edges pull

    net_before.write_html('proverbs_before_leiden.html', notebook=False)
    print(
        f"Before: {nxgraph.number_of_nodes()} nodes, {nxgraph.number_of_edges()} edges")


def visualize_leiden_graph_2d(nxgraph, node_to_community):
    net_after = Network(height='900px', width='100%',
                        bgcolor='#222222', font_color='white')

    def get_community_color(comm_id):
        colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
                  '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
                  '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
                  '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080',]
        return colors[comm_id % len(colors)]

    # Add nodes colored by community
    for node in nxgraph.nodes():
        node_id = str(node)
        comm_id = node_to_community[node]
        net_after.add_node(node_id,
                           label=node_id,
                           color=get_community_color(comm_id),
                           title=f"Verse: {node_id}<br>Community: {comm_id}",
                           size=10)

    # Add edges only within the same community
    # This visualization shows intra-community connections only
    # Edges between communities are hidden to make communities visually distinct
    for edge in nxgraph.edges(data=True):
        node1, node2 = edge[0], edge[1]
        # Only add edge if both nodes are in the same community
        if node_to_community[node1] == node_to_community[node2]:
            weight = float(edge[2].get('weight', 1))
            net_after.add_edge(str(node1), str(node2), value=weight)

    net_after.barnes_hut(gravity=-5000, central_gravity=0.3,
                         spring_length=150, spring_strength=0.001)
    net_after.write_html('proverbs_after_leiden.html', notebook=False)


def visualize_initial_graph_3d(nxgraph, embeddings_matrix, verse_refs):
    # Create a 3D Plotly visualization of the graph before community detection.
    from sklearn.decomposition import PCA

    # PCA (Principal Component Analysis): reduces high-dimensional embeddings to 3D
    # Finds the 3 directions with most variance (most information)
    # This preserves as much structure as possible while making it visualizable
    pca = PCA(n_components=3)
    positions_3d = pca.fit_transform(embeddings_matrix)

    # Create a mapping from verse_ref to position
    verse_to_pos = {verse_refs[i]: positions_3d[i]
                    for i in range(len(verse_refs))}

    # Prepare node data
    node_x = []
    node_y = []
    node_z = []
    node_text = []

    for node in nxgraph.nodes():
        if node in verse_to_pos:
            pos = verse_to_pos[node]
            node_x.append(pos[0])
            node_y.append(pos[1])
            node_z.append(pos[2])
            node_text.append(f"Verse: {node}")

    # Prepare edge data for 3D visualization
    # Plotly needs edges as line segments: [x1, x2, None, x3, x4, None, ...]
    # The None values create breaks between separate line segments
    edge_x = []
    edge_y = []
    edge_z = []

    for edge in nxgraph.edges():
        if edge[0] in verse_to_pos and edge[1] in verse_to_pos:
            pos0 = verse_to_pos[edge[0]]
            pos1 = verse_to_pos[edge[1]]
            # Each edge is drawn as: start point, end point, None (break)
            edge_x.extend([pos0[0], pos1[0], None])
            edge_y.extend([pos0[1], pos1[1], None])
            edge_z.extend([pos0[2], pos1[2], None])

    # Create edge trace
    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='rgba(125, 125, 125, 0.3)', width=1),
        hoverinfo='none',
        name='Connections'
    )

    # Create node trace
    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        marker=dict(
            size=5,
            color='#97c2fc',
            line=dict(color='white', width=0.5)
        ),
        text=node_text,
        hoverinfo='text',
        name='Verses'
    )

    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace])

    fig.update_layout(
        title=dict(
            text=f'Proverbs Graph - Before Leiden (3D)<br>{nxgraph.number_of_nodes()} nodes, {nxgraph.number_of_edges()} edges',
            x=0.5,
            xanchor='center'
        ),
        showlegend=False,
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False, title=''),
            yaxis=dict(showbackground=False, showticklabels=False, title=''),
            zaxis=dict(showbackground=False, showticklabels=False, title=''),
            bgcolor='rgb(30, 30, 30)'
        ),
        paper_bgcolor='rgb(20, 20, 20)',
        plot_bgcolor='rgb(20, 20, 20)',
        font=dict(color='white'),
        height=900,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.write_html('proverbs_before_leiden_3d.html')
    print(
        f"3D Before: {nxgraph.number_of_nodes()} nodes, {nxgraph.number_of_edges()} edges")
    print("Saved to: proverbs_before_leiden_3d.html")


def visualize_leiden_graph_3d(nxgraph, node_to_community, embeddings_matrix, verse_refs):
    # Create a 3D Plotly visualization of the graph after community detection.
    from sklearn.decomposition import PCA

    # Reduce embeddings to 3D for visualization
    pca = PCA(n_components=3)
    positions_3d = pca.fit_transform(embeddings_matrix)

    # Create a mapping from verse_ref to position
    verse_to_pos = {verse_refs[i]: positions_3d[i]
                    for i in range(len(verse_refs))}

    # Color palette for communities
    color_palette = [
        '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
        '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
        '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
        '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080'
    ]

    def get_community_color(comm_id):
        return color_palette[comm_id % len(color_palette)]

    # Group nodes by community
    communities_dict = {}
    for node, comm_id in node_to_community.items():
        if comm_id not in communities_dict:
            communities_dict[comm_id] = []
        communities_dict[comm_id].append(node)

    # Prepare edge data (only edges within the same community)
    edge_x = []
    edge_y = []
    edge_z = []

    for edge in nxgraph.edges():
        node1, node2 = edge[0], edge[1]
        # Only add edge if both nodes are in the same community
        if (node1 in verse_to_pos and node2 in verse_to_pos and
                node_to_community[node1] == node_to_community[node2]):
            pos0 = verse_to_pos[node1]
            pos1 = verse_to_pos[node2]
            edge_x.extend([pos0[0], pos1[0], None])
            edge_y.extend([pos0[1], pos1[1], None])
            edge_z.extend([pos0[2], pos1[2], None])

    # Create edge trace
    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='rgba(125, 125, 125, 0.2)', width=1),
        hoverinfo='none',
        name='Connections',
        showlegend=False
    )

    # Create node traces (one per community for legend)
    node_traces = []

    for comm_id in sorted(communities_dict.keys()):
        nodes = communities_dict[comm_id]
        node_x = []
        node_y = []
        node_z = []
        node_text = []

        for node in nodes:
            if node in verse_to_pos:
                pos = verse_to_pos[node]
                node_x.append(pos[0])
                node_y.append(pos[1])
                node_z.append(pos[2])
                node_text.append(f"Verse: {node}<br>Community: {comm_id}")

        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers',
            marker=dict(
                size=5,
                color=get_community_color(comm_id),
                line=dict(color='white', width=0.5)
            ),
            text=node_text,
            hoverinfo='text',
            name=f'Community {comm_id} ({len(nodes)} verses)'
        )
        node_traces.append(node_trace)

    # Create figure
    fig = go.Figure(data=[edge_trace] + node_traces)

    fig.update_layout(
        title=dict(
            text=f'Proverbs Graph - After Leiden (3D)<br>{nxgraph.number_of_nodes()} nodes, {len(communities_dict)} communities',
            x=0.5,
            xanchor='center'
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(0,0,0,0.5)'
        ),
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False, title=''),
            yaxis=dict(showbackground=False, showticklabels=False, title=''),
            zaxis=dict(showbackground=False, showticklabels=False, title=''),
            bgcolor='rgb(30, 30, 30)'
        ),
        paper_bgcolor='rgb(20, 20, 20)',
        plot_bgcolor='rgb(20, 20, 20)',
        font=dict(color='white'),
        height=900,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.write_html('proverbs_after_leiden_3d.html')
    print(
        f"3D After: {nxgraph.number_of_nodes()} nodes, {len(communities_dict)} communities")
    print("Saved to: proverbs_after_leiden_3d.html")


def create_community_graph(data, verse_refs):
    # Create a community graph from the data.
    similarity_matrix = cosine_similarity(data)

    # create_similarity_score_histogram_with_buckets(data)

    # evaluate_threshold_edge_counts(
    # similarity_matrix=similarity_matrix,  total_possible=(len(data) * (len(data) - 1)) / 2)

    # cosine_similarity_histogram(data)
    # evaluate_threshold_clustering(
    #     verse_refs=verse_refs, similarity_matrix=similarity_matrix)
    THRESHOLD = 0.48

    community_graph = create_community_graph_with_threshold(
        verse_refs=verse_refs, similarity_matrix=similarity_matrix, threshold_value=THRESHOLD)

    # Extract largest connected component: the biggest group of connected verses
    # We focus on this because community detection works best on connected graphs
    # Isolated nodes/small components are often outliers and can be analyzed separately
    largest_cc = max(nx.connected_components(community_graph), key=len)
    G_main = community_graph.subgraph(largest_cc).copy()

    # Get embeddings for nodes in G_main only
    # We need to align embeddings with the subgraph nodes for visualization
    # This creates a mapping: which embeddings correspond to which nodes in the subgraph
    node_indices = [verse_refs.index(node) for node in G_main.nodes()]
    G_main_embeddings = data[node_indices]
    G_main_verse_refs = [verse_refs[i] for i in node_indices]

    node_to_community = run_leiden_algo(G_main)

    # 2D visualizations
    visualize_initial_graph_2d(G_main)
    visualize_leiden_graph_2d(G_main, node_to_community)

    # 3D visualizations
    # visualize_initial_graph_3d(G_main, G_main_embeddings, G_main_verse_refs)
    # visualize_leiden_graph_3d(
    # G_main, node_to_community, G_main_embeddings, G_main_verse_refs)
