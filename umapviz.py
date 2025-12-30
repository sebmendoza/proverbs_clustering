import matplotlib.pyplot as plt
import umap
from utils import saveGraph
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path


def explore_umap_parameters_2d(data, cluster_labels, save_filename="umap_2d_parameter_exploration.png"):
    # Create a grid of 2D UMAP plots with different parameter combinations.
    # Define parameter ranges to explore
    n_neighbors_options = [5, 15, 30, 50]
    metric_options = ['euclidean', 'cosine', 'manhattan']
    min_dist_options = [0.0, 0.1, 0.25, 0.5]

    # Create a large figure with subplots
    # We'll do a grid: metrics (rows) x n_neighbors (cols) for min_dist=0.1
    n_rows = len(metric_options)
    n_cols = len(n_neighbors_options)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 18))
    fig.suptitle('UMAP 2D Parameter Exploration: Metrics vs n_neighbors (min_dist=0.1)',
                 fontsize=20, fontweight='bold', y=0.995)

    print("Starting UMAP 2D parameter exploration...")

    for i, metric in enumerate(metric_options):
        for j, n_neighbors in enumerate(n_neighbors_options):
            ax = axes[i, j]

            print(f"  Processing: metric={metric}, n_neighbors={n_neighbors}")

            # Fit UMAP
            reducer = umap.UMAP(
                n_components=2,
                random_state=42,
                n_neighbors=n_neighbors,
                metric=metric,
                min_dist=0.1
            )
            embeddings_2d = reducer.fit_transform(data)

            # Plot
            scatter = ax.scatter(
                embeddings_2d[:, 0],
                embeddings_2d[:, 1],
                c=cluster_labels,
                cmap='tab20',
                alpha=0.6,
                s=20
            )

            ax.set_title(f'{metric}\nn_neighbors={n_neighbors}',
                         fontsize=10, fontweight='bold')
            ax.set_xlabel('UMAP 1', fontsize=8)
            ax.set_ylabel('UMAP 2', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)

    # Add a colorbar
    fig.colorbar(scatter, ax=axes.ravel().tolist(), label='Cluster ID',
                 shrink=0.8, pad=0.01)

    plt.tight_layout()
    saveGraph(save_filename, plt)
    plt.close(fig)
    print(f"Saved 2D exploration to {save_filename}")


def explore_umap_parameters_2d_min_dist(data, cluster_labels, save_filename="umap_2d_min_dist_exploration.png"):
    # Create a grid of 2D UMAP plots exploring min_dist parameter.
    # Define parameter ranges
    n_neighbors_options = [5, 15, 30, 50]
    min_dist_options = [0.0, 0.1, 0.25, 0.5]
    metric = 'cosine'  # Fix metric for this exploration

    n_rows = len(min_dist_options)
    n_cols = len(n_neighbors_options)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 18))
    fig.suptitle(f'UMAP 2D min_dist Exploration: min_dist vs n_neighbors (metric={metric})',
                 fontsize=20, fontweight='bold', y=0.995)

    print("Starting UMAP 2D min_dist exploration...")

    for i, min_dist in enumerate(min_dist_options):
        for j, n_neighbors in enumerate(n_neighbors_options):
            ax = axes[i, j]

            print(
                f"  Processing: min_dist={min_dist}, n_neighbors={n_neighbors}")

            # Fit UMAP
            reducer = umap.UMAP(
                n_components=2,
                random_state=42,
                n_neighbors=n_neighbors,
                metric=metric,
                min_dist=min_dist
            )
            embeddings_2d = reducer.fit_transform(data)

            # Plot
            scatter = ax.scatter(
                embeddings_2d[:, 0],
                embeddings_2d[:, 1],
                c=cluster_labels,
                cmap='tab20',
                alpha=0.6,
                s=20
            )

            ax.set_title(f'min_dist={min_dist}\nn_neighbors={n_neighbors}',
                         fontsize=10, fontweight='bold')
            ax.set_xlabel('UMAP 1', fontsize=8)
            ax.set_ylabel('UMAP 2', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)

    # Add a colorbar
    fig.colorbar(scatter, ax=axes.ravel().tolist(), label='Cluster ID',
                 shrink=0.8, pad=0.01)

    plt.tight_layout()
    saveGraph(save_filename, plt)
    plt.close(fig)
    print(f"Saved min_dist exploration to {save_filename}")


def explore_umap_parameters_3d(data, cluster_labels, save_filename_prefix="umap_3d_exploration"):
    # Create multiple 3D UMAP plots with different parameters using Plotly (interactive grid layout).
    # Define parameter ranges - fewer than 2D to avoid overcrowding
    n_neighbors_options = [5, 30]
    metric_options = ['euclidean', 'cosine']
    min_dist = 0.1  # Keep this fixed

    n_rows = len(metric_options)
    n_cols = len(n_neighbors_options)

    # Create subplots with 3D scenes
    subplot_titles = []
    for metric in metric_options:
        for n_neighbors in n_neighbors_options:
            subplot_titles.append(f'{metric}, n_neighbors={n_neighbors}')

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        specs=[[{'type': 'scatter3d'}
                for _ in range(n_cols)] for _ in range(n_rows)],
        vertical_spacing=0.1,
        horizontal_spacing=0.05
    )

    print("Starting UMAP 3D parameter exploration...")

    plot_idx = 0
    for i, metric in enumerate(metric_options):
        for j, n_neighbors in enumerate(n_neighbors_options):
            print(
                f"  Processing 3D: metric={metric}, n_neighbors={n_neighbors}")

            # Fit UMAP
            reducer = umap.UMAP(
                n_components=3,
                random_state=42,
                n_neighbors=n_neighbors,
                metric=metric,
                min_dist=min_dist
            )
            embeddings_3d = reducer.fit_transform(data)

            # Add 3D scatter plot
            fig.add_trace(
                go.Scatter3d(
                    x=embeddings_3d[:, 0],
                    y=embeddings_3d[:, 1],
                    z=embeddings_3d[:, 2],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=cluster_labels,
                        colorscale='Viridis',
                        # Only show colorbar for first plot
                        showscale=(plot_idx == 0),
                        colorbar=dict(title="Cluster",
                                      x=1.05) if plot_idx == 0 else None,
                        opacity=0.7
                    ),
                    text=[f'Cluster {label}' for label in cluster_labels],
                    hovertemplate='Cluster %{text}<br>%{x:.2f}, %{y:.2f}, %{z:.2f}<extra></extra>',
                    showlegend=False
                ),
                row=i+1, col=j+1
            )

            # Update axes labels for this subplot
            scene_name = 'scene' if plot_idx == 0 else f'scene{plot_idx+1}'
            fig.update_layout(**{
                scene_name: dict(
                    xaxis_title='UMAP 1',
                    yaxis_title='UMAP 2',
                    zaxis_title='UMAP 3',
                )
            })

            plot_idx += 1

    fig.update_layout(
        title_text=f'UMAP 3D Parameter Exploration (min_dist={min_dist})',
        title_font_size=20,
        width=1600,
        height=1200
    )

    # Save and show
    outdir = Path("visualizations")
    filename_html = f"{save_filename_prefix}_grid.html"
    filename_png = f"{save_filename_prefix}_grid.png"
    fig.write_html(outdir / filename_html)

    # Try to save PNG, but continue if Chrome/Kaleido not available
    try:
        fig.write_image(outdir / filename_png, width=1600, height=1200)
        print(
            f"Saved 3D exploration to {filename_html} (interactive) and {filename_png}")
    except Exception as e:
        print(
            f"Saved 3D exploration to {filename_html} (PNG export skipped: {type(e).__name__})")

    fig.show()


def explore_umap_parameters_3d_individual(data, cluster_labels, save_filename_prefix="umap_3d_individual"):
    # Create separate 3D plots for each parameter combination using Plotly (better quality individual plots).
    # Define parameter combinations to explore
    param_combinations = [
        {'n_neighbors': 5, 'metric': 'euclidean', 'min_dist': 0.1},
        {'n_neighbors': 5, 'metric': 'cosine', 'min_dist': 0.1},
        {'n_neighbors': 15, 'metric': 'euclidean', 'min_dist': 0.1},
        {'n_neighbors': 15, 'metric': 'cosine', 'min_dist': 0.1},
        {'n_neighbors': 30, 'metric': 'euclidean', 'min_dist': 0.1},
        {'n_neighbors': 30, 'metric': 'cosine', 'min_dist': 0.1},
        {'n_neighbors': 15, 'metric': 'cosine', 'min_dist': 0.0},
        {'n_neighbors': 15, 'metric': 'cosine', 'min_dist': 0.5},
    ]

    print("Starting individual 3D UMAP plots...")
    outdir = Path("visualizations")

    for idx, params in enumerate(param_combinations):
        print(f"  Processing {idx+1}/{len(param_combinations)}: {params}")

        # Fit UMAP
        reducer = umap.UMAP(
            n_components=3,
            random_state=42,
            **params
        )
        embeddings_3d = reducer.fit_transform(data)

        # Create Plotly 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=embeddings_3d[:, 0],
            y=embeddings_3d[:, 1],
            z=embeddings_3d[:, 2],
            mode='markers',
            marker=dict(
                size=4,
                color=cluster_labels,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Cluster ID"),
                opacity=0.7
            ),
            text=[f'Cluster {label}' for label in cluster_labels],
            hovertemplate='<b>Cluster %{text}</b><br>UMAP1: %{x:.2f}<br>UMAP2: %{y:.2f}<br>UMAP3: %{z:.2f}<extra></extra>'
        )])

        title = f"UMAP 3D: n_neighbors={params['n_neighbors']}, " \
            f"metric={params['metric']}, min_dist={params['min_dist']}"

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='UMAP 1',
                yaxis_title='UMAP 2',
                zaxis_title='UMAP 3'
            ),
            width=1200,
            height=900
        )

        # Save as HTML (interactive) and PNG (static if Chrome available)
        filename_base = f"{save_filename_prefix}_n{params['n_neighbors']}_" \
            f"{params['metric']}_md{params['min_dist']}"
        fig.write_html(outdir / f"{filename_base}.html")

        # Try to save PNG, but continue if Chrome/Kaleido not available
        try:
            fig.write_image(outdir / f"{filename_base}.png")
        except Exception:
            pass  # Silently skip PNG for individual plots

    print(f"Saved {len(param_combinations)} individual 3D plots as HTML (PNG export skipped if Chrome not available)")
