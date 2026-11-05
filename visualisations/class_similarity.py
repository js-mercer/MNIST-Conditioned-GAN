import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
from sklearn.decomposition import PCA

from models.generator import Generator


def parse_args() -> argparse.Namespace:
    """Parse paths for generating class-embedding visualisations."""
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Visualise learned MNIST class-label embeddings."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=project_root/"outputs"/"class_similarity",
        help="Directory for the 2D PNG and interactive 3D HTML",
    )
    parser.add_argument(
        "--weights_path",
        type=Path,
        default=project_root/"weights"/"generator_cgan.pth",
        help="Path to trained generator weights",
    )
    return parser.parse_args()


def load_class_embeddings(weights_path: Path):
    """Load the generator's learned class-label embedding vectors."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    state_dict = torch.load(weights_path, map_location=device)
    generator.load_state_dict(state_dict)
    generator.eval()
    return generator.label_embedding.weight.detach().cpu().numpy()


def save_2d_plot(embeddings_2d, variance_retained, output_path: Path):
    """Save a labeled 2D PCA scatter plot."""
    figure, axis = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # Plot each class embedding as a scatter point with its label
    for digit, (x, y) in enumerate(embeddings_2d):
        axis.scatter(
            x,
            y,
            color=colors[digit],
            s=300,
            edgecolors="black",
            zorder=2,
        )
        axis.text(
            x,
            y,
            str(digit),
            fontsize=14,
            fontweight="bold",
            ha="center",
            va="center",
            color="white",
            zorder=3,
        )

    axis.grid(True, linestyle="--", alpha=0.5, zorder=1)
    axis.set_title(
        "2D PCA of learned class-label embeddings "
        f"({variance_retained:.1f}% variance retained)"
    )
    axis.set_xlabel("Principal component 1")
    axis.set_ylabel("Principal component 2")
    figure.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(figure)


def save_3d_plot(embeddings_3d, variance_retained, output_path: Path):
    """Save a standalone, interactive 3D PCA plot as HTML."""
    # Colours for the 10 classes
    colors = [
        "#636EFA",
        "#EF553B",
        "#00CC96",
        "#AB63FA",
        "#FFA15A",
        "#19D3F3",
        "#FF6692",
        "#B6E880",
        "#EB93EB",
        "#FECB52",
    ]
    # Create a 3D scatter plot with Plotly
    figure = go.Figure(
        data=[
            go.Scatter3d(
                x=embeddings_3d[:, 0],
                y=embeddings_3d[:, 1],
                z=embeddings_3d[:, 2],
                mode="markers+text",
                text=[str(digit) for digit in range(10)],
                textposition="middle center",
                marker=dict(
                    size=8,
                    color=colors,
                    line=dict(color="black", width=1),
                ),
                hovertemplate=(
                    "Digit %{text}<br>"
                    "PC1: %{x:.3f}<br>"
                    "PC2: %{y:.3f}<br>"
                    "PC3: %{z:.3f}<extra></extra>"
                ),
            )
        ]
    )
    # Set the layout for the 3D plot with axis labels and title
    figure.update_layout(
        title=(
            "Interactive 3D PCA of learned class-label embeddings "
            f"({variance_retained:.1f}% variance retained)"
        ),
        scene=dict(
            xaxis_title="Principal component 1",
            yaxis_title="Principal component 2",
            zaxis_title="Principal component 3",
        ),
    )
    figure.write_html(output_path, include_plotlyjs=True)


def pca_class_similarity(args: argparse.Namespace):
    # Load the learned class-label embeddings from the trained generator
    embeddings = load_class_embeddings(args.weights_path)
    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(embeddings)
    # Calculate the variance retained by each principal component
    variance_by_component = pca.explained_variance_ratio_

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Save the 2D and 3D PCA visualisations
    plot_2d_path = args.output_dir/"embeddings_pca_2d.png"
    plot_3d_path = args.output_dir/"embeddings_pca_3d.html"

    save_2d_plot(
        embeddings_3d[:, :2],
        variance_by_component[:2].sum() * 100,
        plot_2d_path,
    )
    save_3d_plot(
        embeddings_3d,
        variance_by_component.sum() * 100,
        plot_3d_path,
    )

    print(f"2D PCA plot saved to: {plot_2d_path}")
    print(f"Interactive 3D PCA plot saved to: {plot_3d_path}")


if __name__ == "__main__":
    pca_class_similarity(parse_args())
