import argparse
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from models.generator import Generator

def parse_args() -> argparse.Namespace:
    """Defines and parses command line arguments for generating digit and hybrid digit images."""
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Generate a digit (or hybrid digit) image using a trained Conditioned MNIST GAN")
    
    # Parameters
    parser.add_argument("--base_digit", type=int, default=0, help="Base digit for image generation (0-9)")
    parser.add_argument("--digit2", type=int, default=5, help="Second digit for interpolation")
    parser.add_argument("--blend", type=float, default=0.0, help="Proportion of second class embedding (0.0 to 1.0)")
    parser.add_argument("--seed", type=int, default=4, help="Random seed for reproducibility")
    
    # Weight/output paths
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=project_root/"outputs"/"generate",
        help="Directory to save image",
    )
    parser.add_argument(
        "--weights_path",
        type=Path,
        default=project_root/"weights"/"generator_cgan.pth",
        help="Path to generator weights",
    )

    args = parser.parse_args()
    if args.base_digit < 0 or args.base_digit > 9 or args.digit2 < 0 or args.digit2 > 9:
        parser.error("--base_digit and --digit2 must be between 0 and 9")
    if args.blend < 0 or args.blend >= 1.0 :
            parser.error("--blend must be between 0.0 (inclusive) and 1.0 (exclusive)")   

    return args

def generate_digit_image(args):
    # Set random seed for reproducible style features
    torch.manual_seed(args.seed)

    # Ensure output directory exists
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the trained generator and class embedding models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    state_dict = torch.load(args.weights_path, map_location=device)
    generator.load_state_dict(state_dict)
    generator.eval()
    # Extract the class embedding weights for latent space interpolation
    embeddings = generator.label_embedding.weight.detach()

    # Lock style features with a single fixed noise vector
    fixed_noise = torch.randn(1, 100, 1, 1, device=device)

    # Create a label by averaging the embeddings of the two specified digits (default is pure base digit label)
    label = ((1 - args.blend) * embeddings[args.base_digit] + args.blend * embeddings[args.digit2]).unsqueeze(0)

    with torch.no_grad():
        image = generator(fixed_noise, label)

    image_to_show = image.cpu().squeeze().numpy()

    fig, axis = plt.subplots(figsize=(3, 3))
    axis.imshow(image_to_show, cmap="gray_r", vmin=-1, vmax=1)
    axis.axis("off")

    # Correct the title based on whether it's a pure digit or a hybrid blend
    if args.blend == 0.0:
        axis.set_title(f"Digit: {args.base_digit}")
    else:
        axis.set_title(f"Hybrid Blend: {args.base_digit} and {args.digit2}"       
                       f"\nBlend Proportion: {args.blend:.2f}")

    if args.blend == 0.0:
        image_path = output_dir / f"generated_digit_{args.base_digit}_seed_{args.seed}.png"
    else:
        image_path = output_dir / (
            f"generated_hybrid_digit_{args.base_digit}_{args.digit2}_blend_{args.blend}_seed_{args.seed}.png"
        )

    fig.savefig(image_path, bbox_inches="tight", pad_inches=0.1)
    plt.show()
    plt.close(fig)
    print(f"Generated digit image saved to: {image_path}")

if __name__ == "__main__":
    arguments = parse_args()
    generate_digit_image(arguments)