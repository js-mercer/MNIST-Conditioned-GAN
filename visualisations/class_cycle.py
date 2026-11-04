import argparse
import torch
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path

from models.generator import Generator

def parse_args():
    """Defines and parses command line arguments for generating digit morphing animations."""
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate a digit morphing animation using a trained Conditioned MNIST GAN")
    
    # Parameters
    parser.add_argument("--seed", type=int, default=4, help="Random seed for reproducibility")
    parser.add_argument("--frames", type=int, default=20, help="Number of frames for each digit transition (minimum 2)")
    
    # Weight/output paths
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(project_root/"outputs"/"class_cycle"),
        help="Directory to save animation",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(project_root/"weights"/"generator_cgan.pth"),
        help="Path to generator weights",
    )

    args = parser.parse_args()
    if args.frames < 2:
        parser.error("--frames must be at least 2")
    return args

def generate_morphing_animation(args):
    # Set random seed for reproducible style features
    torch.manual_seed(args.seed)

    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the trained generator and class embedding models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    state_dict = torch.load(args.weights, map_location=device)
    generator.load_state_dict(state_dict)
    generator.eval()
    # Extract the class embedding weights for latent space interpolation
    embeddings = generator.label_embedding.weight.detach()

    # Lock style features with a single fixed noise vector
    fixed_noise = torch.randn(1, 100, 1, 1, device=device)

    # How many frames to generate for each digit transition
    frames_per_transition = args.frames
    # How many frames to freeze on the digit before transitioning to the next digit
    linger_count = 10 
    frames = []

    # Map consecutive transitions including the loop back:
    digit_pairs = np.arange(0,10)
    digit_pairs = np.append(digit_pairs, digit_pairs[0])
    digits_iterable = digit_pairs.tolist()

    print("Generating digit morphing gif...")
    
    for start_digit, end_digit in zip(digits_iterable[:-1], digits_iterable[1:]):

        # Generate a linear interpolation between the two digit embeddings
        t_steps = np.linspace(0.0, 1.0, num=frames_per_transition, endpoint=False)

        for t in t_steps:
            # Compute latent coordinates vector between the digit pair
            blended_vector = ((1.0 - t) * embeddings[start_digit] + t * embeddings[end_digit]).unsqueeze(0)

            # Generate hybrid digit
            with torch.no_grad():
                img_tensor = generator(fixed_noise, blended_vector)

            # Convert the generated tensor to a numpy array and normalize to [0, 255]
            img_np = img_tensor.cpu().squeeze().numpy()
            img_normalized = ((img_np + 1.0) / 2.0 * 255.0).astype(np.uint8)

            # Convert to an image object
            pil_img = Image.fromarray(img_normalized).convert('L')
            pil_img = ImageOps.invert(pil_img)
            # Upscale GIF (28x28 -> 280x280)
            pil_img = pil_img.resize((280, 280), Image.Resampling.NEAREST)

            # Linger on pure digits
            if t == 0.0:
                # If pure digit, append multiple times to create a pause
                for _ in range(linger_count):
                    frames.append(pil_img)
            else:
                # Else, append the morphing frame once
                frames.append(pil_img)

    # Save and compile into a GIF
    gif_filename = output_dir / "class_cycle.gif"
    if frames:
        frames[0].save(
            gif_filename,
            save_all=True,
            append_images=frames[1:],
            duration=50,  # Milliseconds per frame 
            loop=0        # loop infinitely
        )
        print(f"Class cycle GIF saved as: {gif_filename}")
    else:
        print("No frames were generated.")

if __name__ == "__main__":
    arguments = parse_args()
    generate_morphing_animation(arguments)