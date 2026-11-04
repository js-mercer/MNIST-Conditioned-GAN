import argparse
import re
from pathlib import Path

from PIL import Image

EPOCH_IMAGE_PATTERN = re.compile(r"^epoch_(\d+)\.png$")

def parse_args():
    """Parse input and output options for the training-progress GIF."""
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Animate the sample grids saved during GAN training."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=project_root / "outputs",
        help="Directory containing epoch_NNN.png training images",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "outputs" / "training_progress.gif",
        help="Path for the generated GIF",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=250,
        help="Duration of each epoch frame in milliseconds",
    )
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be greater than zero")
    return args


def find_epoch_images(input_dir):
    """Return epoch images ordered by their numeric epoch number."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Training image directory does not exist: {input_dir}")

    epoch_images = []
    for path in input_dir.iterdir():
        match = EPOCH_IMAGE_PATTERN.fullmatch(path.name)
        if path.is_file() and match:
            epoch_images.append((int(match.group(1)), path))

    epoch_images.sort(key=lambda item: item[0])
    if not epoch_images:
        raise FileNotFoundError(
            f"No epoch_NNN.png images found in {input_dir}. "
            "Run training first, or pass the correct --input_dir."
        )
    return [path for _, path in epoch_images]


def create_training_gif(input_dir, output_path, duration):
    """Combine saved epoch sample grids into a looping animated GIF."""
    image_paths = find_epoch_images(input_dir)
    frames = []
    try:
        for path in image_paths:
            with Image.open(path) as image:
                frames.append(image.convert("RGB"))

        frame_size = frames[0].size
        if any(frame.size != frame_size for frame in frames[1:]):
            raise ValueError("All epoch images must have the same dimensions.")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Linger on last frame for 1s before looping back
        frame_durations = [duration] * (len(frames) - 1) + [1000]
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=frame_durations,
            loop=0,
            optimize=False,
        )
    finally:
        for frame in frames:
            frame.close()

    print(f"Training progress GIF saved to: {output_path}")
    print(f"Included {len(image_paths)} epoch images.")


if __name__ == "__main__":
    args = parse_args()
    create_training_gif(args.input_dir, args.output, args.duration)