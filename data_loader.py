from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_loaders(batch_size, data_dir="./data"):
    """Create the shuffled MNIST training loader used by train.py."""

    # Transform data to tensor and normalise to [-1, 1] range
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]
    )

    # Load MNIST dataset
    trainset = datasets.MNIST(
        root=data_dir,
        train=True,
        transform=transform,
        download=True,
    )

    # Return a shuffled DataLoader
    return DataLoader(trainset, batch_size=batch_size, shuffle=True)