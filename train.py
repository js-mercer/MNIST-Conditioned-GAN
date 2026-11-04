import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision

from models.generator import Generator
from models.discriminator import Discriminator
from data_loader import get_mnist_loaders 


def parse_args():
    """Defines and parses command line arguments for hyperparameters."""
    parser = argparse.ArgumentParser(description="Train a Conditioned MNIST GAN")
    
    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr_d", type=float, default=0.0001, help="Discriminator learning rate")
    parser.add_argument("--lr_g", type=float, default=0.0004, help="Generator learning rate")
    
    # Environment/Output paths
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save generated epoch images")
    parser.add_argument("--weights_dir", type=str, default="./weights", help="Directory to save model weights")
    
    return parser.parse_args()

def train_gan(args):
    # Setup hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure output directories exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.weights_dir, exist_ok=True)

    # Initialise networks and shift to device
    generator = Generator().to(device=device)
    discriminator = Discriminator().to(device=device)

    loss_function = nn.MSELoss()

    # Load data 
    train_loader = get_mnist_loaders(args.batch_size)

    # Setup optimisers
    optimiser_discriminator = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(0.5, 0.999))
    optimiser_generator = torch.optim.Adam(generator.parameters(), lr=args.lr_g, betas=(0.5, 0.999))

    loss_discriminator_val = 0.0

    for epoch in range(args.epochs):
        generator.train()
        discriminator.train()
        
        for n, (real_samples, mnist_labels) in enumerate(train_loader):
            current_batch_size = real_samples.size(0)

            # Data for training discriminator
            real_samples = real_samples.to(device=device)
            mnist_labels = mnist_labels.to(device=device)
            real_samples_labels = torch.full((current_batch_size, 1), 0.9, device=device)

            latent_space_samples = torch.randn((current_batch_size, 100, 1, 1), device=device)
            generated_samples = generator(latent_space_samples, mnist_labels)
            generated_samples_labels = torch.zeros((current_batch_size, 1), device=device)

            all_samples = torch.cat((real_samples, generated_samples.detach()))
            all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))
            all_conditional_labels = torch.cat((mnist_labels, mnist_labels))

            # Training the discriminator
            discriminator.zero_grad()

            # Introduce noise to stabilise training and decay it over epochs
            # 0.1 scaled noise for 25% of epochs, smoothly decay to zero noise by 50% of epochs
            noisey_epochs = int(args.epochs * 0.25)
            decay_epochs = int(args.epochs * 0.5)

            if epoch <= noisey_epochs:
                noise = torch.randn_like(all_samples) * 0.1
            elif epoch <= decay_epochs:
                noise = torch.randn_like(all_samples) * 0.1 * (decay_epochs - epoch) / (decay_epochs - noisey_epochs)
            else:
                noise = torch.randn_like(all_samples) * 0.00

            output_discriminator = discriminator(all_samples + noise, all_conditional_labels).view(-1, 1)
            loss_discriminator = loss_function(output_discriminator, all_samples_labels)
            loss_discriminator.backward()
            optimiser_discriminator.step()
            loss_discriminator_val = loss_discriminator.item()

            # Data for training the generator
            latent_space_examples = torch.randn((current_batch_size, 100, 1, 1), device=device)
            random_gen_labels = torch.randint(0, 10, (current_batch_size,), device=device)

            # Training the generator
            generator.zero_grad()
            generated_samples = generator(latent_space_examples, random_gen_labels)
            generator_targets = torch.ones((current_batch_size, 1), device=device)
            output_discriminator_generated = discriminator(generated_samples, random_gen_labels).view(-1, 1)
            loss_generator = loss_function(output_discriminator_generated, generator_targets)
            loss_generator.backward()
            optimiser_generator.step()

            # Print progress
            if (n + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{args.epochs}], Batch [{n+1}/{len(train_loader)}], '
                      f'Loss_D: {loss_discriminator_val:.4f}, Loss_G: {loss_generator.item():.4f}')

        # Evaluate progress and save generated samples after each epoch
        generator.eval()
        with torch.no_grad():
            eval_labels = torch.tensor([i for i in range(10) for _ in range(10)], device=device)
            eval_noise = torch.randn(100, 100, 1, 1, device=device)

            fake_samples = generator(eval_noise, eval_labels).cpu()
            fake_grid = torchvision.utils.make_grid(fake_samples, nrow=10, padding=2, normalize=True)
            
            # Save the image file to the output directory
            img_save_path = os.path.join(args.output_dir, f"epoch_{epoch+1:03d}.png")
            plt.figure(figsize=(6, 6))
            plt.imshow(np.transpose(fake_grid, (1, 2, 0)))
            plt.axis('off')
            plt.savefig(img_save_path, bbox_inches='tight')
            plt.close() 

    # Save network weights
    gen_save_path = os.path.join(args.weights_dir, 'generator_cgan.pth')
    disc_save_path = os.path.join(args.weights_dir, 'discriminator_cgan.pth')

    torch.save(generator.state_dict(), gen_save_path)
    torch.save(discriminator.state_dict(), disc_save_path)

    print(f"Training completed successfully.")
    print(f"Generator weights saved to: {gen_save_path}")
    print(f"Discriminator weights saved to: {disc_save_path}")

if __name__ == "__main__":
    arguments = parse_args()
    train_gan(arguments)