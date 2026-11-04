import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
      super(Generator, self).__init__()

      # Map class label into higher dimension to allow model to learn more complex representations of the label information
      self.label_embedding = nn.Embedding(10, 16)

      # Transposed convolutional layers to upsample latent space
      # Batch normalisation layers to stabilise training and improve convergence
      # ReLU activation for generator
      self.model = nn.Sequential(
          nn.ConvTranspose2d(116, 256, 7, 1, 0),
          nn.BatchNorm2d(256),
          nn.ReLU(inplace=True),
          nn.ConvTranspose2d(256, 128, 4, 2, 1),
          nn.BatchNorm2d(128),
          nn.ReLU(inplace=True),
          nn.ConvTranspose2d(128, 1, 4, 2, 1),
          nn.Tanh()
      )

    def forward(self, noise_vector, labels):

      # If labels are simple one-hot vector then embed them, else assume embedded
      if labels.dtype == torch.long:
        c = self.label_embedding(labels)
      else:
        c = labels

      # Reshape noise and label to match input dimensions for conv layers
      if len(noise_vector.shape) == 2:
          noise_vector = noise_vector.view(noise_vector.size(0), 100, 1, 1)
      if len(c.shape) == 2:
          c = c.view(c.size(0), 16, 1, 1)

      # Concatenate noise and label embeddings
      x = torch.cat((noise_vector, c), 1)

      # Pass through model
      return self.model(x)