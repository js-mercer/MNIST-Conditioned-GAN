import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

class Discriminator(nn.Module):
    def __init__(self):
      super(Discriminator, self).__init__()

      # Map class label into higher dimension to allow model to learn more complex representations of the label information
      self.label_embedding = nn.Embedding(10, 16)
      # Linear layer to scale embedding to image size and to disentangle label information from image features
      self.label_transform = nn.Linear(16, 28*28)

      # Leaky relu activation for discriminator to maintain gradient flow
      # Dropout layers to prevent overfitting and improve generalisation
      # Use spectral normalisation to stabilize training
      self.model = nn.Sequential(
          spectral_norm(nn.Conv2d(2, 128, 4, 2, 1)),
          nn.LeakyReLU(0.2, inplace=True),
          nn.Dropout(0.3),
          spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
          nn.LeakyReLU(0.2, inplace=True),
          nn.Dropout(0.3),
          nn.Flatten(),
          spectral_norm(nn.Linear(256*7*7, 1))
      )

    def forward(self, x, labels):
      # Embed labels, pass through linear layer, reshape to match image dimensions
      c = self.label_embedding(labels)
      c = self.label_transform(c)
      c = c.view(c.size(0), 1 , 28, 28)
      # Concatenate image and label embedding along channel dimension to condition the input
      x_conditioned = torch.cat((x, c), 1)

      # Pass through model
      return self.model(x_conditioned)