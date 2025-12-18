import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, img_channels=3):
        super(Critic, self).__init__()

        self.model = nn.Sequential(
            # Input: img_channels x 218 x 178
            nn.Conv2d(img_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=7, stride=1, padding=0)  # Output a single score
        )

    def forward(self, img):
        out = self.model(img)
        out = nn.AdaptiveAvgPool2d(1)(out)  # force 1x1 spatial size
        return out.view(out.size(0), -1)  # Flatten to (batch_size, 1)

