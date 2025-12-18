import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, img_channels=3, ndf=64):
        """
        Optimized Critic with:
        - Spectral normalization (better than instance norm for WGAN-GP)
        - More efficient architecture
        - Proper downsampling to single score
        
        Args:
            img_channels: input channels (3 for RGB)
            ndf: base number of discriminator filters
        """
        super().__init__()
        
        # No normalization on first layer (standard practice)
        self.model = nn.Sequential(
            # 3 x 218 x 178 → 64 x 109 x 89
            nn.utils.spectral_norm(
                nn.Conv2d(img_channels, ndf, 4, 2, 1)
            ),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64 x 109 x 89 → 128 x 54 x 44
            nn.utils.spectral_norm(
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1)
            ),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128 x 54 x 44 → 256 x 27 x 22
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)
            ),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 256 x 27 x 22 → 512 x 13 x 11
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1)
            ),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 512 x 13 x 11 → 512 x 6 x 5
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1)
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Final layer to score (no spectral norm on output)
        self.final = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, 4, 1, 0),  # 512 x 6 x 5 → 1 x 3 x 2
            nn.AdaptiveAvgPool2d(1)  # → 1 x 1 x 1
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, img):
        """
        Args:
            img: (batch_size, 3, 218, 178)
        Returns:
            scores: (batch_size, 1) - Wasserstein distance estimate
        """
        features = self.model(img)
        out = self.final(features)
        return out.view(out.size(0), -1)