import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, z_dim=128, img_channels=3, ngf=64):
        """
        Optimized Generator with:
        - Smaller initial projection
        - Progressive upsampling
        - Batch normalization for stability
        
        Args:
            z_dim: latent dimension
            img_channels: output channels (3 for RGB)
            ngf: base number of generator filters
        """
        super().__init__()
        self.z_dim = z_dim
        
        # More efficient initial projection: z -> 512 x 7 x 6
        # (7*8=56, 6*8=48 → 224x192 after 4 upsamples → crop to 218x178)
        self.init_size = (7, 6)
        self.l1 = nn.Sequential(
            nn.Linear(z_dim, ngf * 8 * self.init_size[0] * self.init_size[1]),
            nn.BatchNorm1d(ngf * 8 * self.init_size[0] * self.init_size[1]),
            nn.ReLU(True)
        )
        
        # Progressive upsampling: 7x6 → 14x12 → 28x24 → 56x48 → 112x96 → 224x192
        self.conv_blocks = nn.Sequential(
            # 512 x 7 x 6 → 256 x 14 x 12
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            # 256 x 14 x 12 → 128 x 28 x 24
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            # 128 x 28 x 24 → 64 x 56 x 48
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            # 64 x 56 x 48 → 32 x 112 x 96
            nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1),
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True),
            
            # 32 x 112 x 96 → 3 x 224 x 192
            nn.ConvTranspose2d(ngf // 2, img_channels, 4, 2, 1),
            nn.Tanh()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, z):
        """
        Args:
            z: (batch_size, z_dim)
        Returns:
            Generated images: (batch_size, 3, 218, 178)
        """
        out = self.l1(z)
        out = out.view(out.size(0), -1, *self.init_size)
        out = self.conv_blocks(out)
        
        # Crop to exact CelebA dimensions (218, 178)
        # From 224x192 → 218x178
        out = out[:, :, 3:221, 7:185]
        
        return out