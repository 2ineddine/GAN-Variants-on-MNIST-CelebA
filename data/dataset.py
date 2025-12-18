import torch
import torchvision.utils as vutils
import os

class Visualizer:
    def __init__(self, G, device='cuda', output_dir='generated_images', z_dim=128):
        """
        G: Generator model
        device: 'cuda' or 'cpu'
        output_dir: folder to save generated images
        z_dim: dimensionality of noise vector
        """
        self.G = G.to(device)
        self.device = device
        self.output_dir = output_dir
        self.z_dim = z_dim
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_and_save(self, epoch, n_images=4, nrow=2):
        """Generate n_images and save them to disk"""
        self.G.eval()
        with torch.no_grad():
            z = torch.randn(n_images, self.z_dim).to(self.device)
            fake_imgs = self.G(z)
            # Denormalize from [-1,1] to [0,1]
            fake_imgs = (fake_imgs + 1) / 2.0  

            # Save grid
            save_path = os.path.join(self.output_dir, f"epoch_{epoch}.png")
            vutils.save_image(fake_imgs, save_path, nrow=nrow, padding=2)
        self.G.train()
