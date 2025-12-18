import torch
from wasserstein_loss import wasserstein_loss, gradient_penalty

from data.dataset import Visualizer
class Trainer:
    def __init__(self, G, D, dataloader, config):
        self.device = config['device']
        self.G = G.to(self.device)
        self.D = D.to(self.device)
        self.dataloader = dataloader
        self.visualizer = Visualizer(self.G, device=self.device, z_dim=self.z_dim)  
        self.n_critic = config['n_critic']
        self.lambda_gp = config['lambda_gp']

        self.opt_G = torch.optim.Adam(G.parameters(), lr=config['lr'], betas=(0.0, 0.9))
        self.opt_D = torch.optim.Adam(D.parameters(), lr=config['lr'], betas=(0.0, 0.9))
        self.checkpoint_path = config['checkpoint_path']

    def train(self, epochs):
        for epoch in range(1, epochs + 1):
            for i, real_imgs in enumerate(self.dataloader):
                real_imgs = real_imgs.to(self.device)
                batch_size = real_imgs.size(0)

                # ----- Update Critic -----
                for _ in range(self.n_critic):
                    z = torch.randn(batch_size, self.z_dim).to(self.device)
                    fake_imgs = self.G(z)
                    D_real = self.D(real_imgs)
                    D_fake = self.D(fake_imgs.detach())

                    D_loss = -torch.mean(D_real) + torch.mean(D_fake)
                    gp = gradient_penalty(self.D, real_imgs, fake_imgs, device=self.device)
                    D_loss_total = D_loss + self.lambda_gp * gp

                    self.opt_D.zero_grad()
                    D_loss_total.backward()
                    self.opt_D.step()

                # ----- Update Generator -----
                z = torch.randn(batch_size, self.z_dim).to(self.device)
                fake_imgs = self.G(z)
                G_loss = -torch.mean(self.D(fake_imgs))

                self.opt_G.zero_grad()
                G_loss.backward()
                self.opt_G.step()
            # ----- Visualize generated images -----
            self.visualizer.generate_and_save(epoch, n_images=4)    
            if epoch % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'G_state_dict': self.G.state_dict(),
                    'D_state_dict': self.D.state_dict(),
                    'opt_G_state_dict': self.opt_G.state_dict(),
                    'opt_D_state_dict': self.opt_D.state_dict(),
                }, f"{self.checkpoint_path}_epoch{epoch}.pth")

            print(f"Epoch [{epoch}/{epochs}], D_loss: {D_loss_total.item():.4f}, G_loss: {G_loss.item():.4f}")
