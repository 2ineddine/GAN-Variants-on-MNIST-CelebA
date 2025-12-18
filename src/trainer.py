import torch
import torch.nn as nn
import os
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import time

class Trainer:
    def __init__(self, G, D, dataloader, config):
        """
        Optimized WGAN-GP Trainer with:
        - Mixed precision training (AMP)
        - Efficient memory management
        - Better logging and checkpointing
        - Gradient clipping
        """
        self.device = config['device']
        self.z_dim = config['z_dim']
        
        self.G = G.to(self.device)
        self.D = D.to(self.device)
        self.dataloader = dataloader
        
        # Import locally to avoid circular dependencies
        from data.dataset import Visualizer
        self.visualizer = Visualizer(self.G, device=self.device, z_dim=self.z_dim)
        
        self.n_critic = config['n_critic']
        self.lambda_gp = config['lambda_gp']
        
        # Optimizers with better hyperparameters for WGAN-GP
        self.opt_G = torch.optim.Adam(
            G.parameters(), 
            lr=config['lr'], 
            betas=(0.0, 0.9),
            eps=1e-8
        )
        self.opt_D = torch.optim.Adam(
            D.parameters(), 
            lr=config['lr'], 
            betas=(0.0, 0.9),
            eps=1e-8
        )
        
        # Mixed precision training
        self.use_amp = config.get('use_amp', True)
        self.scaler_G = GradScaler(enabled=self.use_amp)
        self.scaler_D = GradScaler(enabled=self.use_amp)
        
        # Checkpointing
        self.checkpoint_path = config['checkpoint_path']
        self.save_every = config.get('save_every', 5)
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        
        # Gradient clipping for stability
        self.grad_clip = config.get('grad_clip', 1.0)
        
        # Import gradient penalty
        from src.wasserstein_loss import gradient_penalty
        self.gradient_penalty = gradient_penalty
    
    def train_critic_step(self, real_imgs):
        """Single critic training step"""
        batch_size = real_imgs.size(0)
        
        # Generate fake images
        z = torch.randn(batch_size, self.z_dim, device=self.device)
        
        with autocast(enabled=self.use_amp):
            fake_imgs = self.G(z)
            
            # Critic scores
            D_real = self.D(real_imgs)
            D_fake = self.D(fake_imgs.detach())
            
            # Wasserstein loss
            D_loss = -torch.mean(D_real) + torch.mean(D_fake)
            
            # Gradient penalty (computed outside autocast for numerical stability)
        
        # Compute GP in full precision
        gp = self.gradient_penalty(self.D, real_imgs, fake_imgs, device=self.device)
        
        with autocast(enabled=self.use_amp):
            D_loss_total = D_loss + self.lambda_gp * gp
        
        # Backward pass
        self.opt_D.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        self.scaler_D.scale(D_loss_total).backward()
        
        # Gradient clipping
        self.scaler_D.unscale_(self.opt_D)
        torch.nn.utils.clip_grad_norm_(self.D.parameters(), self.grad_clip)
        
        self.scaler_D.step(self.opt_D)
        self.scaler_D.update()
        
        return D_loss.item(), gp.item()
    
    def train_generator_step(self, batch_size):
        """Single generator training step"""
        z = torch.randn(batch_size, self.z_dim, device=self.device)
        
        with autocast(enabled=self.use_amp):
            fake_imgs = self.G(z)
            G_loss = -torch.mean(self.D(fake_imgs))
        
        self.opt_G.zero_grad(set_to_none=True)
        self.scaler_G.scale(G_loss).backward()
        
        # Gradient clipping
        self.scaler_G.unscale_(self.opt_G)
        torch.nn.utils.clip_grad_norm_(self.G.parameters(), self.grad_clip)
        
        self.scaler_G.step(self.opt_G)
        self.scaler_G.update()
        
        return G_loss.item()
    
    def train(self, epochs, start_epoch=1):
        """Main training loop"""
        print(f"Starting training on {self.device}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"Batch size: {self.dataloader.batch_size}")
        print(f"Steps per epoch: {len(self.dataloader)}")
        
        for epoch in range(start_epoch, start_epoch + epochs):
            self.G.train()
            self.D.train()
            
            epoch_start = time.time()
            D_losses, G_losses, gp_values = [], [], []
            
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}/{start_epoch + epochs - 1}")
            
            for i, real_imgs in enumerate(pbar):
                real_imgs = real_imgs.to(self.device, non_blocking=True)
                batch_size = real_imgs.size(0)
                
                # ----- Update Critic n_critic times -----
                for _ in range(self.n_critic):
                    D_loss, gp = self.train_critic_step(real_imgs)
                    D_losses.append(D_loss)
                    gp_values.append(gp)
                
                # ----- Update Generator -----
                G_loss = self.train_generator_step(batch_size)
                G_losses.append(G_loss)
                
                # Update progress bar
                pbar.set_postfix({
                    'D_loss': f'{D_loss:.4f}',
                    'G_loss': f'{G_loss:.4f}',
                    'GP': f'{gp:.4f}'
                })
            
            epoch_time = time.time() - epoch_start
            
            # Epoch statistics
            avg_D_loss = sum(D_losses) / len(D_losses)
            avg_G_loss = sum(G_losses) / len(G_losses)
            avg_gp = sum(gp_values) / len(gp_values)
            
            print(f"\nEpoch [{epoch}] completed in {epoch_time:.2f}s")
            print(f"  D_loss: {avg_D_loss:.4f} | G_loss: {avg_G_loss:.4f} | GP: {avg_gp:.4f}")
            
            # ----- Save checkpoint -----
            if epoch % self.save_every == 0:
                self.save_checkpoint(epoch)
            
            # ----- Generate and save images -----
            self.visualizer.generate_and_save(epoch, n_images=16, nrow=4)
            
            # Clear cache
            torch.cuda.empty_cache()
    
    def save_checkpoint(self, epoch):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'G_state_dict': self.G.state_dict(),
            'D_state_dict': self.D.state_dict(),
            'opt_G_state_dict': self.opt_G.state_dict(),
            'opt_D_state_dict': self.opt_D.state_dict(),
            'scaler_G_state_dict': self.scaler_G.state_dict(),
            'scaler_D_state_dict': self.scaler_D.state_dict(),
        }
        save_path = f"{self.checkpoint_path}_epoch{epoch}.pth"
        torch.save(checkpoint, save_path)
        print(f"✓ Checkpoint saved: {save_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.G.load_state_dict(checkpoint['G_state_dict'])
        self.D.load_state_dict(checkpoint['D_state_dict'])
        self.opt_G.load_state_dict(checkpoint['opt_G_state_dict'])
        self.opt_D.load_state_dict(checkpoint['opt_D_state_dict'])
        
        if 'scaler_G_state_dict' in checkpoint:
            self.scaler_G.load_state_dict(checkpoint['scaler_G_state_dict'])
            self.scaler_D.load_state_dict(checkpoint['scaler_D_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
        
        return start_epoch