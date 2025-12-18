import torch
import torch.nn as nn

def wasserstein_loss_(y_true, y_pred):
    """Wasserstein loss for WGAN"""
    return torch.mean(y_true * y_pred)

def gradient_penalty(critic, real, fake, device="cpu"):
    """
    Optimized gradient penalty for WGAN-GP
    Key fix: detach fake images to prevent memory leak
    """
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    
    # CRITICAL FIX: Detach fake to break computational graph
    interpolated = (alpha * real + (1 - alpha) * fake.detach()).requires_grad_(True)
    
    pred = critic(interpolated)
    
    gradients = torch.autograd.grad(
        outputs=pred,
        inputs=interpolated,
        grad_outputs=torch.ones_like(pred, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,  # Optimization: only compute gradients w.r.t. inputs
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    
    # Compute penalty
    penalty = ((gradient_norm - 1) ** 2).mean()
    
    return penalty