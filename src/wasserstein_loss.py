import torch
import torch.nn as nn

def wasserstein_loss(y_true, y_pred):
    return torch.mean(y_true * y_pred)



def gradient_penalty(critic, real, fake, device="cpu"):
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    pred = critic(interpolated)
    gradients = torch.autograd.grad(
        outputs=pred,
        inputs=interpolated,
        grad_outputs=torch.ones_like(pred),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.view(batch_size, -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()
