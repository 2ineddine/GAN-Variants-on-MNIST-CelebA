import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from model import Generator, Critic
from src import Trainer

# ----- Load config -----
with open("config.yaml") as f:
    config = yaml.safe_load(f)

device = config['device']

# ----- Dataset -----
transform = transforms.Compose([
    transforms.Resize((config['image_size']['height'], config['image_size']['width'])),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
dataset = datasets.ImageFolder(root=config['dataset_path'], transform=transform)
dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

# ----- Models -----
G = Generator(z_dim=config['z_dim'])
D = Critic()

# ----- Trainer -----
trainer = Trainer(G, D, dataloader, config)

# ----- Train -----
trainer.train(epochs=config['epochs'])
