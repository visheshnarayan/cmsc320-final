import einops
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import functional as F
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import Linear
from torch.nn import LayerNorm
from torch.nn import Dropout
from torch.nn import Embedding
from torch.nn import Parameter
from torch.nn import init
from torch.nn import Sequential
from torch.nn import ReLU
from torch.nn import Conv2d
from torch.nn import Conv1d
from torch.nn import BatchNorm2d
from torch.nn import BatchNorm1d
from torch.nn import Identity
from torch.nn import GELU
from torch.nn import ModuleDict
from torch.nn import Softmax
from torch.nn import MultiheadAttention
from torch.nn import TransformerEncoder
from torch.nn import TransformerEncoderLayer
from torch.nn import TransformerDecoder
from torch.nn import TransformerDecoderLayer
from torch.nn import Transformer
from torchvision.models import vgg11, VGG11_Weights
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt


import pickle
import polars as pl
import sklearn
import numpy as np
import pandas as pd
import random

## dataset 
class SpectogramDataset(Dataset): 
    def __init__(self,
        spectograms, 
        labels,
        transform=None,
        target_transform=None
    ):
        self.spectograms = spectograms
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.spectograms)
    
    def __getitem__(self, idx):
        spectogram = np.stack(self.spectograms[idx])
        label = self.labels[idx]
        
        try:
            spectogram = np.asarray(spectogram, dtype=np.float32)
        except Exception as e:
            print(f"❌ Spectrogram at index {idx} could not be converted: {e}")
            raise


        if self.transform:
            spectogram = self.transform(spectogram)
        if self.target_transform:
            label = self.target_transform(label)

        return np.stack(spectogram), label
    

## Transformer Head 
class TransformerHead(nn.Module):
    def __init__(
            self, in_dim, num_classes,
            num_patches=49, dim=128, depth=2,
            heads=4, mlp_dim=256, dropout=0.1
    ):
        super().__init__()
        self.patch_proj = nn.Linear(in_dim, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')  # flatten spatial dims
        x = self.patch_proj(x) + self.pos_embed[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)  # mean pooling
        return self.classifier(x)

## VGG11 + Transformer Model with Preprocessing 
class VGGWithTransformer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        vgg = vgg11(weights=VGG11_Weights.DEFAULT)

        self.preprocess = nn.Sequential(
            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False),
            nn.Conv2d(1, 3, kernel_size=1)
        )
        
        self.features = vgg.features  # outputs [B, 512, 7, 7]
        self.transformer_head = TransformerHead(in_dim=512, num_classes=num_classes, num_patches=49)

    def forward(self, x):  # x shape: (B, 1, 128, 128)
        x = self.preprocess(x)  # (B, 3, 224, 224)
        x = self.features(x)    # (B, 512, 7, 7)
        return self.transformer_head(x)

## dataset
def prepare_spectrogram_dataset(
        spectrograms, labels, 
        min_samples=2,
        test_size=0.2, val_size=0.1,
        transform_train=None, transform_eval=None, seed=42
    ):

    # Filter out rare classes
    counts = Counter(labels)
    valid_idxs = [i for i, lbl in enumerate(labels) if counts[lbl] >= min_samples]
    specs = [spectrograms[i] for i in valid_idxs]
    lbls = [labels[i] for i in valid_idxs]

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(lbls)

    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(specs, y, test_size=(test_size + val_size), stratify=y, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size/(test_size + val_size), stratify=y_temp, random_state=seed)

    # Wrap in datasets
    train_ds = SpectogramDataset(X_train, y_train, transform=transform_train)
    val_ds   = SpectogramDataset(X_val,   y_val,   transform=transform_eval)
    test_ds  = SpectogramDataset(X_test,  y_test,  transform=transform_eval)

    return train_ds, val_ds, test_ds, le

## Transforms
def mask_patch(x, axis=2, width=10):
    max_dim = x.size(axis)
    if max_dim <= width:
        return x
    start = random.randint(0, max_dim - width)
    if axis == 1:
        x[:, start:start+width, :] = 0
    elif axis == 2:
        x[:, :, start:start+width] = 0
    return x

## train loop
def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=5,
    lr=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu",
    class_weights=None
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)


    history = {"train_loss": [], "train_acc": [], "val_acc": [], "lr": []}

    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / total
        train_acc = correct / total
        val_acc = evaluate(model, val_loader, device)

        history["train_loss"].append(avg_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        # Step the scheduler based on validation accuracy
        scheduler.step(val_acc)

        print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    return model, history

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            # Remove the incorrect channel handling
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

# ------------------ Training Loop ------------------ #

# data 
balanced = pd.read_parquet("balanced_audio_data.parquet")
# balanced = balanced.sample(n=1000, random_state=42).reset_index(drop=True)

# transforms
base_transform = transforms.Compose([
    transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
    transforms.Lambda(lambda x: x.unsqueeze(0) if x.ndim == 2 else x),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
augment_transform = transforms.Compose([
    transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
    transforms.Lambda(lambda x: x.unsqueeze(0) if x.ndim == 2 else x),
    transforms.Lambda(lambda x: x + 0.01 * torch.randn_like(x) if random.random() < 0.5 else x),
    transforms.Lambda(lambda x: mask_patch(x, axis=2, width=20) if random.random() < 0.5 else x),
    transforms.Lambda(lambda x: mask_patch(x, axis=1, width=8) if random.random() < 0.5 else x)
])

# datasets (torch)
train_ds, val_ds, test_ds, label_encoder = prepare_spectrogram_dataset(
    spectrograms=balanced["Spectrogram"],
    labels=balanced["Label"],
    min_samples=2,
    transform_train=augment_transform,
    transform_eval=base_transform
)

# data loaders (torch)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4)
test_loader = DataLoader(test_ds, batch_size=4)

model = VGGWithTransformer(num_classes=len(balanced["Label"].unique()))

y_train_labels = [label for _, label in train_ds]
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_labels),
    y=y_train_labels
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to("cuda" if torch.cuda.is_available() else "cpu")


train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    lr=1e-4,
    num_epochs=1000,
    class_weights=class_weights_tensor
)

# Save the model
torch.save(model.state_dict(), "vgg_transformer_model.pth")
