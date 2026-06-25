from scipy.io import loadmat
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# =========================
# CARICAMENTO DATI
# =========================

data = loadmat("Indian_pines_corrected.mat")
labels = loadmat("Indian_pines_gt.mat")

X = data["indian_pines_corrected"]
y = labels["indian_pines_gt"]

# Rimuove lo sfondo (classe 0)
mask = y > 0

X_samples = X[mask]
y_samples = y[mask]

# Le classi devono partire da 0
y_samples = y_samples - 1

print("Shape X:", X_samples.shape)
print("Shape y:", y_samples.shape)

# =========================
# NORMALIZZAZIONE
# =========================

scaler = StandardScaler()
X_samples = scaler.fit_transform(X_samples)

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X_samples,
    y_samples,
    test_size=0.2,
    random_state=42,
    stratify=y_samples
)

# =========================
# PYTORCH
# =========================

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train, y_train)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

# =========================
# MODELLO
# =========================

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(200, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 16)
        )

    def forward(self, x):
        return self.net(x)

# =========================
# DEVICE
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

model = MLP().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# =========================
# TRAINING
# =========================

epochs = 20

for epoch in range(epochs):

    model.train()

    running_loss = 0

    for batch_x, batch_y in train_loader:

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        outputs = model(batch_x)

        loss = criterion(outputs, batch_y)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{epochs} "
        f"Loss: {running_loss:.4f}"
    )

# =========================
# TEST
# =========================

model.eval()

with torch.no_grad():

    outputs = model(X_test.to(device))

    predictions = outputs.argmax(dim=1)

accuracy = accuracy_score(
    y_test.cpu(),
    predictions.cpu()
)

print("\nAccuracy finale:")
print(f"{accuracy*100:.2f}%")