import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import TensorDataset, DataLoader

import torch.nn as nn

from pathlib import Path
# =====================
# CARICAMENTO DATI
# =====================

X = np.load("X_patches.npy")

y = np.load("y_patches.npy")

print("Shape originale X:", X.shape)

print("Shape originale y:", y.shape)


# =====================
# RIORDINO DIMENSIONI
# =====================

X = np.transpose(X, (0, 3, 1, 2))

print("\nNuova shape X:", X.shape)



torch.manual_seed(42)

np.random.seed(42)

if torch.cuda.is_available():

    torch.cuda.manual_seed(42)

    torch.cuda.manual_seed_all(42)

# =====================
# LABELS
# =====================

y = y - 1

# =====================
# TRAIN TEST SPLIT
# =====================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain:", X_train.shape)

print("Test:", X_test.shape)


# =====================
# NORMALIZZAZIONE
# =====================

N_train = X_train.shape[0]

N_test = X_test.shape[0]

X_train_flat = X_train.reshape(N_train, -1)

X_test_flat = X_test.reshape(N_test, -1)

scaler = StandardScaler()

X_train_flat = scaler.fit_transform(X_train_flat)

X_test_flat = scaler.transform(X_test_flat)

X_train = X_train_flat.reshape(
    N_train,
    200,
    9,
    9
)

X_test = X_test_flat.reshape(
    N_test,
    200,
    9,
    9
)

# =====================
# PYTORCH
# =====================

X_train = torch.tensor(X_train, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)

y_test = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(
    X_train,
    y_train
)

test_dataset = TensorDataset(
    X_test,
    y_test
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

print("\nDataLoader creati.")





# =====================
# MODELLO CNN
# =====================

class CNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                in_channels=200,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Flatten()

        )

        self.classifier = nn.Sequential(

            nn.Linear(128 * 4 * 4, 128),

            nn.ReLU(),

            nn.Linear(128,16)

        )

    def forward(self,x):

        x = self.features(x)

        x = self.classifier(x)

        return x
    

# =====================
# DEVICE
# =====================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nDevice:", device)

print("Torch:", torch.__version__)

model = CNN().to(device)

total_params = sum(
    p.numel()
    for p in model.parameters()
)

print("\nNumero parametri:", total_params)

print(model)

criterion = nn.CrossEntropyLoss(
    label_smoothing=0.1
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 20

best_accuracy = 0

# =====================
# TRAINING
# =====================

for epoch in range(epochs):

    model.train()

    running_loss = 0

    for batch_x, batch_y in train_loader:

        batch_x = batch_x.to(device)

        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        outputs = model(batch_x)

        loss = criterion(
            outputs,
            batch_y
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    print(
        f"Epoch {epoch+1}/{epochs}"
        f" Loss: {avg_loss:.4f}"
    )
    


# =====================
# TEST
# =====================

model.eval()

correct = 0

total = 0

with torch.no_grad():

    for batch_x, batch_y in test_loader:

        batch_x = batch_x.to(device)

        batch_y = batch_y.to(device)

        outputs = model(batch_x)

        predictions = outputs.argmax(1)

        total += batch_y.size(0)

        correct += (
            predictions == batch_y
        ).sum().item()

accuracy = correct / total

print("\nAccuracy finale:")

print(f"{accuracy*100:.2f}%")

results_dir = Path(__file__).parent.parent / "results"

results_dir.mkdir(exist_ok=True)

with open(results_dir / "cnn_results.txt", "w") as f:

    f.write("Dataset: Indian Pines\n")

    f.write("Modello: CNN baseline\n")

    f.write(f"Accuracy: {accuracy*100:.2f}%\n")

    f.write(f"Epoche: {epochs}\n")

    f.write("Batch size: 64\n")

    f.write("Learning rate: 0.001\n")

    f.write("Patch size: 9x9\n")

    f.write("Label smoothing: 0.1\n")

    f.write("Seed: 42\n")

    f.write(f"Device: {device}\n")

    f.write(f"Parametri: {total_params}\n")

    f.write("\nNote:\n")

    f.write("Train/test split casuale sui pixel.\n")

    f.write("Possibile presenza di spatial leakage.\n")

print("\nRisultati salvati in results/cnn_results.txt")