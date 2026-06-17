import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import prepare_data
from models.baselines.cnn_hybrid import HybridSN

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "hyperspectral")


def evaluate(model, loader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            out = model(x)
            preds = torch.argmax(out, dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. DATA
    train_dataset, test_dataset = prepare_data(
    os.path.join(DATA_DIR, "Indian_pines_corrected.mat"),
    os.path.join(DATA_DIR, "Indian_pines_gt.mat"),
    window_size=10,
    pca_components=15
)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False
    )

    # 2. MODEL
    model = HybridSN(num_classes=16).to(device)

    # 3. LOSS + OPTIMIZER
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 4. TRAIN LOOP
    epochs = 2

    for epoch in range(epochs):

        model.train()
        running_loss = 0

        for x, y in train_loader:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            out = model(x)
            loss = criterion(out, y)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        acc = evaluate(model, test_loader, device)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {running_loss/len(train_loader):.4f} "
            f"Test Acc: {acc:.4f}"
        )

    # 5. SALVATAGGIO MODELLO
    torch.save(model.state_dict(), "hybridsn_indian_pines.pth")
    print("Model saved.")


if __name__ == "__main__":
    main()