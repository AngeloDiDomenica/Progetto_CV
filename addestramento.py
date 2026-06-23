import sys

import numpy as np

import torch

import torch.nn as nn

from pathlib import Path

from hyperspectral.dataset_loader import get_dataloaders

from models.baselines.conv_baseline import SimpleConv

from models.baselines.conv_kagn_baseline import SimpleConvKAGN

from models.baselines.conv_wavkan_baseline import SimpleConvWavKAN


# =====================
# PARAMETRI
# =====================

MODEL_NAME = "baseline"

VALID_MODELS = [

    "baseline",

    "kagn",

    "wavkan"

]

BATCH_SIZE = 64

LEARNING_RATE = 0.003

EPOCHS = 1

INPUT_CHANNELS = 200

NUM_CLASSES = 16

LAYER_SIZES = [32,64,128,256]


# =====================
# ARGOMENTO TERMINALE
# =====================

if len(sys.argv) > 1:

    MODEL_NAME = sys.argv[1].lower()

if MODEL_NAME not in VALID_MODELS:

    raise ValueError(

        f"Modello non valido. Usa: {VALID_MODELS}"

    )


# =====================
# SEED
# =====================

torch.manual_seed(42)

np.random.seed(42)

if torch.cuda.is_available():

    torch.cuda.manual_seed(42)

    torch.cuda.manual_seed_all(42)


# =====================
# DATALOADER
# =====================

train_loader, test_loader = get_dataloaders(

    batch_size=BATCH_SIZE

)

print("\nDataLoader creati.")

# =====================
# DEVICE
# =====================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nDevice:", device)

print("Torch:", torch.__version__)

if MODEL_NAME == "baseline":

    model = SimpleConv(

        layer_sizes=LAYER_SIZES,

        input_channels=INPUT_CHANNELS,

        num_classes=NUM_CLASSES

    )

elif MODEL_NAME == "kagn":

    model = SimpleConvKAGN(

        layer_sizes=LAYER_SIZES,

        input_channels=INPUT_CHANNELS,

        num_classes=NUM_CLASSES,

        degree=3

    )

elif MODEL_NAME == "wavkan":

    model = SimpleConvWavKAN(

        layer_sizes=LAYER_SIZES,

        input_channels=INPUT_CHANNELS,

        num_classes=NUM_CLASSES,

        wavelet_type="mexican_hat"

    )

else:

    raise ValueError("Modello non valido")

model = model.to(device)

print(f"\nModello selezionato: {MODEL_NAME}")

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

    lr=LEARNING_RATE

)

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

results_dir = Path(__file__).parent / "results"

results_dir.mkdir(exist_ok=True)

with open(results_dir / f"{MODEL_NAME}_results.txt", "w") as f:

    f.write("Dataset: Indian Pines\n")

    f.write(f"Modello: {MODEL_NAME}\n")

    f.write(f"Accuracy: {accuracy*100:.2f}%\n")

    f.write(f"Epoche: {EPOCHS}\n")

    f.write(f"Batch size: {BATCH_SIZE}\n")

    f.write(f"Learning rate: {LEARNING_RATE}\n")

    f.write("Patch size: 9x9\n")

    f.write("Label smoothing: 0.1\n")

    f.write("Seed: 42\n")

    f.write(f"Device: {device}\n")

    f.write(f"Parametri: {total_params}\n")

    f.write("\nNote:\n")

    f.write("Train/test split casuale sui pixel.\n")

    f.write("Possibile presenza di spatial leakage.\n")

print(f"\nRisultati salvati in results/{MODEL_NAME}_results.txt")

print(model.__class__.__name__)