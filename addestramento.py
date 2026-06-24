import sys

import numpy as np

import torch

import torch.nn as nn

from pathlib import Path

from hyperspectral.dataset_loader import get_dataloaders

from models.baselines.conv_baseline import SimpleConv

from models.baselines.conv_kagn_baseline import SimpleConvKAGN

from models.baselines.conv_wavkan_baseline import SimpleConvWavKAN

import csv


# =====================
# PARAMETRI
# =====================

MODEL_NAME = "cnn"

VALID_MODELS = [

    "cnn",

    "kagn",

    "wavkan"

]

BATCH_SIZE = 64

LEARNING_RATE = 0.003

EPOCHS = 50

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

train_loader, val_loader, test_loader = get_dataloaders(

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

if MODEL_NAME == "cnn":

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

print(optimizer)

# =====================
# CARTELLE OUTPUT
# =====================

results_dir = Path(__file__).parent / "results"

results_dir.mkdir(exist_ok=True)

logs_dir = Path(__file__).parent / "logs"

logs_dir.mkdir(exist_ok=True)

csv_file = logs_dir / f"{MODEL_NAME}_log.csv"

txt_file = logs_dir / f"{MODEL_NAME}_log.txt"

# =====================
# INIZIALIZZAZIONE CSV
# =====================

with open(csv_file, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow(

        [

            "epoch",

            "train_loss",

            "train_accuracy_%",

            "validation_accuracy_%"

        ]

    )

with open(txt_file, "w") as file:

    file.write(f"Modello: {MODEL_NAME}\n\n")

# =====================
# TRAINING
# =====================

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0

    train_correct = 0

    train_total = 0

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

        predictions = outputs.argmax(1)

        train_total += batch_y.size(0)

        train_correct += (

            predictions == batch_y

        ).sum().item()

    avg_loss = running_loss / len(train_loader)

    train_accuracy = train_correct / train_total

    #print(
    #    f"Epoch {epoch+1}/{EPOCHS}"
    #    f" Loss: {avg_loss:.4f}"
    #    f" Train Accuracy: {train_accuracy*100:.2f}%"
    #)
    
    # =====================
    # VALIDATION
    # =====================

    model.eval()

    val_correct = 0

    val_total = 0

    with torch.no_grad():

        for batch_x, batch_y in val_loader:

            batch_x = batch_x.to(device)

            batch_y = batch_y.to(device)

            outputs = model(batch_x)

            predictions = outputs.argmax(1)

            val_total += batch_y.size(0)

            val_correct += (

                predictions == batch_y

            ).sum().item()

    val_accuracy = val_correct / val_total
    #print(
    #    f"Validation Accuracy: {val_accuracy*100:.2f}%"
    #)
    print(

        f"Epoch {epoch+1}/{EPOCHS}"

        f" | Loss: {avg_loss:.4f}"

        f" | Train: {train_accuracy*100:.2f}%"

        f" | Val: {val_accuracy*100:.2f}%"

    )

    with open(txt_file, "a") as file:

        file.write(

            f"Epoch {epoch+1}/{EPOCHS}"

            f" | Loss: {avg_loss:.4f}"

            f" | Train: {train_accuracy*100:.2f}%"

            f" | Val: {val_accuracy*100:.2f}%\n"

        )

    with open(csv_file, "a", newline="") as file:

        writer = csv.writer(file)

        writer.writerow(

            [

                epoch + 1,

                round(avg_loss,4),

                round(train_accuracy*100,2),

                round(val_accuracy*100,2)

            ]

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

    f.write("Split: 70% Train - 10% Validation - 20% Test.\n")

    f.write("Suddivisione stratificata sui pixel.\n")

    f.write("Possibile presenza di spatial leakage.\n")

print(f"\nRisultati salvati in results/{MODEL_NAME}_results.txt")

print(model.__class__.__name__)