import sys

import numpy as np

import torch

import torch.nn as nn

from pathlib import Path

from hyperspectral.dataset_loader_pavia import get_dataloaders

from models.baselines.conv_baseline import SimpleConv

from models.baselines.conv_kagn_baseline import SimpleConvKAGN

from models.baselines.conv_wavkan_baseline import SimpleConvWavKAN

import csv

from codecarbon import EmissionsTracker


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

INPUT_CHANNELS = 103

NUM_CLASSES = 9

LAYER_SIZES = [32,64,128,256]

CNN_LAYER_SIZES = [72, 144, 288, 576]


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

        layer_sizes=CNN_LAYER_SIZES,

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

# =====================
# CARTELLE OUTPUT
# =====================

results_dir = Path(__file__).parent / "results"

results_dir.mkdir(exist_ok=True)

logs_dir = Path(__file__).parent / "logs"

logs_dir.mkdir(exist_ok=True)

carbon_dir = Path(__file__).parent / "carbon_logs"

carbon_dir.mkdir(exist_ok=True)

csv_file = logs_dir / f"{MODEL_NAME}_pavia_log.csv"

txt_file = logs_dir / f"{MODEL_NAME}_pavia_log.txt"

optimizer = torch.optim.Adam(

    model.parameters(),

    lr=LEARNING_RATE

)

print(optimizer)

# =====================
# BEST MODEL
# =====================

best_val_accuracy = 0.0

best_model_path = results_dir / f"{MODEL_NAME}_best.pth"

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
# TRAINING TRACKER
# =====================

training_tracker = EmissionsTracker(

    project_name=f"{MODEL_NAME}_training",

    output_dir=carbon_dir,

    output_file=f"{MODEL_NAME}_training.csv"

)

training_tracker.start()

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

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy
    
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
# FINE TRAINING TRACKER
# =====================

training_emissions = training_tracker.stop()

print(

    f"\nTraining CO2: {training_emissions:.8f} kg"

)

# =====================
# TEST TRACKER
# =====================

testing_tracker = EmissionsTracker(

    project_name=f"{MODEL_NAME}_testing",

    output_dir=carbon_dir,

    output_file=f"{MODEL_NAME}_testing.csv"

)

testing_tracker.start()

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

# =====================
# FINE TEST TRACKER
# =====================

testing_emissions = testing_tracker.stop()

print(

    f"Testing CO2: {testing_emissions:.8f} kg"

)

print("\nAccuracy finale:")

print(f"{accuracy*100:.2f}%")

# =====================
# AGGIUNTA CO2 AL CSV
# =====================

with open(csv_file, "a", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([])

    writer.writerow(
        [
            f"Training CO2: {training_emissions:.8f} kg"
        ]
    )

    writer.writerow(
        [
            f"Testing CO2: {testing_emissions:.8f} kg"
        ]
    )

    writer.writerow(
        [
            f"Best Validation Accuracy: {best_val_accuracy*100:.2f} %"
        ]
    )

    writer.writerow([])

with open(results_dir / f"{MODEL_NAME}_results.txt", "w") as f:

    f.write("Dataset: Pavia\n")

    f.write(f"Modello: {MODEL_NAME}\n")

    f.write(f"Accuracy: {accuracy*100:.2f}%\n")

    f.write(f"Epoche: {EPOCHS}\n")

    f.write(f"Batch size: {BATCH_SIZE}\n")

    f.write(f"Learning rate: {LEARNING_RATE}\n")

    f.write(f"Optimizer: {optimizer.__class__.__name__}\n")

    f.write("Patch size: 9x9\n")

    f.write("Label smoothing: 0.1\n")

    f.write("Seed: 42\n")

    f.write(f"Device: {device}\n")

    f.write(f"Parametri: {total_params}\n")

    f.write("\nCarbon Footprint\n")

    f.write(f"Training CO2: {training_emissions:.8f} kg\n")

    f.write(f"Testing CO2: {testing_emissions:.8f} kg\n")

    f.write("\nNote:\n")

    f.write("Split: 70% Train - 10% Validation - 20% Test.\n")

    f.write("Suddivisione stratificata sui pixel.\n")

    f.write("Possibile presenza di spatial leakage.\n")

print(f"\nRisultati salvati in results/{MODEL_NAME}_results.txt")

print(model.__class__.__name__)