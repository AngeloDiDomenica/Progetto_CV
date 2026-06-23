import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

import torch

from torch.utils.data import TensorDataset

from torch.utils.data import DataLoader

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def get_dataloaders(batch_size=64):

    # =====================
    # CARICAMENTO DATI
    # =====================

    X = np.load(BASE_DIR / "X_patches.npy")

    y = np.load(BASE_DIR / "y_patches.npy")

    # =====================
    # RIORDINO DIMENSIONI
    # =====================

    X = np.transpose(X, (0, 3, 1, 2))

    CHANNELS = X.shape[1]

    PATCH_SIZE = X.shape[2]

    # =====================
    # LABELS
    # =====================

    y = y - 1

    # =====================
    # TRAIN / VALIDATION / TEST SPLIT
    # =====================

    X_temp, X_test, y_temp, y_test = train_test_split(

        X,

        y,

        test_size=0.2,

        random_state=42,

        stratify=y

    )

    X_train, X_val, y_train, y_val = train_test_split(

        X_temp,

        y_temp,

        test_size=0.125,

        random_state=42,

        stratify=y_temp

    )

    print("\nTrain:", X_train.shape)

    print("Validation:", X_val.shape)

    print("Test:", X_test.shape)

    # =====================
    # NORMALIZZAZIONE
    # =====================

    N_train = X_train.shape[0]

    N_val = X_val.shape[0]

    N_test = X_test.shape[0]


    X_train_flat = X_train.reshape(N_train, -1)

    X_val_flat = X_val.reshape(N_val, -1)

    X_test_flat = X_test.reshape(N_test, -1)


    scaler = StandardScaler()

    X_train_flat = scaler.fit_transform(X_train_flat)

    X_val_flat = scaler.transform(X_val_flat)

    X_test_flat = scaler.transform(X_test_flat)


    X_train = X_train_flat.reshape(

        N_train,

        CHANNELS,

        PATCH_SIZE,

        PATCH_SIZE

    )

    X_val = X_val_flat.reshape(

        N_val,

        CHANNELS,

        PATCH_SIZE,

        PATCH_SIZE

    )

    X_test = X_test_flat.reshape(

        N_test,

        CHANNELS,

        PATCH_SIZE,

        PATCH_SIZE

    )
    # =====================
    # PYTORCH
    # =====================

    X_train = torch.tensor(

        X_train,

        dtype=torch.float32

    )

    X_val = torch.tensor(

        X_val,

        dtype=torch.float32

    )

    X_test = torch.tensor(

        X_test,

        dtype=torch.float32

    )

    y_train = torch.tensor(

        y_train,

        dtype=torch.long

    )

    y_val = torch.tensor(

        y_val,

        dtype=torch.long

    )

    y_test = torch.tensor(

        y_test,

        dtype=torch.long

    )

    train_dataset = TensorDataset(

        X_train,

        y_train

    )

    val_dataset = TensorDataset(

        X_val,

        y_val

    )

    test_dataset = TensorDataset(

        X_test,

        y_test

    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=batch_size,

        shuffle=True

    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=batch_size,

        shuffle=False

    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=batch_size,

        shuffle=False

    )

    return train_loader, val_loader, test_loader