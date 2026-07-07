import numpy as np
import torch

from pathlib import Path
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

BASE_DIR = Path(__file__).resolve().parent

# Split stratificato random per classe, stesso protocollo di HyperKAN:
# 20% train, dei rimanenti 12.5% val e 87.5% test (-> ~10% val, ~70% test sul totale).
# Nessuno split spaziale: e' la pratica standard della letteratura HSI.
# Firma identica a prima: get_dataloaders(batch_size) -> (train, val, test loader)
# Shape patch: (N, 200, 9, 9). addestramento.py non va toccato.

PATCH_SIZE = 9
TRAIN_RATIO = 0.20   # percentuale di pixel per classe usata come training
VAL_RATIO   = 0.10   # percentuale sul totale per validation (dal pool test)
SEED = 42


def _load():
    data = loadmat(BASE_DIR / "Indian_pines_corrected.mat")
    gt   = loadmat(BASE_DIR / "Indian_pines_gt.mat")
    X = data["indian_pines_corrected"].astype(np.float32)
    y = gt["indian_pines_gt"].astype(np.int64)
    return X, y


def _extract_patches(X, coords, y_flat):
    margin = PATCH_SIZE // 2
    X_pad = np.pad(X, ((margin, margin), (margin, margin), (0, 0)), mode="constant")
    patches = []
    for i, j in coords:
        patches.append(X_pad[i:i + PATCH_SIZE, j:j + PATCH_SIZE, :])
    X_out = np.array(patches, dtype=np.float32)
    X_out = np.transpose(X_out, (0, 3, 1, 2))   # (N, bands, win, win)
    return X_out, np.array(y_flat, dtype=np.int64)


def get_dataloaders(batch_size=64):
    X, y = _load()

    # Raccoglie le coordinate di tutti i pixel etichettati
    rows, cols = np.nonzero(y)
    coords_all = list(zip(rows.tolist(), cols.tolist()))
    labels_all = [y[r, c] - 1 for r, c in coords_all]   # 0-based

    # Split stratificato train / resto (80%)
    coords_train, coords_rest, y_train_flat, y_rest_flat = train_test_split(
        coords_all, labels_all,
        train_size=TRAIN_RATIO,
        stratify=labels_all,
        random_state=SEED
    )

    # Dal pool restante: split val / test stratificato
    # VAL_RATIO e' rispetto al totale, quindi rispetto al pool restante e'
    # VAL_RATIO / (1 - TRAIN_RATIO)
    val_size_adjusted = VAL_RATIO / (1.0 - TRAIN_RATIO)
    coords_val, coords_test, y_val_flat, y_test_flat = train_test_split(
        coords_rest, y_rest_flat,
        train_size=val_size_adjusted,
        stratify=y_rest_flat,
        random_state=SEED
    )

    print(f"\nPixel  train: {len(coords_train)}  val: {len(coords_val)}  test: {len(coords_test)}")

    X_train, y_train = _extract_patches(X, coords_train, y_train_flat)
    X_val,   y_val   = _extract_patches(X, coords_val,   y_val_flat)
    X_test,  y_test  = _extract_patches(X, coords_test,  y_test_flat)

    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    # Normalizzazione per banda: fit solo sul train
    n_tr = X_train.shape[0]
    bands = X_train.shape[1]
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(n_tr, -1))

    def _scale(arr):
        n = arr.shape[0]
        flat = scaler.transform(arr.reshape(n, -1))
        return flat.reshape(n, bands, PATCH_SIZE, PATCH_SIZE).astype(np.float32)

    X_train = _scale(X_train)
    X_val   = _scale(X_val)
    X_test  = _scale(X_test)

    def _loader(Xa, ya, shuffle, drop_last):
        ds = TensorDataset(
            torch.tensor(Xa, dtype=torch.float32),
            torch.tensor(ya, dtype=torch.long),
        )

        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last
        )

    train_loader = _loader(
        X_train,
        y_train,
        shuffle=True,
        drop_last=True
    )

    val_loader = _loader(
        X_val,
        y_val,
        shuffle=False,
        drop_last=False
    )

    test_loader = _loader(
        X_test,
        y_test,
        shuffle=False,
        drop_last=False
    )

    return train_loader, val_loader, test_loader
