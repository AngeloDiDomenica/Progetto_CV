
import numpy as np
import torch
from scipy.spatial import cKDTree
from pathlib import Path
from scipy.io import loadmat
from scipy.ndimage import binary_dilation
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

BASE_DIR = Path(__file__).resolve().parent

PATCH_SIZE = 9
BLOCK_SIZE = 10
GUARD = PATCH_SIZE // 2
BUFFER = PATCH_SIZE - 1 
SEED = 42


def _load():
    print("load Pavia")
    data = loadmat(BASE_DIR / "PaviaU.mat")
    gt   = loadmat(BASE_DIR / "PaviaU_gt.mat")
    
    X = data["paviaU"].astype(np.float32)
    y = gt["paviaU_gt"].astype(np.int64)
    return X, y


def _assign_blocks(y):
    rng = np.random.default_rng(SEED)
    h, w = y.shape
    assign = np.zeros((h, w), dtype=np.int64)
    for bi in range((h + BLOCK_SIZE - 1) // BLOCK_SIZE):
        for bj in range((w + BLOCK_SIZE - 1) // BLOCK_SIZE):
            r = rng.random()
            tag = 3 if r < 0.20 else (2 if r < 0.30 else 1)
            r0 = bi * BLOCK_SIZE
            c0 = bj * BLOCK_SIZE
            mask = y[r0:r0+BLOCK_SIZE, c0:c0+BLOCK_SIZE] != 0
            assign[r0:r0+BLOCK_SIZE, c0:c0+BLOCK_SIZE][mask] = tag
    return assign


def _apply_guard(assign):
    struct = np.ones((2*BUFFER+1, 2*BUFFER+1), dtype=bool)
    assign = assign.copy()

    # train troppo vicino a val o test -> scartato
    held_by_valtest = binary_dilation((assign == 2) | (assign == 3), structure=struct)
    assign[(assign == 1) & held_by_valtest] = 0

    # test troppo vicino a val -> scartato
    held_by_val = binary_dilation(assign == 2, structure=struct)
    assign[(assign == 3) & held_by_val] = 0

    return assign


def _check_no_leakage(assign):
    tr = np.argwhere(assign == 1)
    va = np.argwhere(assign == 2)
    te = np.argwhere(assign == 3)
    if len(tr) == 0 or len(va) == 0 or len(te) == 0:
        return
    for name, a, b in [("train-val", tr, va), ("train-test", tr, te), ("val-test", va, te)]:
        tree = cKDTree(a)
        dist, _ = tree.query(b, k=1)
        min_d = dist.min()
        assert min_d > BUFFER, f"Leakage possibile tra {name}: distanza minima {min_d} <= BUFFER {BUFFER}"
    print("Nessuna sovrapposizione di patch rilevata tra train/val/test.")

def _extract(X, y, assign, tag):
    margin = GUARD
    X_pad = np.pad(X, ((margin, margin), (margin, margin), (0, 0)), mode="constant")
    patches, labels = [], []
    rows, cols = np.where(assign == tag)
    for i, j in zip(rows, cols):
        patches.append(X_pad[i:i+PATCH_SIZE, j:j+PATCH_SIZE, :])
        labels.append(y[i, j] - 1)
    X_out = np.array(patches, dtype=np.float32)
    X_out = np.transpose(X_out, (0, 3, 1, 2))
    return X_out, np.array(labels, dtype=np.int64)


def get_dataloaders(batch_size=64):
    X, y = _load()

    assign = _assign_blocks(y)
    assign = _apply_guard(assign)
    _check_no_leakage(assign)

    n_tr = int((assign == 1).sum())
    n_va = int((assign == 2).sum())
    n_te = int((assign == 3).sum())
    n_sc = int((y != 0).sum()) - n_tr - n_va - n_te
    print(f"\nPixel  train: {n_tr}  val: {n_va}  test: {n_te}  scartati(guard): {n_sc}")

    X_train, y_train = _extract(X, y, assign, 1)
    X_val,   y_val   = _extract(X, y, assign, 2)
    X_test,  y_test  = _extract(X, y, assign, 3)

    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    n_tr_n = X_train.shape[0]
    bands  = X_train.shape[1]
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(n_tr_n, -1))

    def _scale(arr):
        n = arr.shape[0]
        flat = scaler.transform(arr.reshape(n, -1))
        return flat.reshape(n, bands, PATCH_SIZE, PATCH_SIZE).astype(np.float32)

    X_train = _scale(X_train)
    X_val   = _scale(X_val)
    X_test  = _scale(X_test)

    def _loader(Xa, ya, shuffle):
        ds = TensorDataset(
            torch.tensor(Xa, dtype=torch.float32),
            torch.tensor(ya, dtype=torch.long),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return _loader(X_train, y_train, True), _loader(X_val, y_val, False), _loader(X_test, y_test, False)