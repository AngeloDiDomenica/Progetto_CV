import numpy as np
import scipy.io as sio
import torch

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def load_indian_pines(data_path, gt_path):
    """
    Carica i file .mat
    """

    data = sio.loadmat(data_path)
    gt = sio.loadmat(gt_path)

    X = data["indian_pines_corrected"]
    y = gt["indian_pines_gt"]

    return X, y


def apply_pca(X, n_components=30):
    """
    Riduce le bande spettrali tramite PCA.
    """

    h, w, bands = X.shape

    X_reshaped = X.reshape(-1, bands)

    pca = PCA(
        n_components=n_components,
        whiten=True
    )

    X_pca = pca.fit_transform(X_reshaped)

    X_pca = X_pca.reshape(
        h,
        w,
        n_components
    )

    return X_pca


def pad_with_zeros(X, margin):
    """
    Padding dell'immagine.
    """

    newX = np.zeros(
        (
            X.shape[0] + 2 * margin,
            X.shape[1] + 2 * margin,
            X.shape[2]
        )
    )

    newX[
        margin:margin + X.shape[0],
        margin:margin + X.shape[1],
        :
    ] = X

    return newX


def create_patches(X, y, window_size=25):
    """
    Estrae le patch e rimuove i pixel con label 0.
    """

    margin = window_size // 2

    X_padded = pad_with_zeros(X, margin)

    patches = []
    labels = []

    for row in range(margin, X_padded.shape[0] - margin):

        for col in range(margin, X_padded.shape[1] - margin):

            label = y[row - margin, col - margin]

            if label == 0:
                continue

            patch = X_padded[
                row - margin:row + margin + 1,
                col - margin:col + margin + 1,
                :
            ]

            patches.append(patch)
            labels.append(label - 1)

    patches = np.array(patches)
    labels = np.array(labels)

    return patches, labels


class HSIDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y,
            dtype=torch.long
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):

        sample = self.X[idx]

        # (25,25,30) -> (30,25,25)
        sample = sample.permute(2, 0, 1)

        # (30,25,25) -> (1,30,25,25)
        sample = sample.unsqueeze(0)

        return sample, self.y[idx]


def prepare_data(
    data_path,
    gt_path,
    window_size=25,
    pca_components=30,
    test_size=0.2,
    random_state=42
):
    """
    Pipeline corretta senza spatial leakage:
    1. PCA
    2. pixel split (prima delle patch)
    3. patch separati per train/test
    """

    # 1. LOAD
    X, y = load_indian_pines(data_path, gt_path)

    # 2. PCA
    X = apply_pca(X, n_components=pca_components)

    h, w, c = X.shape

    # 3. flatten pixel-wise
    X_pixels = X.reshape(-1, c)
    y_pixels = y.reshape(-1)

    # rimuovi background
    mask = y_pixels != 0
    X_pixels = X_pixels[mask]
    y_pixels = y_pixels[mask]

    # 4. SPLIT PRIMA DELLE PATCH (FONDAMENTALE)
    X_train_pix, X_test_pix, y_train_pix, y_test_pix = train_test_split(
        X_pixels,
        y_pixels,
        test_size=test_size,
        random_state=random_state,
        stratify=y_pixels
    )

    # 5. ricostruzione immagini parziali (per patch extraction)
    X_train_img = np.zeros((h, w, c))
    X_test_img = np.zeros((h, w, c))

    y_train_img = np.zeros((h, w))
    y_test_img = np.zeros((h, w))

    idx = 0

    # ricostruzione train
    for i in range(h):
        for j in range(w):
            if y[i, j] != 0:
                if idx < len(y_train_pix):
                    X_train_img[i, j] = X[i, j]
                    y_train_img[i, j] = y[i, j]
                else:
                    X_test_img[i, j] = X[i, j]
                    y_test_img[i, j] = y[i, j]
                idx += 1

    # 6. PATCH SEPARATE
    X_train, y_train = create_patches(
        X_train_img,
        y_train_img,
        window_size=window_size
    )

    X_test, y_test = create_patches(
        X_test_img,
        y_test_img,
        window_size=window_size
    )

    # 7. DATASET
    train_dataset = HSIDataset(X_train, y_train)
    test_dataset = HSIDataset(X_test, y_test)

    return train_dataset, test_dataset