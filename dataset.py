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
    Pipeline completa.
    """

    X, y = load_indian_pines(
        data_path,
        gt_path
    )

    X = apply_pca(
        X,
        n_components=pca_components
    )

    X, y = create_patches(
        X,
        y,
        window_size=window_size
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    train_dataset = HSIDataset(
        X_train,
        y_train
    )

    test_dataset = HSIDataset(
        X_test,
        y_test
    )

    return train_dataset, test_dataset