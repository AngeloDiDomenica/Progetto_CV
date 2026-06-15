from scipy.io import loadmat
import numpy as np

# =====================
# PARAMETRI
# =====================

PATCH_SIZE = 9
HALF = PATCH_SIZE // 2

# =====================
# CARICAMENTO DATI
# =====================

data = loadmat("Indian_pines_corrected (1).mat")
labels = loadmat("Indian_pines_gt.mat")

X = data["indian_pines_corrected"]
y = labels["indian_pines_gt"]

print("Dataset originale:", X.shape)

# =====================
# PADDING
# =====================

X_padded = np.pad(
    X,
    ((HALF, HALF),
     (HALF, HALF),
     (0, 0)),
    mode="constant"
)

print("Dataset con padding:", X_padded.shape)

# =====================
# ESTRAZIONE PATCH
# =====================

patches = []
labels_list = []

for i in range(y.shape[0]):

    for j in range(y.shape[1]):

        label = y[i, j]

        # ignora lo sfondo
        if label == 0:
            continue

        patch = X_padded[
            i:i+PATCH_SIZE,
            j:j+PATCH_SIZE,
            :
        ]

        patches.append(patch)

        labels_list.append(label)

# =====================
# ARRAY FINALI
# =====================

X_patches = np.array(patches)

y_patches = np.array(labels_list)

print("\nShape patches:")
print(X_patches.shape)

print("\nShape labels:")
print(y_patches.shape)

print("\nPrima patch:")
print(X_patches[0].shape)

print("\nClasse prima patch:")
print(y_patches[0])