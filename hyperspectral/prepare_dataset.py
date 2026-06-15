from scipy.io import loadmat
import numpy as np

# Caricamento dataset
data = loadmat("Indian_pines_corrected (1).mat")
labels = loadmat("Indian_pines_gt.mat")

X = data["indian_pines_corrected"]
y = labels["indian_pines_gt"]

# Tieni solo i pixel etichettati
mask = y > 0

X_samples = X[mask]
y_samples = y[mask]

print("Shape originale X:", X.shape)
print("Shape originale y:", y.shape)

print("\nShape X_samples:", X_samples.shape)
print("Shape y_samples:", y_samples.shape)

print("\nPrimo campione:")
print(X_samples[0])

print("\nClasse del primo campione:")
print(y_samples[0])