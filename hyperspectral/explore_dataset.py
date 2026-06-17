from scipy.io import loadmat
import numpy as np

data = loadmat("Indian_pines_corrected.mat")
labels = loadmat("Indian_pines_gt.mat")

X = data["indian_pines_corrected"]
y = labels["indian_pines_gt"]

print("Shape dati:", X.shape)
print("Shape labels:", y.shape)

print("\nTipo dati:", X.dtype)

print("\nValore minimo:", np.min(X))
print("Valore massimo:", np.max(X))

print("\nClassi presenti:")
print(np.unique(y))

print("\nNumero classi (escluso sfondo):")
print(len(np.unique(y)) - 1)

print("\nDistribuzione classi:")

unique, counts = np.unique(y, return_counts=True)

for cls, count in zip(unique, counts):
    print(f"Classe {cls}: {count} pixel")


print("\nNumero pixel etichettati:")
print(np.sum(y > 0))

print("\nPercentuale pixel etichettati:")
print(np.sum(y > 0) / y.size * 100)