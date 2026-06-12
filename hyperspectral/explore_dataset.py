from scipy.io import loadmat

data = loadmat("datasets/Indian_pines_corrected.mat")
labels = loadmat("datasets/Indian_pines_gt.mat")

print("DATA KEYS:")
print(data.keys())

print("\nLABEL KEYS:")
print(labels.keys())