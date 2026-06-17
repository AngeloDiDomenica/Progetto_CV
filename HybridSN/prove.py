import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "hyperspectral")

print("CONTENUTO CARTELLA:")
print(os.listdir(DATA_DIR))