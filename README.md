# Classificazione di immagini iperspettrali mediante CNN e Kolmogorov-Arnold Networks

Progetto sviluppato per il corso di **Computer Vision and Deep Learning**.

L'obiettivo è confrontare differenti architetture neurali per la classificazione di immagini iperspettrali utilizzando il dataset **Indian Pines**, valutandone sia le prestazioni di classificazione sia l'impatto ambientale durante l'addestramento e il testing.

---

# Modelli implementati
Sono stati implementati e confrontati tre modelli:

- CNN (baseline convoluzionale)
- Conv-KAGN
- Conv-WavKAN

Tutti i modelli condividono lo stesso protocollo sperimentale, mantenendo invariati il preprocessing, la suddivisione del dataset e gli iperparametri di addestramento, in modo da ottenere un confronto sperimentale il più possibile equo.

---

# Dataset
È stato utilizzato il dataset pubblico **Indian Pines**, costituito da immagini iperspettrali acquisite dal sensore AVIRIS.

Caratteristiche principali:

- 145 × 145 pixel
- 200 bande spettrali
- 16 classi
- patch 9 × 9

Download:

- Dati:
  http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat

- Labels:
  http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat

---

# Preparazione del dataset
Il caricamento del dataset è gestito automaticamente dal file

```
hyperspectral/dataset_loader_fixed.py
```

Durante il caricamento vengono eseguite automaticamente le seguenti operazioni:

- caricamento dei file `.mat`
- estrazione delle patch 9 × 9
- suddivisione stratificata Train / Validation / Test
- normalizzazione tramite StandardScaler
- creazione dei DataLoader PyTorch

Lo split utilizzato è:

- Training: 20%
- Validation: 10%
- Test: 70%

con seed fisso pari a 42.

---

# Addestramento
L'intero processo di addestramento è gestito tramite

```
addestramento.py
```

Per selezionare il modello è sufficiente specificarlo da riga di comando.

CNN

```bash
python addestramento.py cnn
```

Conv-KAGN

```bash
python addestramento.py kagn
```

Conv-WavKAN

```bash
python addestramento.py wavkan
```

---

# Parametri sperimentali
Per tutti gli esperimenti sono stati utilizzati:

| Parametro         | Valore    |
|-------------------|-----------|
| Batch size        | 64        |
| Learning Rate     | 0.003     |
| Epoch             | 100       |
| Optimizer         | Adam      |
| Label Smoothing   | 0.1       |
| Patch Size        | 9 × 9     |
| Seed              | 42        |

---

# Monitoraggio delle emissioni
Le emissioni di CO₂ vengono monitorate tramite la libreria **CodeCarbon**.

Per ogni esecuzione vengono registrati separatamente:

- Training CO₂
- Testing CO₂

---

# Output
Durante l'esecuzione vengono generati:

## logs/
Contiene, per ogni modello:

```
cnn_log.csv
kagn_log.csv
wavkan_log.csv
```

Ogni file include:

- Loss di training
- Accuratezza di training
- Accuratezza di validazione
- CO₂ prodotta durante il training
- CO₂ prodotta durante il testing
- Accuratezza finale sul test set

---

## results/
Contiene un file riassuntivo più leggibile per ciascun modello con:

- accuratezza finale
- parametri dell'esperimento
- dispositivo utilizzato
- numero di parametri del modello
- emissioni di CO₂

---

# Struttura della repository
.
├── addestramento.py
├── hyperspectral/
│   └── dataset_loader.py
├── models/
├── kans/
├── kan_convs/
├── logs/
├── results/
└── README.md

Il progetto è distribuito a scopo didattico e di ricerca.