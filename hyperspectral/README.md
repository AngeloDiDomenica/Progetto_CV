# Hyperspectral Classification

Dataset:
- Indian Pines

Obiettivo:
- CNN baseline
- Conv-KAN comparison

Task:
- Hyperspectral image classification

# Hyperspectral Image Classification - Indian Pines

## Obiettivo

Valutare le prestazioni delle reti neurali standard e delle Conv-KAN nella classificazione di immagini iperspettrali utilizzando il dataset Indian Pines.

L'obiettivo è costruire una baseline CNN e successivamente confrontarla con un'architettura Conv-KAN basata sull'implementazione di Ivan Drokin.

---

## Dataset

### Indian Pines

Dataset hyperspectral acquisito tramite sensore AVIRIS.

Caratteristiche:

* Dimensione immagine: 145 × 145 pixel
* Bande spettrali: 200
* Classi: 16
* Pixel totali: 21025
* Pixel etichettati: 10249
* Percentuale pixel etichettati: 48.75%

---

## Esplorazione Dataset

### Shape

Dati:

```python
(145, 145, 200)
```

Label:

```python
(145, 145)
```

### Range valori

Tipo:

```python
uint16
```

Valore minimo:

```python
955
```

Valore massimo:

```python
9604
```

### Classi presenti

```python
[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
```

dove:

* 0 = sfondo
* 1-16 = classi reali

---

## Distribuzione classi

| Classe | Pixel |
| ------ | ----: |
| 1      |    46 |
| 2      |  1428 |
| 3      |   830 |
| 4      |   237 |
| 5      |   483 |
| 6      |   730 |
| 7      |    28 |
| 8      |   478 |
| 9      |    20 |
| 10     |   972 |
| 11     |  2455 |
| 12     |   593 |
| 13     |   205 |
| 14     |  1265 |
| 15     |   386 |
| 16     |    93 |

Osservazione:

Il dataset è fortemente sbilanciato. Alcune classi possiedono meno di 50 campioni mentre altre superano i 2000 campioni.

---

## Preprocessing

### Rimozione sfondo

Sono stati mantenuti esclusivamente i pixel con etichetta diversa da zero.

```python
mask = y > 0
```

### Dataset finale

Input:

```python
X_samples.shape = (10249, 200)
```

Target:

```python
y_samples.shape = (10249,)
```

Ogni campione corrisponde alla firma spettrale di un singolo pixel.

---

## Pipeline prevista

1. Esplorazione dataset
2. Preprocessing
3. Baseline MLP
4. Creazione patch spaziali
5. CNN baseline
6. Conv-KAN
7. Confronto risultati

---

## Stato attuale

* [x] Download dataset
* [x] Analisi struttura dati
* [x] Analisi distribuzione classi
* [x] Preprocessing pixel-wise
* [ ] Baseline MLP
* [ ] Patch extraction
* [ ] CNN baseline
* [ ] Conv-KAN
* [ ] Analisi finale


## Baseline MLP

Architettura:

200 -> 128 -> 64 -> 16

Attivazione:
ReLU

Ottimizzatore:
Adam

Learning Rate:
0.001

Epoche:
20

Accuracy finale:
90.29%

Device:
CPU

Osservazioni:

Nonostante l'assenza di informazioni spaziali, la sola firma spettrale dei pixel permette di raggiungere un'accuratezza superiore al 90%.


## Baseline MLP

Dataset: Indian Pines

Input:
200 bande spettrali

Architettura:
200 -> 128 -> 64 -> 16

Ottimizzatore:
Adam

Learning Rate:
0.001

Epoche:
20

Device:
CPU

Accuracy:
90.29%