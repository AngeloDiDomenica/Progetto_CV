## Obiettivo

Confrontare una CNN standard e una Conv-KAN nella classificazione di immagini iperspettrali utilizzando il dataset Indian Pines.

## Stato del progetto

- [x] Download dataset
- [x] Analisi dataset
- [x] Analisi distribuzione classi
- [x] Preprocessing pixel-wise
- [x] Baseline MLP
- [x] Patch extraction
- [x] Implementazione CNN baseline
- [ ] Training CNN baseline
- [ ] Conv-KAN
- [ ] Confronto finale

## CNN Baseline

Input:

9×9×200

Pipeline:

(10249,9,9,200)

↓

(10249,200,9,9)

↓

Conv2D(200→64)

↓

ReLU

↓

MaxPool2D

↓

Conv2D(64→128)

↓

ReLU

↓

Flatten

↓

Linear(2048→128)

↓

Linear(128→16)

Parametri:

- Learning Rate: 0.001
- Batch Size: 64
- Epoche: 20
- Random Seed: 42

| Esperimento | Input | Accuracy |
|-------------|-------|----------|
| MLP | 200 | 90.29% |
| CNN | 9×9×200 | ? |
| Conv-KAN | 9×9×200 | ? |

## Osservazioni

- Indian Pines contiene una singola immagine iperspettrale di dimensione 145×145×200.
- Il dataset è fortemente sbilanciato.
- Sono stati utilizzati solo i pixel etichettati.
- È stata costruita una baseline MLP utilizzando esclusivamente l'informazione spettrale.
- Successivamente sono state create patch 9×9 per introdurre l'informazione spaziale.


## CNN Baseline

Accuracy finale:

99.90%

Numero parametri:

453456

Device:

CPU

Osservazione:

L'introduzione dell'informazione spaziale (patch 9×9) migliora drasticamente le prestazioni rispetto alla baseline MLP.

È importante notare che è stato utilizzato uno split casuale dei pixel appartenenti alla stessa immagine. Questo approccio può introdurre una sovrastima delle prestazioni a causa della forte correlazione spaziale tra train e test.

L'aggiunta dell'informazione spaziale sembra migliorare drasticamente la classificazione rispetto alla sola firma spettrale, anche se il protocollo di validazione potrebbe sovrastimare le prestazioni a causa della correlazione spaziale tra train e test.

# Stato attuale

Data: 15/06/2026

Baseline CNN completata.

Accuracy: 99.90%

Configurazione:

- Patch: 9x9
- Batch size: 64
- Learning rate: 0.001
- Epoche: 20
- Label smoothing: 0.1
- Seed: 42

Nota:
Il train/test split è stato effettuato casualmente sui pixel, quindi il risultato potrebbe essere influenzato da spatial leakage.

CNN custom: baseline preliminare sviluppata per verificare la pipeline di preprocessing del dataset Indian Pines. Il confronto ufficiale viene effettuato utilizzando le baseline del framework di Ivan Drokin (Vanilla, KAGN e WavKAN).