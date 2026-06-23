Pipeline di preparazione del dataset Indian Pines
1. explore_dataset.py
Obiettivo

Il primo passo è stato analizzare la struttura del dataset Indian Pines prima di applicare qualsiasi algoritmo di Deep Learning.

Il dataset è composto da un'unica immagine iperspettrale acquisita dal sensore AVIRIS.

Ogni pixel dell'immagine non contiene solamente le classiche informazioni RGB, ma una firma spettrale composta da 200 bande.

Lo scopo di explore_dataset.py è stato comprendere:

dimensioni del dataset;
numero di bande spettrali;
numero di classi;
distribuzione dei campioni;
presenza di eventuali sbilanciamenti.
Operazioni effettuate

Sono stati caricati i due file .mat:

Indian_pines_corrected.mat
Indian_pines_gt.mat

Il primo contiene i dati iperspettrali.

Il secondo contiene le etichette (ground truth).

Sono state estratte le seguenti informazioni:

Shape dati: (145,145,200)

Shape etichette: (145,145)

Classi: 16

Pixel etichettati: 10249
Considerazione importante

Il dataset è fortemente sbilanciato.

Alcune classi possiedono migliaia di campioni, mentre altre soltanto poche decine.

Ad esempio:

Classe 11 → 2455 pixel

Classe 9 → 20 pixel

Questa osservazione è molto importante perché influenza l'addestramento delle reti neurali.

2. prepare_dataset.py
Obiettivo

Le reti neurali non possono utilizzare direttamente la struttura originale del dataset.

L'immagine originale è infatti organizzata come:

145 × 145 × 200

dove:

145 = altezza
145 = larghezza
200 = bande spettrali

L'obiettivo di prepare_dataset.py è trasformare l'immagine in un dataset supervisionato classico.

Operazioni effettuate

Sono stati eliminati tutti i pixel appartenenti allo sfondo.

Lo sfondo è identificato dalla label:

0

È stata creata una maschera:

mask = y > 0

Successivamente ogni pixel etichettato è stato trasformato in un singolo campione.

La trasformazione è:

(145,145,200)

↓

(10249,200)

Ogni riga rappresenta un pixel.

Ogni colonna rappresenta una banda spettrale.

Le etichette diventano:

(10249,)
Significato

Ogni campione è costituito unicamente dall'informazione spettrale.

In questa fase non viene utilizzata alcuna informazione spaziale.

3. patch_dataset.py
Obiettivo

La sola informazione spettrale spesso non è sufficiente.

È importante considerare anche il contesto spaziale circostante.

Per questo motivo è stato costruito un dataset basato su patch.

Concetto di patch

Per ogni pixel etichettato viene estratta una finestra quadrata centrata sul pixel stesso.

È stata scelta una patch:

9 × 9

Poiché ogni pixel contiene 200 bande spettrali, la dimensione finale di ogni campione diventa:

9 × 9 × 200
Padding

Per estrarre correttamente le patch ai bordi dell'immagine è stato applicato un padding di 4 pixel.

L'immagine passa da:

145 × 145 × 200

a:

153 × 153 × 200
Dataset finale

Sono state create:

10249 patch

La struttura finale diventa:

X_patches.shape

(10249,9,9,200)

y_patches.shape

(10249,)
Significato di una patch

Ogni campione contiene contemporaneamente:

Informazione spaziale
9 × 9

ovvero il vicinato del pixel.

Informazione spettrale
200

ovvero la firma spettrale di ogni pixel.

In altre parole:

Una patch rappresenta una piccola porzione di territorio osservata in 200 diverse frequenze elettromagnetiche.

4. dataset_loader.py
Obiettivo

Creare un componente riutilizzabile che prepari automaticamente il dataset per tutte le reti neurali.

Questo script evita di duplicare il codice all'interno dei vari esperimenti.

Operazioni effettuate
1. Caricamento

Vengono caricati:

X_patches.npy

y_patches.npy
2. Riordino dimensioni

PyTorch richiede il formato:

(batch,channels,height,width)

Mentre le patch sono salvate come:

(10249,9,9,200)

Vengono convertite in:

(10249,200,9,9)
3. Suddivisione del dataset

Viene effettuato un train-test split:

80%

20%

mantenendo la distribuzione delle classi tramite:

stratify=y
4. Normalizzazione

Viene applicata la Standardizzazione:

z = (x - media) / deviazione_standard

La normalizzazione viene calcolata solamente sul training set per evitare il fenomeno del data leakage.

5. DataLoader

Infine vengono creati:

train_loader

test_loader

che verranno utilizzati da tutte le reti neurali.

Riassunto finale (molto utile da dire al professore)

Il preprocessing è stato suddiviso in quattro fasi progressive. Prima abbiamo analizzato il dataset con explore_dataset.py, poi abbiamo trasformato l'immagine iperspettrale in un dataset supervisionato tramite prepare_dataset.py. Successivamente abbiamo introdotto il contesto spaziale creando patch 9×9×200 con patch_dataset.py. Infine abbiamo centralizzato tutto il preprocessing in dataset_loader.py, che si occupa automaticamente di normalizzazione, suddivisione train-test e creazione dei DataLoader da utilizzare nelle varie architetture neurali.

Secondo me questa spiegazione è già adatta a un'esposizione universitaria.

# Adattamento di mnist_conv.py ad Indian Pines

| Modifica | Descrizione | Motivazione |
|----------|-------------|-------------|
| 1 | Rimozione MNIST/CIFAR | Utilizzo di Indian Pines |
| 2 | Rimozione trasformazioni torchvision | Dataset già preprocessato |
| 3 | Inserimento dataset_loader | Riutilizzo pipeline |
| 4 | Numero classi: 16 | Indian Pines |
| 5 | Input channels: 200 | Bande spettrali |
| 6 | Selezione modelli | Vanilla, KAGN, WavKAN |

