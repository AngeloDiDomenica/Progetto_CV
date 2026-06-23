MODIFICA 1

File:
indian_pines_conv.py

Codice originale:

from torchvision.transforms import v2

from torchvision.transforms.autoaugment import AutoAugmentPolicy

Azione eseguita:

Commentato.

Motivazione:

Le trasformazioni torchvision erano utilizzate dal repository originale per effettuare data augmentation su dataset di immagini tradizionali come MNIST.

Indian Pines è un dataset hyperspectral già preprocessato mediante patch 9×9×200, pertanto queste trasformazioni non sono necessarie.

Il codice originale è stato mantenuto commentato per garantire la tracciabilità delle modifiche apportate al repository.