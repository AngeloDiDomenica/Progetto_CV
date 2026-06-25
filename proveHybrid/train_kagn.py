"""
Adattamento headless (no Jupyter) di HybridSN.ipynb per Indian Pines.
Modifiche rispetto al notebook originale:
- backend matplotlib 'Agg' (niente GUI) invece di %matplotlib inline
- tqdm normale invece di tqdm.notebook
- percorsi con os.path.join invece di backslash stile Windows
- niente spectral.imshow interattivo (si tengono solo i salvataggi su file)
"""
import time
import os
import matplotlib
matplotlib.use('Agg')  # niente display, ambiente headless

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
import numpy as np
import matplotlib.pyplot as plt
import spectral
import random
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, recall_score, cohen_kappa_score, accuracy_score
from scipy.io import loadmat
from tqdm import tqdm

        # Importa la classe dal tuo nuovo file
from Kagn import KAGNConv3DLayerV2, KAGNConv2DLayerV2

# ----------------------------- HYPERPARAMETERS -----------------------------
RANDOM_SEED = 666
DATASET = 'IP'          # Pavia University
TRAIN_RATE = 0.03       # <-- MODIFICATO (3%)
VAL_RATE = 0.02         # <-- MODIFICATO (2%)
EPOCH = 10             # <-- MODIFICATO
VAL_EPOCH = 5
LR = 0.003             # <-- MODIFICATO
WEIGHT_DECAY = 1e-4     # <-- MODIFICATO
BATCH_SIZE = 64         # <-- MODIFICATO
DEVICE = 0              #-1=cpu, 0=cuda
N_PCA = 15
PATCH_SIZE = 30         # <-- MODIFICATO (Opzionale ma consigliato)
SAVE_PATH = os.path.join('results', DATASET)
os.makedirs(SAVE_PATH, exist_ok=True)

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ----------------------------- DATA LOADING -----------------------------
def loadData(name):
    data_path = os.path.join(os.getcwd(), 'dataset')
    if name == 'IP':#indian
        data = loadmat(os.path.join(data_path, 'IndianPines', 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = loadmat(os.path.join(data_path, 'IndianPines', 'Indian_pines_gt.mat'))['indian_pines_gt']
        class_name = ["Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture",
                      "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed", "Oats", "Soybean-notill",
                      "Soybean-mintill", "Soybean-clean", "Wheat", "Woods",
                      "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers"]
    elif name == 'SA':
        data = loadmat(os.path.join(data_path, 'Salinas', 'Salinas_corrected.mat'))['salinas_corrected']
        labels = loadmat(os.path.join(data_path, 'Salinas', 'Salinas_gt.mat'))['salinas_gt']
        class_name = ['Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 'Fallow_rough_plow',
                      'Fallow_smooth', 'Stubble', 'Celery', 'Grapes_untrained', 'Soil_vinyard_develop',
                      'Corn_senesced_green', 'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk',
                      'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk', 'Vinyard_untrained', 'Vinyard_vertical']
    elif name == 'PU':
        data = loadmat(os.path.join(data_path, 'PaviaU', 'PaviaU.mat'))['paviaU']
        labels = loadmat(os.path.join(data_path, 'PaviaU', 'PaviaU_gt.mat'))['paviaU_gt']
        class_name = ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets', 'Bare Soil',
                      'Bitumen', 'Self-Blocking Bricks', 'Shadows']
    return data, labels, class_name


print(f"[INFO] Caricamento dataset {DATASET}...")
data, label, class_name = loadData(DATASET)
NUM_CLASS = label.max()
print(f"[INFO] data shape: {data.shape}, label shape: {label.shape}, classi: {NUM_CLASS}")

# salvataggio immagini RGB/GT (no display interattivo)
spectral.save_rgb(os.path.join('results', f'{DATASET}_RGB_origin.jpg'), data, (30, 20, 10))
spectral.save_rgb(os.path.join('results', f'{DATASET}_gt.jpg'), label, colors=spectral.spy_colors)


def applyPCA(X, numComponents=15):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX, pca


data, pca = applyPCA(data, N_PCA)
print(f"[INFO] shape dopo PCA: {data.shape}")


def sample_gt(gt, train_rate):
    indices = np.nonzero(gt)
    X = list(zip(*indices))
    y = gt[indices].ravel()
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    if train_rate > 1:
        train_rate = int(train_rate)
    train_indices, test_indices = train_test_split(X, train_size=train_rate, stratify=y, random_state=100)
    train_indices = [t for t in zip(*train_indices)]
    test_indices = [t for t in zip(*test_indices)]
    train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
    test_gt[tuple(test_indices)] = gt[tuple(test_indices)]
    return train_gt, test_gt


train_gt, test_gt = sample_gt(label, TRAIN_RATE)
val_gt, test_gt = sample_gt(test_gt, VAL_RATE / (1 - TRAIN_RATE))

sample_report = f"{'class': ^10}{'train_num':^10}{'val_num': ^10}{'test_num': ^10}{'total': ^10}\n"
for i in np.unique(label):
    if i == 0:
        continue
    sample_report += f"{i: ^10}{(train_gt==i).sum(): ^10}{(val_gt==i).sum(): ^10}{(test_gt==i).sum(): ^10}{(label==i).sum(): ^10}\n"
sample_report += f"{'total': ^10}{np.count_nonzero(train_gt): ^10}{np.count_nonzero(val_gt): ^10}{np.count_nonzero(test_gt): ^10}{np.count_nonzero(label): ^10}"
print(sample_report)


# ----------------------------- DATASET -----------------------------
class PatchSet(Dataset):
    def __init__(self, data, gt, patch_size, is_pred=False):
        super(PatchSet, self).__init__()
        self.is_pred = is_pred
        self.patch_size = patch_size
        p = self.patch_size // 2
        self.data = np.pad(data, ((p, p), (p, p), (0, 0)), 'constant', constant_values=0)
        if is_pred:
            gt = np.ones_like(gt)
        self.label = np.pad(gt, (p, p), 'constant', constant_values=0)
        x_pos, y_pos = np.nonzero(gt)
        x_pos, y_pos = x_pos + p, y_pos + p
        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos)])
        if not is_pred:
            np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size
        data = self.data[x1:x2, y1:y2]
        label = self.label[x, y]
        data = np.asarray(data, dtype='float32').transpose((2, 0, 1))
        label = np.asarray(label, dtype='int64')
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        if self.is_pred:
            return data
        else:
            return data, label


train_data = PatchSet(data, train_gt, PATCH_SIZE)
val_data = PatchSet(data, val_gt, PATCH_SIZE)
all_data = PatchSet(data, label, PATCH_SIZE, is_pred=True)
train_loader = DataLoader(train_data, BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, BATCH_SIZE, shuffle=True)
all_loader = DataLoader(all_data, BATCH_SIZE, shuffle=False)

d, g = train_data.__getitem__(0)
print(f"[INFO] esempio patch shape: {d.shape}, label: {g}")


# ----------------------------- MODEL -----------------------------
class HybridSN(nn.Module):
    def __init__(self, in_chs, patch_size, class_nums, degree=3):
        super().__init__()
        self.in_chs = in_chs
        self.patch_size = patch_size
        

        # --- REPARTO 3D (KAGN) ---
        self.conv1 = KAGNConv3DLayerV2(
            input_dim=1, 
            output_dim=8, 
            kernel_size=(7, 3, 3),
            degree=degree
        )
        self.conv2 = KAGNConv3DLayerV2(
            input_dim=8, 
            output_dim=16, 
            kernel_size=(5, 3, 3),
            degree=degree
        )
        self.conv3 = KAGNConv3DLayerV2(
            input_dim=16, 
            output_dim=32, 
            kernel_size=(3, 3, 3),
            degree=degree
        )

        self.x1_shape = self.get_shape_after_3dconv()
        
        # --- PONTE E REPARTO 2D (KAGN) ---
        self.conv4 = KAGNConv2DLayerV2(
            input_dim=self.x1_shape[1] * self.x1_shape[2], 
            output_dim=64, 
            kernel_size=(3, 3),
            degree=degree
        )
        self.x2_shape = self.get_shape_after_2dconv()
        
        # --- REPARTO DECISIONALE (Standard Lineare) ---
        self.dense1 = nn.Sequential(
            nn.Linear(self.x2_shape, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4))
        self.dense2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4))
        self.dense3 = nn.Sequential(
            nn.Linear(128, class_nums))

    def get_shape_after_2dconv(self):
        x = torch.randn((1, self.x1_shape[1] * self.x1_shape[2], self.x1_shape[3], self.x1_shape[4]))
        with torch.no_grad():
            x = self.conv4(x)
        return x.shape[1] * x.shape[2] * x.shape[3]

    def get_shape_after_3dconv(self):
        x = torch.randn((1, 1, self.in_chs, self.patch_size, self.patch_size))
        with torch.no_grad():
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
        return x.shape

    def forward(self, X):
            X = X.unsqueeze(1)
            x = self.conv1(X)
            x = self.conv2(x)
            x = self.conv3(x)
            
            # Ponte 3D -> 2D
            x = x.view(x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4])
            
            x = self.conv4(x)
            
            # Appiattimento per i layer densi
            x = x.contiguous().view(x.shape[0], -1)
            x = self.dense1(x)
            x = self.dense2(x)
            out = self.dense3(x)
            return out


net = HybridSN(N_PCA, PATCH_SIZE, class_nums=NUM_CLASS)
summary(net, input_size=(1, N_PCA, PATCH_SIZE, PATCH_SIZE),
        col_names=['num_params', 'kernel_size', 'mult_adds', 'input_size', 'output_size'],
        col_width=10, row_settings=['var_names'], depth=4)


# ----------------------------- TRAINING -----------------------------
device = torch.device(DEVICE if DEVICE >= 0 and torch.cuda.is_available() else 'cpu')#se è disponibile una gpu, la usa.

print(f"[INFO] device: {device}")
loss_list = []
acc_list = []
val_acc_list = []
val_epoch_list = []

#model = HybridSN(N_PCA, PATCH_SIZE, class_nums=NUM_CLASS)
model = HybridSN(N_PCA, PATCH_SIZE, class_nums=NUM_CLASS, degree=3)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), LR, weight_decay=WEIGHT_DECAY)
loss_func = nn.CrossEntropyLoss()
batch_num = len(train_loader)
train_num = train_loader.dataset.__len__()
val_num = val_loader.dataset.__len__()

t0 = time.time()
e = -1
try:
    for e in range(EPOCH):
        model.train()
        avg_loss = 0.
        train_acc = 0
        for batch_idx, (bdata, target) in enumerate(train_loader):
            bdata, target = bdata.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(bdata)
            target = target - 1
            loss = loss_func(out, target)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            _, pred = torch.max(out, dim=1)
            train_acc += (pred == target).sum().item()
        loss_list.append(avg_loss / train_num)
        acc_list.append(train_acc / train_num)
        elapsed = time.time() - t0
        print(f"epoch {e}/{EPOCH} loss:{loss_list[e]:.4f} acc:{acc_list[e]:.4f}  (elapsed {elapsed:.1f}s)")
        if (e + 1) % VAL_EPOCH == 0 or (e + 1) == EPOCH:
            val_acc = 0
            model.eval()
            with torch.no_grad():
                for batch_idx, (bdata, target) in enumerate(val_loader):
                    bdata, target = bdata.to(device), target.to(device)
                    out = model(bdata)
                    target = target - 1
                    _, pred = torch.max(out, dim=1)
                    val_acc += (pred == target).sum().item()
            val_acc_list.append(val_acc / val_num)
            val_epoch_list.append(e)
            print(f"epoch {e}/{EPOCH}  val_acc:{val_acc_list[-1]:.4f}")
            save_name = os.path.join(SAVE_PATH, f"epoch_{e}_acc_{val_acc_list[-1]:.4f}.pth")
            torch.save(model.state_dict(), save_name)
except Exception as exc:
    print(exc)
finally:
    print(f"Stop in epoch {e}, tempo totale: {time.time()-t0:.1f}s")

if loss_list:
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    ax1.plot(np.arange(e + 1), loss_list)
    ax1.set_title('loss')
    ax1.set_xlabel('epoch')
    ax2.plot(np.arange(e + 1), acc_list, label='train_acc')
    if val_acc_list:
        ax2.plot(val_epoch_list, val_acc_list, label='val_acc')
    ax2.set_title('acc')
    ax2.set_xlabel('epoch')
    ax2.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_PATH, 'training_curves.png'))

#adesso arriva la fase di validazione e inferenza.
# ----------------------------- INFERENCE -----------------------------
def get_best_model(acc_list, epoch_list, save_path):
    acc_list = np.array(acc_list)
    epoch_list = np.array(epoch_list)
    best_index = np.argwhere(acc_list == np.max(acc_list))[-1].item()
    best_epoch = epoch_list[best_index]
    best_acc = acc_list[best_index]
    file_name = f"epoch_{best_epoch}_acc_{best_acc:.4f}.pth"
    best_model_path = os.path.join(save_path, file_name)
    print(f"best model:{file_name}")
    for f in os.listdir(save_path):
        if f[-3:] == 'pth' and os.path.join(save_path, f) != best_model_path:
            os.remove(os.path.join(save_path, f))
    return best_model_path


if val_acc_list:
    best_model_path = get_best_model(val_acc_list, val_epoch_list, SAVE_PATH)
    best_model = HybridSN(N_PCA, PATCH_SIZE, class_nums=NUM_CLASS)
    best_model.load_state_dict(torch.load(best_model_path, map_location=device))
    best_model.to(device)
    best_model.eval()
    pred_map = []
    with torch.no_grad():
        for batch_idx, bdata in enumerate(all_loader):
            bdata = bdata.to(device)
            target = best_model(bdata)
            _, pred = torch.max(target, dim=1)
            pred_map += [np.array(pred.detach().cpu() + 1)]
    pred_map = np.asarray(np.hstack(pred_map), dtype=np.uint8).reshape(label.shape[0], label.shape[1])
    spectral.save_rgb(os.path.join(SAVE_PATH, "prediction.jpg"), pred_map, colors=spectral.spy_colors)
    spectral.save_rgb(os.path.join(SAVE_PATH, "prediction_masked.jpg"), pred_map * (label != 0), colors=spectral.spy_colors)

    test_pred = pred_map[test_gt != 0]
    test_true = test_gt[test_gt != 0]
    OA = accuracy_score(test_true, test_pred)
    AA = recall_score(test_true, test_pred, average='macro')
    kappa = cohen_kappa_score(test_true, test_pred)
    report_log = f"OA: {OA}\nAA: {AA}\nKappa: {kappa}\n"
    report_log += classification_report(test_true, test_pred, target_names=class_name, digits=4)
    print(report_log)
    with open(os.path.join(SAVE_PATH, 'classification_report.txt'), 'w+') as fp:
        fp.writelines(report_log)
else:
    print("[WARN] Nessuna validazione eseguita, salto l'inferenza completa.")
