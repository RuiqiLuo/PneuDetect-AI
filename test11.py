import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, cohen_kappa_score
from albumentations import (
    Compose, Normalize, Resize,
    HorizontalFlip, VerticalFlip, ShiftScaleRotate,
    GridDistortion, ElasticTransform, RandomRotate90,  # 替换 Rotate
    RandomGamma, CLAHE, RandomBrightnessContrast,
    CoarseDropout
)
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from PIL import Image
import glob

# --- 0. 全局设置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 2  # 例如：肺炎 vs. 正常 (assuming 2 classes like NORMAL/PNEUMONIA)
IMG_SIZE = 224
BATCH_SIZE = 16 # 根据您的GPU显存调整
NUM_EPOCHS = 50 # 示例值，根据实际情况调整
LEARNING_RATE = 0.001 # 初始学习率
PATIENCE_EARLY_STOPPING = 10 # 早停的耐心轮数

# --- 用户指定的输入路径 ---
DATA_DIR = 'D:\\archive\\chest_xray'

# 早期检查数据路径是否存在
if not os.path.isdir(DATA_DIR):
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"!!! ERROR: Dataset directory not found at '{DATA_DIR}'")
    print(f"!!! Please update the 'DATA_DIR' variable in the script.")
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    exit()

# 使用原始数据集中的 'train' 和 'test' 目录
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'test') # Using 'test' as validation

# --- 用户指定的输出路径 ---
OUTPUT_BASE_DIR = r'C:\Users\LRQ\Desktop\output'
if not os.path.exists(OUTPUT_BASE_DIR):
    os.makedirs(OUTPUT_BASE_DIR)
    print(f"Created output directory: {OUTPUT_BASE_DIR}")

MODEL_SAVE_PATH = os.path.join(OUTPUT_BASE_DIR, "best_medical_cnn_model.pth")
TRAINING_HISTORY_IMG_PATH = os.path.join(OUTPUT_BASE_DIR, "training_history.png")
GRAD_CAM_EXAMPLE_IMG_PATH = os.path.join(OUTPUT_BASE_DIR, "grad_cam_example.png")


# --- 1. 模型架构优化 ---

# (1) 残差块 (Residual Block)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

# (2) 通道注意力机制 (Channel Attention)
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

# (3) 整合的自定义CNN模型
class CustomMedicalCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES): # Removed img_size as it's global
        super().__init__()
        self.layer1 = self._make_layer(3, 64, stride=1)
        self.ca1 = ChannelAttention(64)
        self.pool1 = nn.MaxPool2d(2)

        self.layer2 = self._make_layer(64, 128, stride=2)
        self.ca2 = ChannelAttention(128)
        self.pool2 = nn.MaxPool2d(2)

        self.layer3 = self._make_layer(128, 256, stride=2)
        self.ca3 = ChannelAttention(256)
        self.pool3 = nn.MaxPool2d(2)

        self.layer4 = self._make_layer(256, 512, stride=2)
        self.ca4 = ChannelAttention(512)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def _make_layer(self, in_channels, out_channels, stride):
        return ResidualBlock(in_channels, out_channels, stride)

    def forward(self, x):
        x = self.layer1(x); x = self.ca1(x); x = self.pool1(x)
        x = self.layer2(x); x = self.ca2(x); x = self.pool2(x)
        x = self.layer3(x); x = self.ca3(x); x = self.pool3(x)
        x = self.layer4(x); x = self.ca4(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x

    def get_activations_gradient(self): return self.gradients
    def activations_hook(self, grad): self.gradients = grad
    def get_activations(self, x): # Simplified for direct access to layer4 for GradCAM
        x = self.layer1(x); x = self.pool1(x)
        x = self.layer2(x); x = self.pool2(x)
        x = self.layer3(x); x = self.pool3(x)
        return self.layer4(x)


# --- 2. 数据集和数据加载 ---
class MedicalImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d.name for d in os.scandir(root_dir) if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.idx_to_class = {i: cls_name for cls_name, i in self.class_to_idx.items()}

        if NUM_CLASSES != len(self.classes):
            print(f"Warning: NUM_CLASSES global ({NUM_CLASSES}) does not match found classes ({len(self.classes)}: {self.classes}). Adjust NUM_CLASSES or check dataset structure.")
            # Potentially override NUM_CLASSES if you want it to be dynamic,
            # but for consistency with other parts of the code, it's better to fix it or ensure it matches.
            # For this example, we'll proceed but this could cause issues if NUM_CLASSES is used for model output size.

        self.image_paths = []
        self.labels = []

        for cls_name in self.classes:
            cls_idx = self.class_to_idx[cls_name]
            cls_path = os.path.join(root_dir, cls_name)
            for ext in ('*.jpeg', '*.jpg', '*.png', '*.bmp', '*.gif'): # Common image extensions
                 self.image_paths.extend(glob.glob(os.path.join(cls_path, ext)))
                 self.labels.extend([cls_idx] * len(glob.glob(os.path.join(cls_path, ext)))) # Add labels for the found images

        if not self.image_paths:
            raise RuntimeError(f"No images found in {root_dir}. Check directory structure and image extensions.")

        print(f"Found {len(self.image_paths)} images in {len(self.classes)} classes in {root_dir}.")
        # Store class counts for potential use in weighted loss or sampling
        self.class_counts = np.bincount(self.labels, minlength=len(self.classes))


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert("RGB")
            image = np.array(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}. Returning a placeholder.")
            image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            # Ensure a tensor is returned even on error if transform is applied
            if self.transform:
                augmented = self.transform(image=image) # Apply transform to placeholder
                image = augmented['image']
            else: # Manual conversion if no albumentations transform
                image = torch.from_numpy(image).permute(2,0,1).float() / 255.0
            return {"image": image, "label": torch.tensor(label, dtype=torch.long), "path": img_path}


        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return {"image": image, "label": torch.tensor(label, dtype=torch.long), "path": img_path}


# 数据增强
def get_transforms(data_type='train'):
    if data_type == 'train':
        return Compose([
            Resize(IMG_SIZE, IMG_SIZE),
            RandomRotate90(p=0.5),  # 替换 Rotate，执行 90 度增量旋转
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            RandomGamma(gamma_limit=(80, 120), p=0.3),
            CoarseDropout(max_holes=8, max_height=int(IMG_SIZE*0.08), max_width=int(IMG_SIZE*0.08),
                          min_holes=1, min_height=int(IMG_SIZE*0.04), min_width=int(IMG_SIZE*0.04),
                          fill_value=0, mask_fill_value=None, p=0.3),
            ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
            GridDistortion(p=0.2),
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    elif data_type == 'valid' or data_type == 'test':
        return Compose([
            Resize(IMG_SIZE, IMG_SIZE),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
# --- 3. 类别不平衡处理 ---
class DynamicWeightedLoss(nn.Module):
    def __init__(self, class_counts, device, smoothing=0.9): # Beta for effective number
        super().__init__()
        if not isinstance(class_counts, np.ndarray):
            class_counts = np.array(class_counts)

        if np.any(class_counts == 0):
            print("Warning: One or more classes have zero samples in class_counts for DynamicWeightedLoss. Weights might be problematic. Using uniform weights.")
            weights = np.ones_like(class_counts, dtype=float) / len(class_counts)
        else:
            effective_num = 1.0 - np.power(smoothing, class_counts)
            weights = (1.0 - smoothing) / effective_num
            weights = weights / np.sum(weights) * len(class_counts) # Normalize

        self.weights = torch.FloatTensor(weights).to(device)
        print(f"Using DynamicWeightedLoss with weights: {self.weights}")

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weights, reduction='mean')
        return ce_loss

class TwoStreamSampler(Sampler):
    def __init__(self, dataset_labels_array, batch_size, minority_class_label=0, majority_class_label=1, minority_ratio=0.5):
        self.dataset_labels = np.array(dataset_labels_array)
        self.batch_size = batch_size
        self.minority_ratio = minority_ratio
        self.minority_indices = np.where(self.dataset_labels == minority_class_label)[0]
        self.majority_indices = np.where(self.dataset_labels == majority_class_label)[0]
        self.num_samples = len(self.dataset_labels)

        if len(self.minority_indices) == 0:
            print(f"Warning: No samples found for minority class {minority_class_label} in TwoStreamSampler.")
        if len(self.majority_indices) == 0:
            print(f"Warning: No samples found for majority class {majority_class_label} in TwoStreamSampler.")

    def __iter__(self):
        n_batches = self.num_samples // self.batch_size
        all_indices = []
        for _ in range(n_batches):
            minority_size = int(self.batch_size * self.minority_ratio)
            majority_size = self.batch_size - minority_size
            batch_indices_list = []

            if len(self.minority_indices) > 0:
                chosen_minority = np.random.choice(
                    self.minority_indices,
                    size=minority_size,
                    replace=len(self.minority_indices) < minority_size
                )
                chosen_minority = np.atleast_1d(chosen_minority)
                batch_indices_list.append(chosen_minority)

            if len(self.majority_indices) > 0:
                chosen_majority = np.random.choice(
                    self.majority_indices,
                    size=majority_size,
                    replace=len(self.majority_indices) < majority_size
                )
                chosen_majority = np.atleast_1d(chosen_majority)
                batch_indices_list.append(chosen_majority)

            if not batch_indices_list:
                current_batch_indices = np.random.choice(
                    np.arange(len(self.dataset_labels)),
                    size=self.batch_size,
                    replace=len(self.dataset_labels) < self.batch_size
                )
            else:
                current_batch_indices = np.concatenate(batch_indices_list)

            np.random.shuffle(current_batch_indices)
            all_indices.extend(current_batch_indices.tolist())
        return iter(all_indices)

    def __len__(self):
        return self.num_samples // self.batch_size


    def __len__(self):
        # This should ideally return the number of samples yielded by __iter__
        return (self.num_samples // self.batch_size) * self.batch_size

# --- 4. 训练策略优化 (混合精度在训练循环中) ---

# --- 5. 后处理优化 ---
def find_optimal_threshold(model, val_loader, device):
    model.eval()
    all_probs_class1 = []
    all_labels_list = [] # Renamed to avoid conflict with global 'labels'

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Finding Optimal Threshold"):
            inputs = batch['image'].to(device)
            current_labels = batch['label'].to(device) # Renamed variable

            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            all_probs_class1.extend(probs[:, 1].cpu().numpy())
            all_labels_list.extend(current_labels.cpu().numpy())

    all_probs_class1_np = np.array(all_probs_class1) # Converted to numpy array
    all_labels_np = np.array(all_labels_list)       # Converted to numpy array

    if len(np.unique(all_labels_np)) < 2:
        print("Warning: Optimal threshold calculation requires at least two classes in validation labels. Using default 0.5.")
        return 0.5

    fpr, tpr, thresholds = roc_curve(all_labels_np, all_probs_class1_np)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal threshold found: {optimal_threshold:.4f}")
    return optimal_threshold

class ModelEnsemble(nn.Module):
    # ... (ModelEnsemble class from previous response, ensure CustomMedicalCNN is defined correctly)
    def __init__(self, model_paths, device):
        super().__init__()
        self.models = []
        self.device = device
        for path in model_paths:
            try:
                # Assuming CustomMedicalCNN is defined globally and takes num_classes
                model = CustomMedicalCNN(num_classes=NUM_CLASSES).to(self.device)
                model.load_state_dict(torch.load(path, map_location=self.device))
                model.eval()
                self.models.append(model)
                print(f"Loaded model from {path}")
            except Exception as e:
                print(f"Error loading model from {path}: {e}")
        if not self.models:
            raise ValueError("No models were loaded for ensemble.")

    def forward(self, x):
        outputs_list = []
        with torch.no_grad():
            for model in self.models:
                model_output = model(x)
                outputs_list.append(F.softmax(model_output, dim=1))
        ensembled_probs = torch.mean(torch.stack(outputs_list), dim=0)
        return ensembled_probs

# --- 6. 关键性能监控 (在训练循环中集成) ---

# --- Grad-CAM 实现 ---
class GradCAM:
    def __init__(self, model, target_layer_name_or_module): # Can accept name or module
        self.model = model
        self.gradient = None
        self.activation = None

        if isinstance(target_layer_name_or_module, str):
            # Find the layer by name
            self.target_layer = None
            for name, module in self.model.named_modules():
                if name == target_layer_name_or_module:
                    self.target_layer = module
                    break
            if self.target_layer is None:
                raise ValueError(f"Target layer '{target_layer_name_or_module}' not found in the model.")
        elif isinstance(target_layer_name_or_module, nn.Module):
            self.target_layer = target_layer_name_or_module
        else:
            raise TypeError("target_layer_name_or_module must be a string (layer name) or an nn.Module instance.")

        self.hook_layers()


    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activation = output
            output.register_hook(self.backward_hook) # Register backward hook on the output tensor

        def backward_hook(grad):
            self.gradient = grad
            return None

        self.target_layer.register_forward_hook(forward_hook)
        # No need to register backward hook on the layer itself, it's on the output tensor from forward.


    def generate_cam(self, input_image_tensor, class_idx=None, retain_graph=False):
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_image_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        one_hot_output = torch.zeros_like(output)
        one_hot_output[0][class_idx] = 1
        output.backward(gradient=one_hot_output, retain_graph=retain_graph)

        if self.gradient is None or self.activation is None:
            raise RuntimeError("Gradient or activation not captured. Check hook registration and backward pass.")

        # Detach activation before modifying
        activation_map = self.activation.detach() # Shape (1, C, H, W)
        gradients_val = self.gradient.detach()    # Shape (1, C, H, W)

        pooled_gradients = torch.mean(gradients_val, dim=[0, 2, 3]) # Global avg pooling of gradients

        for i in range(pooled_gradients.size(0)): # num_channels_in_target_layer
            activation_map[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activation_map, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        if torch.max(heatmap) > 0: # Avoid division by zero if heatmap is all zeros
            heatmap /= torch.max(heatmap)
        return heatmap.cpu().numpy(), class_idx


def show_cam_on_image(img_pil, cam_heatmap, save_path=None):
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR) # PIL to CV BGR
    heatmap_resized = cv2.resize(cam_heatmap, (img_cv.shape[1], img_cv.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    superimposed_img = heatmap_colored * 0.4 + img_cv * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1); plt.imshow(img_pil); plt.title("Original Image"); plt.axis('off')
    plt.subplot(1, 2, 2); plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)); plt.title("Grad-CAM"); plt.axis('off') # CV BGR to Matplotlib RGB

    if save_path:
        plt.savefig(save_path)
        print(f"Grad-CAM saved to {save_path}")
    plt.show()


# --- 训练和评估循环 ---
def train_one_epoch(model, train_loader, criterion, optimizer, device, scaler=None, scheduler=None):
    model.train()
    running_loss = 0.0
    all_preds_list = [] # Renamed
    all_labels_list = [] # Renamed

    progress_bar = tqdm(train_loader, desc="Training Epoch")
    for batch in progress_bar:
        inputs = batch['image'].to(device)
        current_labels = batch['label'].to(device) # Renamed

        optimizer.zero_grad()

        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, current_labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, current_labels)
            loss.backward()
            optimizer.step()

        if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        running_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds_list.extend(preds.cpu().numpy())
        all_labels_list.extend(current_labels.cpu().numpy())
        progress_bar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

    epoch_loss = running_loss / len(all_labels_list) # Use length of collected labels
    # Ensure all_labels_list is not empty before calculating metrics
    if not all_labels_list:
        return epoch_loss, 0.0, 0.0

    epoch_f1 = f1_score(all_labels_list, all_preds_list, average='weighted', zero_division=0)
    # For AUC, we need probabilities for the positive class.
    # This requires a re-run or storing probabilities if criterion is just CrossEntropy.
    # For simplicity in train loop, AUC based on argmax preds is a proxy.
    # True AUC calculated in evaluate_model.
    try:
        epoch_auc = roc_auc_score(all_labels_list, all_preds_list) if len(np.unique(all_labels_list)) > 1 else 0.0
    except ValueError: # Handle cases like only one class present in a batch or epoch
        epoch_auc = 0.0
    return epoch_loss, epoch_f1, epoch_auc


def evaluate_model(model, val_loader, criterion, device, optimal_threshold=0.5):
    model.eval()
    running_loss = 0.0
    all_probs_class1 = []
    all_labels_list = [] # Renamed

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Evaluating")
        for batch in progress_bar:
            inputs = batch['image'].to(device)
            current_labels = batch['label'].to(device) # Renamed

            outputs = model(inputs)
            loss = criterion(outputs, current_labels)
            running_loss += loss.item() * inputs.size(0)

            probs = F.softmax(outputs, dim=1)
            all_probs_class1.extend(probs[:, 1].cpu().numpy())
            all_labels_list.extend(current_labels.cpu().numpy())
            progress_bar.set_postfix(loss=loss.item())

    if not all_labels_list: # If val_loader was empty
        print("Warning: Validation data is empty. Returning zero for metrics.")
        return 0.0, 0.0, 0.0, 0.0

    epoch_loss = running_loss / len(all_labels_list)
    all_labels_np = np.array(all_labels_list)
    all_probs_class1_np = np.array(all_probs_class1)

    all_preds_thresholded = (all_probs_class1_np >= optimal_threshold).astype(int)

    if len(np.unique(all_labels_np)) < 2:
        print("Warning: Evaluation metrics (AUC, F1, Kappa) require at least two classes in validation.")
        epoch_auc, epoch_f1, epoch_kappa = 0.0, 0.0, 0.0
    else:
        try:
            epoch_auc = roc_auc_score(all_labels_np, all_probs_class1_np)
        except ValueError: # If only one class in true labels after all
            epoch_auc = 0.0
        epoch_f1 = f1_score(all_labels_np, all_preds_thresholded, average='weighted', zero_division=0)
        epoch_kappa = cohen_kappa_score(all_labels_np, all_preds_thresholded)

    print(f"Validation - Loss: {epoch_loss:.4f}, AUC: {epoch_auc:.4f}, F1 (thr {optimal_threshold:.2f}): {epoch_f1:.4f}, Kappa: {epoch_kappa:.4f}")
    return epoch_loss, epoch_auc, epoch_f1, epoch_kappa


# --- 主执行流程 ---
if __name__ == "__main__":
    # 1. 检查数据目录
    if not os.path.isdir(TRAIN_DIR) or not os.path.isdir(VAL_DIR):
        print(f"!!! ERROR: Train ({TRAIN_DIR}) or Validation ({VAL_DIR}) directory not found!")
        print(f"!!! Please ensure the 'DATA_DIR' is correct and contains 'train' and 'test' subdirectories.")
        exit()

    # 2. 创建 Dataset
    print("Creating datasets...")
    try:
        train_dataset = MedicalImageFolderDataset(TRAIN_DIR, transform=get_transforms('train'))
        val_dataset = MedicalImageFolderDataset(VAL_DIR, transform=get_transforms('valid'))
    except RuntimeError as e:
        print(f"Error creating dataset: {e}")
        exit()

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Error: Training or validation dataset is empty. Please check your data directories.")
        exit()

    # 获取训练集的类别数量 (确保 NUM_CLASSES 和实际类别数一致)
    # MedicalImageFolderDataset stores class_counts and classes
    actual_num_classes_train = len(train_dataset.classes)
    if NUM_CLASSES != actual_num_classes_train:
        print(f"Mismatch: Global NUM_CLASSES is {NUM_CLASSES}, but found {actual_num_classes_train} in TRAIN_DIR.")
        print(f"Please set NUM_CLASSES to {actual_num_classes_train} or verify your dataset structure.")
        NUM_CLASSES = actual_num_classes_train # Override for safety, or exit()
        print(f"NUM_CLASSES has been updated to {NUM_CLASSES} based on training data.")


    train_class_counts = train_dataset.class_counts
    print(f"Class counts in training data ({train_dataset.classes}): {train_class_counts}")


    # 3. 创建 DataLoader
    use_sampler = True # Set to False to disable TwoStreamSampler
    use_weighted_loss = True # Set to False to use standard CrossEntropyLoss

    use_sampler = True
    if use_sampler and NUM_CLASSES == 2 and np.all(train_class_counts > 0):
        minority_label_idx = np.argmin(train_class_counts)
        majority_label_idx = np.argmax(train_class_counts)
        print(f"Using TwoStreamSampler: Minority class {train_dataset.idx_to_class[minority_label_idx]} (idx {minority_label_idx}), Majority {train_dataset.idx_to_class[majority_label_idx]} (idx {majority_label_idx})")
        train_sampler = TwoStreamSampler(
        dataset_labels_array=train_dataset.labels,
        batch_size=BATCH_SIZE,
        minority_class_label=minority_label_idx,
        majority_class_label=majority_label_idx,
        minority_ratio=0.5
    )
        train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=0,
        pin_memory=True
    )
    else:
        if use_sampler and NUM_CLASSES != 2:
            print("TwoStreamSampler currently simplified for binary. Using RandomSampler.")
        if use_sampler and not np.all(train_class_counts > 0):
            print("One class has zero samples, cannot use TwoStreamSampler. Using RandomSampler.")
    print("Using standard RandomSampler for training.")
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
)
    # 4. 初始化模型、损失函数、优化器
    model = CustomMedicalCNN(num_classes=NUM_CLASSES).to(DEVICE) # NUM_CLASSES is now potentially updated
    print(model)

    if use_weighted_loss and np.all(train_class_counts > 0):
        criterion = DynamicWeightedLoss(train_class_counts, device=DEVICE, smoothing=0.9)
    else:
        if use_weighted_loss: print("Cannot use DynamicWeightedLoss if any class count is zero. Using standard CrossEntropyLoss.")
        print("Using standard CrossEntropyLoss.")
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=NUM_EPOCHS,
        pct_start=0.3, div_factor=25, final_div_factor=1e4
    )
    scaler = torch.cuda.amp.GradScaler() if DEVICE.type == 'cuda' else None
    if scaler: print("Using Mixed Precision Training.")


    # 5. 训练循环
    best_val_auc = 0.0
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'train_auc': [], 'val_auc': [], 'train_f1': [], 'val_f1': [], 'val_kappa': []}

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        train_loss, train_f1, train_auc_proxy = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, scaler, scheduler)
        history['train_loss'].append(train_loss)
        history['train_f1'].append(train_f1)
        history['train_auc'].append(train_auc_proxy) # Store proxy AUC
        print(f"Epoch {epoch+1} Train - Loss: {train_loss:.4f}, F1: {train_f1:.4f}, AUC (proxy): {train_auc_proxy:.4f}")

        current_optimal_threshold = 0.5 # Default for eval during training loop
        val_loss, val_auc, val_f1, val_kappa = evaluate_model(model, val_loader, criterion, DEVICE, current_optimal_threshold)
        history['val_loss'].append(val_loss); history['val_auc'].append(val_auc)
        history['val_f1'].append(val_f1); history['val_kappa'].append(val_kappa)

        if val_auc > best_val_auc:
            print(f"Validation AUC improved from {best_val_auc:.4f} to {val_auc:.4f}. Saving model to {MODEL_SAVE_PATH}")
            best_val_auc = val_auc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"Validation AUC did not improve. Current best: {best_val_auc:.4f}. No improvement for {epochs_no_improve} epochs.")

        if epochs_no_improve >= PATIENCE_EARLY_STOPPING:
            print(f"Early stopping triggered after {PATIENCE_EARLY_STOPPING} epochs with no improvement.")
            break
    print("\n--- Training Finished ---")

    # 6. 加载最佳模型并进行最终评估
    if os.path.exists(MODEL_SAVE_PATH):
        print("\nLoading best model for final evaluation...")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))

        print("Finding optimal threshold on validation set with the best model...")
        final_optimal_threshold = find_optimal_threshold(model, val_loader, DEVICE)

        print(f"\nEvaluating best model on validation set using optimal threshold: {final_optimal_threshold:.4f}")
        final_val_loss, final_val_auc, final_val_f1, final_val_kappa = evaluate_model(model, val_loader, criterion, DEVICE, optimal_threshold=final_optimal_threshold)
        print(f"Final Validation Results - Loss: {final_val_loss:.4f}, AUC: {final_val_auc:.4f}, F1: {final_val_f1:.4f}, Kappa: {final_val_kappa:.4f}")
    else:
        print(f"No best model saved at {MODEL_SAVE_PATH}. Skipping final evaluation with best model.")


    # 7. 绘制训练历史
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Over Epochs'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_auc'], label='Train AUC (proxy)')
    plt.plot(history['val_auc'], label='Val AUC')
    plt.plot(history['train_f1'], label='Train F1', linestyle='--')
    plt.plot(history['val_f1'], label='Val F1 (Thresh 0.5)', linestyle='--') # Note threshold used
    plt.title('AUC & F1 Over Epochs'); plt.xlabel('Epoch'); plt.ylabel('Metric'); plt.legend()
    plt.tight_layout()
    plt.savefig(TRAINING_HISTORY_IMG_PATH)
    print(f"Training history plot saved to {TRAINING_HISTORY_IMG_PATH}")
    plt.show()


    # --- 8. Grad-CAM 可视化 ---
    print("\n--- Generating Grad-CAM ---")
    if os.path.exists(MODEL_SAVE_PATH) and len(val_dataset) > 0:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH)) # Ensure best model is loaded
        sample_data = val_dataset[0]
        img_tensor = sample_data['image'].unsqueeze(0).to(DEVICE)
        original_pil_img_path = sample_data['path'] # Path is now stored in dataset item

        try:
            original_pil_img = Image.open(original_pil_img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            # Target layer for GradCAM: model.layer4 is a ResidualBlock instance
            grad_cam_handler = GradCAM(model, target_layer_name_or_module=model.layer4)
            cam_heatmap, predicted_class_idx = grad_cam_handler.generate_cam(img_tensor, retain_graph=False)

            actual_label_idx = sample_data['label'].item()
            actual_label_name = val_dataset.idx_to_class.get(actual_label_idx, "Unknown")
            predicted_class_name = val_dataset.idx_to_class.get(predicted_class_idx, "Unknown")

            print(f"Grad-CAM for sample: {os.path.basename(original_pil_img_path)}")
            print(f"Actual Label: {actual_label_name} (idx {actual_label_idx}), Predicted Class (for CAM): {predicted_class_name} (idx {predicted_class_idx})")
            show_cam_on_image(original_pil_img, cam_heatmap, save_path=GRAD_CAM_EXAMPLE_IMG_PATH)
        except Exception as e:
            print(f"Could not generate Grad-CAM: {e}")
            import traceback
            traceback.print_exc()
    elif not os.path.exists(MODEL_SAVE_PATH):
        print("Grad-CAM skipped: Best model not found.")
    else:
        print("Grad-CAM skipped: Validation dataset is empty.")

    # --- 9. 模型集成 (示例, 保持不变, 需要您自己训练多个模型) ---
    # model_paths_for_ensemble = [os.path.join(OUTPUT_BASE_DIR, "model_fold1.pth"), ...]

    print("\n--- Script Finished ---")