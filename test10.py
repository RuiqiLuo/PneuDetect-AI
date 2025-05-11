# -*- coding: utf-8 -*-
"""
Script for training an optimized custom CNN on Chest X-Ray data (Normal vs Pneumonia)
without pretrained models, with balanced sampling, Grad-CAM visualization, and
detailed evaluation metrics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Sampler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix
import seaborn as sns  # For confusion matrix visualization
import time
import os
import copy
import cv2  # For Grad-CAM visualization
import random
import datetime
import traceback  # Import traceback for detailed error printing
from collections import Counter  # For counting class occurrences

# Print current time and library versions
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"Script executed on: {current_time}")
print(f"PyTorch Version: {torch.__version__}")
print(f"Torchvision Version: {torchvision.__version__}")
print(f"OpenCV Version: {cv2.__version__}")
print("-" * 60)

# ==============================================================================
# 1. Configuration / Hyperparameters
# ==============================================================================
# --- Paths ---
DATA_DIR = 'D:\\archive\\chest_xray'  # User's specified input path

# Check if the data path exists early
if not os.path.isdir(DATA_DIR):
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"!!! ERROR: Dataset directory not found at '{DATA_DIR}'")
    print(f"!!! Please update the 'DATA_DIR' variable in the script.")
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    exit()

# Using the 'test' directory from the original dataset as our validation set
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'test')

# --- Output Directories ---
OUTPUT_BASE_DIR = r'C:\Users\LRQ\Desktop\output'  # User's specified output path

# Ensure output directories exist
try:
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    GRAD_CAM_DIR = os.path.join(OUTPUT_BASE_DIR, 'grad_cam_output')
    os.makedirs(GRAD_CAM_DIR, exist_ok=True)
    print(f"Output directory ensured: {OUTPUT_BASE_DIR}")
    print(f"Grad-CAM output directory ensured: {GRAD_CAM_DIR}")
except Exception as e:
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"!!! ERROR: Could not create output directories: {e}")
    print(f"!!! Please check permissions for '{OUTPUT_BASE_DIR}'.")
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    traceback.print_exc()
    exit()

MODEL_SAVE_PATH = os.path.join(OUTPUT_BASE_DIR, 'best_custom_cnn_optimized.pth')
HISTORY_PLOT_PATH = os.path.join(OUTPUT_BASE_DIR, 'training_history.png')
CONFUSION_MATRIX_PATH = os.path.join(OUTPUT_BASE_DIR, 'confusion_matrix.png')

# --- Training Settings ---
IMG_SIZE = 224       # Input image size for the model
BATCH_SIZE = 32      # Adjust based on GPU memory
NUM_EPOCHS = 20      # Number of training epochs
LEARNING_RATE = 0.0003 # Adjusted learning rate
NUM_GRAD_CAM_IMAGES = 4 # How many Grad-CAM images to save per epoch
RANDOM_SEED = 42     # For reproducible results
WEIGHT_DECAY = 5e-4  # Stronger L2 regularization

# --- Imbalance Handling ---
USE_WEIGHTED_LOSS = True

# --- Setup Reproducibility ---
if RANDOM_SEED is not None:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

# --- Device Selection ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    try:
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"Could not get CUDA device name: {e}")
print("-" * 60)

# ==============================================================================
# 2. Custom CNN Model Definition (Simplified)
# ==============================================================================
class CustomCNN(nn.Module):
    def __init__(self, num_classes=2, input_channels=3, img_size=224):
        super(CustomCNN, self).__init__()
        self.img_size = img_size

        # --- Convolutional Layers (Simplified) ---
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Dynamically Calculate Flattened Size ---
        self.flattened_size = self._get_flattened_size(input_channels, img_size)
        print(f"CustomCNN: Calculated flattened feature size = {self.flattened_size}")

        # --- Fully Connected Layers ---
        self.fc1 = nn.Linear(self.flattened_size, 128)  # Reduced size
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.7)  # Increased dropout
        self.fc2 = nn.Linear(128, num_classes)

    def _forward_conv_features(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        return x

    def _get_flattened_size(self, input_channels, img_size):
        try:
            with torch.no_grad():
                dummy_input = torch.zeros(1, input_channels, img_size, img_size)
                dummy_output = self._forward_conv_features(dummy_input)
                flattened_size = int(torch.flatten(dummy_output, 1).shape[1])
            return flattened_size
        except Exception as e:
            print(f"Error calculating flattened size: {e}")
            traceback.print_exc()
            raise RuntimeError("Could not calculate flattened size for CustomCNN.")

    def forward(self, x):
        x = self._forward_conv_features(x)
        x = torch.flatten(x, 1)
        x = self.relu_fc1(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ==============================================================================
# 3. Dataset Loading and Transforms
# ==============================================================================
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(IMG_SIZE + 16),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("\nLoading datasets...")
try:
    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(f"Train directory not found at '{TRAIN_DIR}'")
    if not os.path.isdir(VAL_DIR):
        raise FileNotFoundError(f"Val directory not found at '{VAL_DIR}'")

    image_datasets = {
        'train': datasets.ImageFolder(TRAIN_DIR, data_transforms['train']),
        'val': datasets.ImageFolder(VAL_DIR, data_transforms['val'])
    }

    # --- Balanced Batch Sampler for Training ---
    class BalancedBatchSampler(Sampler):
        def __init__(self, dataset, labels, batch_size):
            self.dataset = dataset
            self.labels = labels
            self.batch_size = batch_size
            self.num_classes = 2
            self.class_indices = [np.where(np.array(labels) == i)[0] for i in range(self.num_classes)]
            self.batch_per_class = batch_size // self.num_classes

        def __iter__(self):
            class_indices_shuffled = [indices.copy() for indices in self.class_indices]
            for indices in class_indices_shuffled:
                np.random.shuffle(indices)

            batches = []
            num_batches = min(len(self.class_indices[0]), len(self.class_indices[1])) // self.batch_per_class
            for _ in range(num_batches):
                batch = []
                for indices in class_indices_shuffled:
                    batch.extend(indices[:self.batch_per_class])
                    indices = indices[self.batch_per_class:]
                batches.append(batch)
            return iter(batches)

        def __len__(self):
            return len(self.dataset) // self.batch_size

    # Create balanced batch sampler for training
    train_labels = image_datasets['train'].targets
    balanced_batch_sampler = BalancedBatchSampler(image_datasets['train'], train_labels, BATCH_SIZE)

    # DataLoaders with balanced batch sampler for training
    dataloaders = {
        'train': DataLoader(
            image_datasets['train'],
            batch_sampler=balanced_batch_sampler,
            num_workers=min(4, os.cpu_count() // 2),
            pin_memory=True if torch.cuda.is_available() else False
        ),
        'val': DataLoader(
            image_datasets['val'],
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=min(4, os.cpu_count() // 2),
            pin_memory=True if torch.cuda.is_available() else False
        )
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)

    print(f"Classes found: {class_names} ({num_classes} classes)")
    print("Class distribution:")
    train_counts = Counter(train_labels)
    val_labels = image_datasets['val'].targets
    val_counts = Counter(val_labels)

    print(f" - Train:")
    for i, name in enumerate(class_names):
        print(f"     {name}: {train_counts.get(i, 0)}")
    print(f" - Val:")
    for i, name in enumerate(class_names):
        print(f"     {name}: {val_counts.get(i, 0)}")

    print(f"Total training images: {dataset_sizes['train']}")
    print(f"Total validation images: {dataset_sizes['val']}")

    # Recalculate class weights based on inverse frequency
    if USE_WEIGHTED_LOSS and num_classes > 1:
        total_train_samples = sum(train_counts.values())
        class_weights = [total_train_samples / max(1, train_counts.get(i, 0)) for i in range(num_classes)]
        # Adjust weights to emphasize minority class (NORMAL)
        class_weights = [w * 1.2 if i == 0 else w for i, w in enumerate(class_weights)]  # Boost NORMAL weight by 20%
        sum_weights = sum(class_weights)
        if sum_weights > 0:
            class_weights = [w / sum_weights * num_classes for w in class_weights]
        else:
            print("Warning: Could not calculate valid class weights. Using uniform weights.")
            class_weights = [1.0] * num_classes
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
        print(f"Calculated class weights: {class_weights} (tensor on {DEVICE})")
    else:
        class_weights_tensor = None
        print("Weighted loss is DISABLED.")

    print("-" * 60)

except FileNotFoundError as e:
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"!!! ERROR Loading Dataset: {e}")
    print(f"!!! Check if '{TRAIN_DIR}' and '{VAL_DIR}' exist and contain class subfolders.")
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    traceback.print_exc()
    exit()
except Exception as e:
    print(f"An unexpected error occurred loading the dataset: {e}")
    traceback.print_exc()
    exit()

# ==============================================================================
# 4. Grad-CAM Implementation
# ==============================================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self._forward_hook_handle = None
        self._backward_hook_handle = None
        self._register_hooks()

    def _register_hooks(self):
        try:
            if not isinstance(self.target_layer, nn.Module):
                print(f"Error: Target layer for Grad-CAM is not a PyTorch Module: {type(self.target_layer)}")
                return
            self._forward_hook_handle = self.target_layer.register_forward_hook(self._save_activations)
            self._backward_hook_handle = self.target_layer.register_full_backward_hook(self._save_gradients)
            print(f"    Grad-CAM hooks registered for layer: {self.target_layer}")
        except Exception as e:
            print(f"Error registering Grad-CAM hooks: {e}")
            traceback.print_exc()

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        if grad_output and isinstance(grad_output[0], torch.Tensor):
            self.gradients = grad_output[0].detach()
        else:
            print(f"    Warning: Did not receive expected gradient for hook: {grad_output}")
            self.gradients = None

    def generate(self, input_tensor, target_class_idx=None):
        if self._forward_hook_handle is None or self._backward_hook_handle is None:
            return None, None

        self.model.eval()
        self.gradients = None
        self.activations = None

        try:
            input_tensor = input_tensor.clone().detach().to(DEVICE).requires_grad_(True)
            with torch.enable_grad():
                output = self.model(input_tensor)
            if target_class_idx is None:
                target_class_idx = output.argmax(dim=1).item()
            score = output[:, target_class_idx]
            self.model.zero_grad()
            score.backward(retain_graph=False)

            if self.gradients is None or self.activations is None:
                print("    Error: Grad-CAM failed - hooks did not capture gradients or activations.")
                return None, target_class_idx

            pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
            activations_for_cam = self.activations[0]
            weighted_activations = activations_for_cam * pooled_gradients.unsqueeze(-1).unsqueeze(-1)
            heatmap = torch.sum(weighted_activations, dim=0).cpu().numpy()
            heatmap = np.maximum(heatmap, 0)
            max_heatmap_val = np.max(heatmap)
            if max_heatmap_val > 0:
                heatmap /= max_heatmap_val
            return heatmap, target_class_idx
        except Exception as e:
            print(f"    ❌ Error during Grad-CAM generation: {e}")
            traceback.print_exc()
            return None, target_class_idx

    def remove_hooks(self):
        """Remove registered forward and backward hooks"""
        if self._forward_hook_handle is not None:
            self._forward_hook_handle.remove()
            self._forward_hook_handle = None
        if self._backward_hook_handle is not None:
            self._backward_hook_handle.remove()
            self._backward_hook_handle = None

# ==============================================================================
# 5. Grad-CAM Visualization Helper Functions
# ==============================================================================
def denormalize_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    tensor = tensor.clone().cpu()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def tensor_to_cv2_img(tensor):
    try:
        img = denormalize_image(tensor)
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img_bgr
    except Exception as e:
        print(f"    Error in tensor_to_cv2_img: {e}")
        traceback.print_exc()
        return None

def overlay_grad_cam(img_bgr, heatmap, alpha=0.5):
    if heatmap is None or img_bgr is None:
        return None
    try:
        heatmap = heatmap.astype(np.float32)
        heatmap_resized = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_colored, alpha, 0)
        return superimposed_img
    except Exception as e:
        print(f"    Error during heatmap overlay: {e}")
        traceback.print_exc()
        return img_bgr

# ==============================================================================
# 6. Initialize Model, Grad-CAM, Loss, Optimizer, Scheduler
# ==============================================================================
print("\nInitializing Model, Loss, Optimizer...")
model = CustomCNN(num_classes=num_classes, img_size=IMG_SIZE).to(DEVICE)

# --- Initialize Grad-CAM ---
grad_cam_generator = None
try:
    target_layer_name = 'conv3'  # Use the last convolutional layer
    target_layer_cam = getattr(model, target_layer_name, None)
    if target_layer_cam is None or not isinstance(target_layer_cam, nn.Module):
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"!!! WARNING: Could not find valid target layer '{target_layer_name}' in CustomCNN.")
        print(f"!!! Grad-CAM generation will be disabled.")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        grad_cam_generator = GradCAM(model, target_layer_cam)
except Exception as e:
    print(f"An unexpected error occurred during Grad-CAM initialization: {e}")
    traceback.print_exc()

# Initialize criterion with weights
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, threshold=0.001)

print("-" * 60)

# ==============================================================================
# 7. Training Loop Function
# ==============================================================================
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    start_time = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_auc': []}

    # Determine positive class index for AUC
    try:
        if 'PNEUMONIA' in class_names:
            positive_class_idx = class_names.index('PNEUMONIA')
            print(f"Positive class for AUC calculation: 'PNEUMONIA' (index {positive_class_idx})")
        else:
            positive_class_idx = -1
            print("Warning: 'PNEUMONIA' not found in class names. AUC calculation will be disabled.")
    except Exception as e:
        print(f"Error determining positive class index: {e}")
        positive_class_idx = -1

    print("\n🚀 Starting Training...")
    print("-" * 60)

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 15)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            all_preds_indices = []
            all_labels_indices = []
            all_positive_probs = []

            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                try:
                    inputs = inputs.to(DEVICE)
                    labels = labels.to(DEVICE)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds_indices = torch.max(outputs, 1)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    all_preds_indices.extend(preds_indices.cpu().numpy())
                    all_labels_indices.extend(labels.cpu().numpy())

                    if phase == 'val' and positive_class_idx != -1:
                        probs = torch.softmax(outputs, dim=1)
                        if positive_class_idx < probs.shape[1]:
                            positive_probs = probs[:, positive_class_idx].detach().cpu().numpy()
                            all_positive_probs.extend(positive_probs)
                except Exception as e:
                    print(f"    ❌ Error processing batch {i}/{len(dataloaders[phase])} in {phase} phase: {e}")
                    traceback.print_exc()
                    continue

            epoch_loss = running_loss / dataset_sizes[phase] if dataset_sizes[phase] > 0 else 0.0
            epoch_acc = accuracy_score(all_labels_indices, all_preds_indices) if len(all_labels_indices) > 0 else 0.0

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc)

            print(f'  {phase.capitalize()} Results -> Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')

            if phase == 'val':
                # Print classification report
                print(f"  Validation Classification Report:\n{classification_report(all_labels_indices, all_preds_indices, target_names=class_names)}")

                # Compute and save confusion matrix
                cm = confusion_matrix(all_labels_indices, all_preds_indices)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
                plt.title(f'Confusion Matrix at Epoch {epoch}')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.savefig(os.path.join(OUTPUT_BASE_DIR, f'confusion_matrix_epoch_{epoch}.png'))
                plt.close()

                epoch_auc = 0.0
                if positive_class_idx != -1 and len(np.unique(all_labels_indices)) > 1 and len(all_positive_probs) > 0:
                    try:
                        epoch_auc = roc_auc_score(all_labels_indices, all_positive_probs)
                        history['val_auc'].append(epoch_auc)
                        print(f'  Validation AUC: {epoch_auc:.4f} (Best Acc: {best_acc:.4f})')
                    except ValueError as e_auc:
                        print(f"  Warning: Could not calculate AUC for epoch {epoch} ({e_auc}). Setting to 0.")
                        history['val_auc'].append(0.0)
                else:
                    history['val_auc'].append(0.0)

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    try:
                        torch.save(model.state_dict(), MODEL_SAVE_PATH)
                        print(f'      Best model weights saved to {MODEL_SAVE_PATH}')
                    except Exception as e_save:
                        print(f"    ❌ Error saving best model weights: {e_save}")
                        traceback.print_exc()

                if history['val_acc'] and len(history['val_acc']) == epoch + 1:
                    scheduler.step(history['val_acc'][-1])

        # Generate Grad-CAM examples
        if grad_cam_generator is not None and NUM_GRAD_CAM_IMAGES > 0:
            try:
                generate_and_save_grad_cam_epoch(epoch, num_images=NUM_GRAD_CAM_IMAGES)
            except Exception as e_cam_epoch:
                print(f"    ❌ Error during Grad-CAM generation for epoch {epoch}: {e_cam_epoch}")
                traceback.print_exc()

        epoch_time_elapsed = time.time() - epoch_start_time
        print(f'Epoch {epoch} completed in {epoch_time_elapsed // 60:.0f}m {epoch_time_elapsed % 60:.1f}s')
        print('-' * 15)

    time_elapsed = time.time() - start_time
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.1f}s')
    print(f'Best Validation Accuracy achieved: {best_acc:.4f}')
    print(f"Best model weights saved to: {MODEL_SAVE_PATH}")

    try:
        if os.path.exists(MODEL_SAVE_PATH):
            model.load_state_dict(best_model_wts)
            print("Loaded best model weights.")
    except Exception as e_load_best:
        print(f"    ❌ Error loading best model weights back: {e_load_best}")
        traceback.print_exc()

    return model, history

# ==============================================================================
# 8. Grad-CAM Generation Function
# ==============================================================================
def generate_and_save_grad_cam_epoch(epoch, num_images=NUM_GRAD_CAM_IMAGES):
    if grad_cam_generator is None:
        return

    print(f"\n🔍 Generating Grad-CAM examples for epoch {epoch}...")
    model.eval()

    count = 0
    temp_val_loader = DataLoader(image_datasets['val'], batch_size=1, shuffle=False, num_workers=0)
    val_loader_iter = iter(temp_val_loader)

    while count < num_images:
        try:
            batch = next(val_loader_iter)
            inputs, labels = batch
            img_tensor = inputs
            true_label_idx = labels[0].item()
            true_label_name = class_names[true_label_idx]

            heatmap, predicted_class_idx = grad_cam_generator.generate(img_tensor)
            if heatmap is not None and predicted_class_idx is not None:
                predicted_label_name = class_names[predicted_class_idx]
                img_bgr = tensor_to_cv2_img(img_tensor.squeeze(0))
                overlayed_image = overlay_grad_cam(img_bgr, heatmap)
                if overlayed_image is not None:
                    save_filename = f"epoch_{epoch:02d}_img_{count}_true_{true_label_name}_pred_{predicted_label_name}.png"
                    save_path = os.path.join(GRAD_CAM_DIR, save_filename)
                    cv2.imwrite(save_path, overlayed_image)
                    count += 1
        except StopIteration:
            print("    Warning: Reached end of validation set for Grad-CAM examples.")
            break
        except Exception as e:
            print(f"    ❌ Error processing image for Grad-CAM {count}: {e}")
            traceback.print_exc()
            count += 1

    if count >= num_images:
        print(f"✅ Successfully generated and saved {num_images} Grad-CAM images.")
    elif count > 0:
        print(f"⚠️ Finished Grad-CAM attempt, saved {count} images (requested {num_images}).")
    else:
        print(f"❌ Grad-CAM generation/save failed for all attempted images this epoch.")

# ==============================================================================
# 9. Plotting Function
# ==============================================================================
def plot_training_history(history, save_path=HISTORY_PLOT_PATH):
    try:
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            print(f"Plot save directory created: {save_dir}")

        epochs = range(len(history['train_loss']))

        plt.figure(figsize=(18, 6))

        # Plot Loss
        plt.subplot(1, 3, 1)
        plt.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
        plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Plot Accuracy
        plt.subplot(1, 3, 2)
        plt.plot(epochs, history['train_acc'], 'bo-', label='Training Accuracy')
        plt.plot(epochs, history['val_acc'], 'ro-', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        # Plot Validation AUC
        if history.get('val_auc') and len(history['val_auc']) == len(epochs):
            plt.subplot(1, 3, 3)
            plt.plot(epochs, history['val_auc'], 'go-', label='Validation AUC')
            plt.title('Validation AUC')
            plt.xlabel('Epoch')
            plt.ylabel('AUC')
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.savefig(save_path)
        print(f"\nTraining history plot saved to: {save_path}")
        plt.close()
    except Exception as e:
        print(f"Error plotting training history: {e}")
        traceback.print_exc()

# ==============================================================================
# 10. Main Execution Block
# ==============================================================================
if __name__ == '__main__':
    if not os.path.isdir(DATA_DIR) or not os.path.isdir(TRAIN_DIR) or not os.path.isdir(VAL_DIR):
        print("\nData directories are missing. Please check the DATA_DIR configuration.")
        exit()
    elif not os.path.isdir(OUTPUT_BASE_DIR):
        print("\nOutput base directory could not be created. Please check permissions.")
        exit()
    else:
        try:
            trained_model, training_history = train_model(
                model,
                criterion,
                optimizer,
                scheduler,
                num_epochs=NUM_EPOCHS
            )
            print("\n✅ Training script finished successfully.")
            plot_training_history(training_history, save_path=HISTORY_PLOT_PATH)
        except Exception as main_e:
            print(f"\n❌ An unexpected error occurred during the main training process: {main_e}")
            traceback.print_exc()
        finally:
            if 'grad_cam_generator' in globals() and grad_cam_generator:
                print("\nAttempting to remove Grad-CAM hooks...")
                try:
                    grad_cam_generator.remove_hooks()
                    print("Grad-CAM hooks removed.")
                except Exception as e_cleanup:
                    print(f"Error during Grad-CAM hook cleanup: {e_cleanup}")
                    traceback.print_exc()