# -*- coding: utf-8 -*-
"""
Complete script for training a custom CNN on Chest X-Ray data (Normal vs Pneumonia)
with corrected Grad-CAM implementation, improved error handling/imbalance handling,
and configurable output directory creation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix
import time
import os
import copy
import cv2 # For Grad-CAM visualization
import random
import datetime
import traceback # Import traceback for detailed error printing
from collections import Counter # For counting class occurrences

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
# --- Paths (!!! MUST ADJUST THIS PATH !!!) ---
# Define the main directory containing 'train' and 'test' subfolders
DATA_DIR = 'D:\\archive\\chest_xray' # <--- *** CHANGE THIS PATH ***

# Check if the data path exists early
if not os.path.isdir(DATA_DIR):
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"!!! ERROR: Dataset directory not found at '{DATA_DIR}'")
    print(f"!!! Please update the 'DATA_DIR' variable in the script.")
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # Decide how to handle: Exit or continue with a placeholder (will likely fail)
    # To exit gracefully if data is missing:
    # import sys
    # sys.exit("Dataset directory not found. Exiting.")
    # If you want the script to stop here if data is missing, uncomment the two lines above.
    # Otherwise, it will continue, but likely fail later during data loading.
    pass # Continue, but expect errors later


# Using the 'test' directory from the original dataset as our validation set
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'test')


# --- Output Directories ---
# 修改这一行，将输出目录指向你的桌面文件夹
OUTPUT_BASE_DIR = r'C:\Users\LRQ\Desktop\output' # 使用 r'' 确保路径被正确解析

# 确保输出目录及其子目录存在，如果不存在则创建
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
    # Decide how to handle: Exit or continue (will likely fail on saving)
    # import sys
    # sys.exit("Could not create output directory. Exiting.")
    pass # Continue, but expect errors later

MODEL_SAVE_PATH = os.path.join(OUTPUT_BASE_DIR, 'best_custom_cnn_classifier.pth')
HISTORY_PLOT_PATH = os.path.join(OUTPUT_BASE_DIR, 'training_history.png')


# --- Training Settings ---
IMG_SIZE = 224       # Input image size for the model
BATCH_SIZE = 32      # Adjust based on GPU memory (e.g., 16, 32, 64)
NUM_EPOCHS = 20      # Number of training epochs (start with ~15-25, adjust)
LEARNING_RATE = 0.001 # Initial learning rate for Adam
NUM_GRAD_CAM_IMAGES = 4 # How many Grad-CAM images to save per epoch
RANDOM_SEED = 42     # For reproducible results (optional)

# --- Imbalance Handling ---
# Set to True to use weighted Cross-Entropy Loss based on class distribution
USE_WEIGHTED_LOSS = True

# --- Setup Reproducibility (Optional) ---
if RANDOM_SEED is not None:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
        # Might impact performance, but improves reproducibility
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

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
# 2. Custom CNN Model Definition
# ==============================================================================
class CustomCNN(nn.Module):
    def __init__(self, num_classes=2, input_channels=3, img_size=224):
        """
        Custom CNN model.

        Args:
            num_classes (int): Number of output classes.
            input_channels (int): Number of input image channels (3 for RGB).
            img_size (int): The height/width of the input image (needed for flatten calculation).
        """
        super(CustomCNN, self).__init__()
        self.img_size = img_size

        # --- Convolutional Layers ---
        # Block 1
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=5, stride=1, padding=2) # (N, 32, 224, 224)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # (N, 32, 112, 112)

        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # (N, 64, 112, 112)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # (N, 64, 56, 56)

        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) # (N, 128, 56, 56)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # (N, 128, 28, 28)

        # Block 4 (Target layer for Grad-CAM is often here)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1) # (N, 256, 28, 28)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # (N, 256, 14, 14)

        # --- Dynamically Calculate Flattened Size ---
        self.flattened_size = self._get_flattened_size(input_channels, img_size)
        print(f"CustomCNN: Calculated flattened feature size = {self.flattened_size}")


        # --- Fully Connected Layers ---
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5) # Dropout for regularization
        self.fc2 = nn.Linear(512, num_classes) # Output layer

    def _forward_conv_features(self, x):
        """ Helper to pass input through conv layers (used for size calculation). """
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        return x

    def _get_flattened_size(self, input_channels, img_size):
        """ Calculate the flattened size after conv layers dynamically. """
        try:
            with torch.no_grad():
                # Create a dummy input matching the expected batch input shape
                dummy_input = torch.zeros(1, input_channels, img_size, img_size)
                dummy_output = self._forward_conv_features(dummy_input)
                flattened_size = int(torch.flatten(dummy_output, 1).shape[1])
            return flattened_size
        except Exception as e:
            print(f"Error calculating flattened size: {e}")
            traceback.print_exc()
            # Return a placeholder or raise error
            raise RuntimeError("Could not calculate flattened size for CustomCNN.")


    def forward(self, x):
        """ Defines the forward pass. """
        # Pass through convolutional blocks
        x = self._forward_conv_features(x)
        # Flatten the features
        x = torch.flatten(x, 1) # Flatten all dimensions except batch
        # Pass through fully connected layers
        x = self.relu_fc1(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) # Final logits
        return x

# ==============================================================================
# 3. Dataset Loading and Transforms
# ==============================================================================
# Define transformations with augmentation for training, minimal for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet stats
    ]),
    'val': transforms.Compose([
        transforms.Resize(IMG_SIZE + 16), # Resize slightly larger for center crop
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
    dataloaders = {
        # Consider adjusting num_workers based on your system
        # Setting num_workers=0 is good for debugging errors related to data loading
        x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=(x == 'train'), num_workers=min(4, os.cpu_count()//2), pin_memory=True if torch.cuda.is_available() else False)
        for x in ['train', 'val']
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)

    print(f"Classes found: {class_names} ({num_classes} classes)")
    # Display class distribution (important for imbalanced datasets)
    print("Class distribution:")
    train_labels = image_datasets['train'].targets
    val_labels = image_datasets['val'].targets

    train_counts = Counter(train_labels)
    val_counts = Counter(val_labels)

    print(f" - Train:")
    for i, name in enumerate(class_names):
        print(f"     {name}: {train_counts.get(i, 0)}")
    print(f" - Val:")
    for i, name in enumerate(class_names):
        print(f"     {name}: {val_counts.get(i, 0)}")


    print(f"Total training images: {dataset_sizes['train']}")
    print(f"Total validation images: {dataset_sizes['val']}")

    # Calculate class weights for imbalanced dataset
    if USE_WEIGHTED_LOSS and num_classes > 1:
        # Calculate inverse class frequency weights: 1 / count(class)
        # Then normalize them (optional, but good practice)
        total_train_samples = sum(train_counts.values())
        # Handle case where a class might have 0 samples in train (though unlikely with ImageFolder on this data)
        class_weights = [total_train_samples / max(1, train_counts.get(i, 0)) for i in range(num_classes)] # Use max(1, count) to avoid div by zero

        # Normalize weights (optional, sum to num_classes or max to 1)
        # Normalizing to sum to num_classes is a common approach for CrossEntropyLoss
        sum_weights = sum(class_weights)
        if sum_weights > 0:
            class_weights = [w / sum_weights * num_classes for w in class_weights] # Normalize to sum to num_classes
        else:
            print("Warning: Could not calculate valid class weights. Using uniform weights.")
            class_weights = [1.0] * num_classes


        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
        print(f"Calculated class weights: {class_weights} (tensor on {DEVICE})")
    elif USE_WEIGHTED_LOSS and num_classes <= 1:
        print("Warning: Weighted loss requested, but only one or zero classes found. Weighted loss is DISABLED.")
        class_weights_tensor = None
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
# 4. Grad-CAM Implementation (Should work with any PyTorch model)
# ==============================================================================
class GradCAM:
    """ Class for generating Grad-CAM heatmaps. """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._forward_hook_handle = None
        self._backward_hook_handle = None

        # Register hooks safely
        self._register_hooks()

    def _register_hooks(self,):
        """ Register forward and backward hooks to the target layer. """
        try:
            # Check if target_layer is a nn.Module
            if not isinstance(self.target_layer, nn.Module):
                print(f"Error: Target layer for Grad-CAM is not a PyTorch Module: {type(self.target_layer)}")
                return

            self._forward_hook_handle = self.target_layer.register_forward_hook(self._save_activations)
            # Use register_full_backward_hook for potentially non-leaf tensors
            # https://pytorch.org/docs/stable/generated/torch.nn.Module.register_full_backward_hook.html
            self._backward_hook_handle = self.target_layer.register_full_backward_hook(self._save_gradients)
            print(f"    Grad-CAM hooks registered for layer: {self.target_layer}")

        except Exception as e:
            print(f"Error registering Grad-CAM hooks: {e}")
            traceback.print_exc()
            self._forward_hook_handle = None
            self._backward_hook_handle = None


    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        # grad_output is a tuple containing the gradient for the output tensor
        # Make sure the gradient is for the primary output of the module
        if grad_output and isinstance(grad_output[0], torch.Tensor):
            self.gradients = grad_output[0].detach()
        else:
            # Handle cases where grad_output might be empty or not a tensor
            print(f"    Warning: Did not receive expected gradient for hook: {grad_output}")
            self.gradients = None


    def generate(self, input_tensor, target_class_idx=None):
        """ Generates Grad-CAM heatmap. Returns (heatmap, predicted_class_idx) or (None, predicted_class_idx). """
        # Check if hooks were successfully registered
        if self._forward_hook_handle is None or self._backward_hook_handle is None:
            # print("    Error: Grad-CAM hooks not registered properly. Skipping generation.") # Suppress this frequent print
            return None, None

        self.model.eval() # Ensure model is in evaluation mode
        self.gradients = None # Reset gradients and activations
        self.activations = None

        try:
            # Ensure input tensor requires grad for the backward pass
            # Using .clone().detach().requires_grad_(True) is a robust way to create a leaf tensor with grad enabled
            input_tensor = input_tensor.clone().detach().to(DEVICE).requires_grad_(True)

            # Forward pass
            # Need to ensure requires_grad is true during the forward pass through the target layer
            # The outer torch.no_grad() context should NOT be active during CAM generation
            with torch.enable_grad(): # Explicitly enable grad just in case
                 output = self.model(input_tensor) # Logits [1, num_classes]


            if target_class_idx is None:
                # Get the model's prediction
                target_class_idx = output.argmax(dim=1).item()
            else:
                # Ensure the provided target_class_idx is valid
                if target_class_idx < 0 or target_class_idx >= output.shape[1]:
                    print(f"    Error: Provided target_class_idx ({target_class_idx}) is out of bounds for output shape {output.shape}.")
                    return None, None # Cannot proceed with invalid index


            # Select the score for the target class
            # Ensure index is valid for slicing
            if target_class_idx < 0 or target_class_idx >= output.shape[1]:
                 print(f"    Error: Target class index {target_class_idx} is out of bounds for output tensor.")
                 return None, None
            score = output[:, target_class_idx]


            # Backward pass
            self.model.zero_grad() # Clear existing gradients
            # Calculate gradients of the target score with respect to the input to the model
            # This implicitly causes gradients to be computed w.r.t intermediate tensors like target_layer output
            # Gradients are captured by the hook on the target_layer w.r.t *its* output
            try:
                 score.backward(retain_graph=False)


            except RuntimeError as e:
                print(f"    Error during Grad-CAM backward pass: {e}. Ensure requires_grad was enabled.")
                traceback.print_exc()
                return None, target_class_idx # Return current pred index even if CAM failed


            # Check if hooks captured data
            if self.gradients is None or self.activations is None:
                print("    Error: Grad-CAM failed - hooks did not capture gradients or activations. Check layer type or model flow.")
                return None, target_class_idx


            # Pool gradients and compute CAM
            # The gradients captured by the hook are w.r.t the target layer's *output*.
            # They should have the same shape as the target layer's output activations.
            # Gradients: [N, C, H', W'], Activations: [N, C, H', W']
            if self.gradients.ndim == 4 and self.activations.ndim == 4:
                # Average gradients across batch, height, and width
                pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3]) # Result shape [C]
                # Get activations for the first image in the batch (assuming batch size is 1)
                activations_for_cam = self.activations[0] # Result shape [C, H', W']
            else:
                print(f"    Error: Gradients shape {self.gradients.shape} or activations shape {self.activations.shape} not as expected for pooling (should be 4D).")
                return None, target_class_idx


            # Weight channels by gradients
            # Multiply each activation map by the corresponding gradient weight
            if pooled_gradients.shape[0] == activations_for_cam.shape[0]: # Check if channel counts match
                # Reshape pooled_gradients from [C] to [C, 1, 1] for broadcasting
                pooled_gradients = pooled_gradients.unsqueeze(-1).unsqueeze(-1)
                # Element-wise multiplication: [C, H', W'] * [C, 1, 1] -> [C, H', W']
                weighted_activations = activations_for_cam * pooled_gradients

                # Sum weighted activations across channels to get the heatmap
                heatmap = torch.sum(weighted_activations, dim=0).cpu().numpy() # Result shape [H', W']
            else:
                print(f"    Error: Gradient channels ({pooled_gradients.shape[0]}) and activation channels ({activations_for_cam.shape[0]}) do not match for weighting.")
                return None, target_class_idx


            heatmap = np.maximum(heatmap, 0) # Apply ReLU to the heatmap

            # Normalize the heatmap to [0, 1]
            max_heatmap_val = np.max(heatmap)
            if max_heatmap_val > 0:
                heatmap /= max_heatmap_val
            else:
                # If max is 0, heatmap is all zeros, no need to normalize
                heatmap = np.zeros_like(heatmap)
                # print("    Warning: Grad-CAM heatmap is all zeros after ReLU.") # Suppress frequent warning

            return heatmap, target_class_idx

        except Exception as e:
            print(f"    ❌ Error during Grad-CAM generation: {e}")
            traceback.print_exc()
            return None, target_class_idx


# ==============================================================================
# 5. Grad-CAM Visualization Helper Functions
# ==============================================================================
def denormalize_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """ Denormalizes a tensor image. """
    tensor = tensor.clone().cpu() # Ensure on CPU for numpy conversion, clone to not modify original
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m) # Multiply by std and add mean
    return tensor

def tensor_to_cv2_img(tensor):
    """ Converts a PyTorch tensor (C, H, W) to an OpenCV image (H, W, C) in BGR format. """
    try:
        img = denormalize_image(tensor)
        img = img.permute(1, 2, 0) # Convert from C, H, W to H, W, C
        img = img.numpy()
        img = np.clip(img * 255, 0, 255).astype(np.uint8) # Convert to 0-255 range and clip
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Convert RGB to BGR for OpenCV
        return img_bgr
    except Exception as e:
        print(f"    Error in tensor_to_cv2_img: {e}")
        traceback.print_exc()
        return None

def overlay_grad_cam(img_bgr, heatmap, alpha=0.5):
    """ Overlays the heatmap on the image. """
    if heatmap is None or img_bgr is None:
        # print("    Skipping overlay: Heatmap or image is None.") # Suppress frequent print
        return None
    try:
        # Ensure heatmap is float32 for cv2.resize
        heatmap = heatmap.astype(np.float32)
        heatmap_resized = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        # Ensure img_bgr and heatmap_colored have compatible types and channels
        if img_bgr.shape[:2] != heatmap_colored.shape[:2] or img_bgr.shape[2] != heatmap_colored.shape[2]:
            print(f"    Error: Image shape {img_bgr.shape} and heatmap shape {heatmap_colored.shape} mismatch for overlay.")
            return img_bgr # Return original image on error
        superimposed_img = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_colored, alpha, 0)
        return superimposed_img
    except Exception as e:
        print(f"    Error during heatmap overlay: {e}")
        traceback.print_exc()
        return img_bgr # Return original image on error


# ==============================================================================
# 6. Initialize Model, Grad-CAM, Loss, Optimizer, Scheduler
# ==============================================================================
print("\nInitializing Model, Loss, Optimizer...")
model = CustomCNN(num_classes=num_classes, img_size=IMG_SIZE).to(DEVICE)

# --- Initialize Grad-CAM ---
grad_cam_generator = None # Initialize as None
try:
    # --- !!! CHOOSE THE TARGET LAYER FROM YOUR CustomCNN !!! ---
    # Example: Targeting the last convolutional layer 'conv4'
    target_layer_name = 'conv4'
    # Use getattr and check if it's a module
    target_layer_cam = getattr(model, target_layer_name, None)

    if target_layer_cam is None:
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"!!! WARNING: Could not find attribute '{target_layer_name}' in CustomCNN.")
        print(f"!!! Grad-CAM generation will be disabled.")
        print(f"!!! Check the layer name in the CustomCNN class definition.")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    elif not isinstance(target_layer_cam, nn.Module):
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"!!! WARNING: Attribute '{target_layer_name}' is not a PyTorch Module ({type(target_layer_cam)}).")
        print(f"!!! Grad-CAM generation will be disabled.")
        print(f"!!! Grad-CAM requires targeting a nn.Module layer (e.g., Conv2d, ReLU, BatchNorm2d).")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        grad_cam_generator = GradCAM(model, target_layer_cam)
        # Grad-CAM hooks registration printed inside GradCAM.__init__

except Exception as e:
    print(f"An unexpected error occurred during Grad-CAM initialization: {e}")
    traceback.print_exc()
    grad_cam_generator = None # Ensure it's None if initialization fails


# Initialize criterion with weights if USE_WEIGHTED_LOSS is True
# class_weights_tensor is calculated in the dataset loading section
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor) # Pass weights here
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
# Removed verbose=True from scheduler to fix warning
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=3, threshold=0.001) # Step based on Val AUC

print("-" * 60)


# ==============================================================================
# 7. Training Loop Function
# ==============================================================================
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    start_time = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_auc = 0.0 # Track best validation AUC

    # Store history for plotting later
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_auc': []}

    # Determine the index of the 'PNEUMONIA' class (assuming it's the positive class)
    try:
        # Ensure class_names is available and has 'PNEUMONIA'
        if 'PNEUMONIA' in class_names:
            positive_class_idx = class_names.index('PNEUMONIA')
            print(f"Positive class for AUC calculation: 'PNEUMONIA' (index {positive_class_idx})")
        elif len(class_names) == 2: # If not found but binary, assume the second class is positive
            # This assumes the second class alphabetically is 'PNEUMONIA', which might not always be true
            # if class_names = ['NORMAL', 'PNEUMONIA'], index 1 is PNEUMONIA.
            # If class_names = ['PNEUMONIA', 'NORMAL'], index 0 is PNEUMONIA.
            # The check 'PNEUMONIA' in class_names is safer.
            # Fallback to index 1 only if binary and 'PNEUMONIA' name isn't found.
            if class_names[1] == 'PNEUMONIA': # Double check if the second class is actually PNEUMONIA
                 positive_class_idx = 1
                 print(f"Warning: 'PNEUMONIA' class name not found by explicit search, but index 1 maps to '{class_names[1]}'. Assuming index 1 is positive class for AUC.")
            else:
                 # If binary but index 1 isn't PNEUMONIA, something is unexpected
                 positive_class_idx = -1
                 print(f"Warning: 'PNEUMONIA' class name not found, and index 1 is '{class_names[1]}'. Cannot reliably determine positive class for AUC.")

        else: # More than 2 classes, cannot determine positive class reliably
            positive_class_idx = -1 # Invalid index
            print("Warning: Cannot determine positive class for AUC calculation (not binary or 'PNEUMONIA' not found). AUC will not be calculated.")

    except Exception as e:
        print(f"Error determining positive class index for AUC: {e}")
        traceback.print_exc()
        positive_class_idx = -1 # Disable AUC if determination fails


    print("\n🚀 Starting Training...")
    print("-" * 60)

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 15)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            all_preds_indices = []
            all_labels_indices = []
            all_positive_probs = [] # For AUC

            # Iterate over data.
            num_batches = len(dataloaders[phase])
            if num_batches == 0:
                 print(f"Warning: No data for {phase} phase. Skipping.")
                 # Append dummy data or handle skipped phase gracefully in history
                 history[f'{phase}_loss'].append(0.0)
                 history[f'{phase}_acc'].append(0.0)
                 if phase == 'val':
                     history['val_auc'].append(0.0)
                 continue # Skip to next phase/epoch


            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                try: # Add try-except around batch processing
                    inputs = inputs.to(DEVICE)
                    labels = labels.to(DEVICE)

                    optimizer.zero_grad()

                    # Forward pass
                    # Enable gradients only during the training phase
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs) # Logits
                        loss = criterion(outputs, labels)
                        _, preds_indices = torch.max(outputs, 1) # Predicted class indices

                        # Backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # Statistics
                    running_loss += loss.item() * inputs.size(0)
                    all_preds_indices.extend(preds_indices.cpu().numpy())
                    all_labels_indices.extend(labels.cpu().numpy())

                    # Store probabilities for validation AUC
                    if phase == 'val' and positive_class_idx != -1: # Only calculate if positive class index is valid
                        probs = torch.softmax(outputs, dim=1)
                        # Ensure the positive_class_idx is within the bounds of the outputs/probs tensor
                        if positive_class_idx < probs.shape[1]:
                            positive_probs = probs[:, positive_class_idx].detach().cpu().numpy()
                            all_positive_probs.extend(positive_probs)
                        else:
                            print(f"    Warning: Positive class index {positive_class_idx} out of bounds for output tensor shape {probs.shape}. Skipping AUC for this batch.")


                    # Print batch progress for training phase
                    if phase == 'train' and (i + 1) % max(1, num_batches // 4) == 0: # Print ~4 times per epoch, handle num_batches < 4
                        print(f'  [{time.strftime("%H:%M:%S")}] Train Batch {i + 1}/{num_batches} | Loss: {loss.item():.4f}')

                except Exception as e:
                    print(f"    ❌ Error processing batch {i}/{num_batches} in {phase} phase: {e}")
                    traceback.print_exc()
                    # Decide how to handle: skip batch or raise error. Skipping allows training to continue.
                    continue # Skip this batch and continue with the next one


            # --- Epoch Statistics ---
            # Handle potential division by zero if dataset_sizes[phase] is 0 or batches were skipped
            epoch_loss = running_loss / dataset_sizes[phase] if dataset_sizes[phase] > 0 else 0.0
            epoch_acc = accuracy_score(all_labels_indices, all_preds_indices) if len(all_labels_indices) > 0 else 0.0

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc)

            print(f'  {phase.capitalize()} Results -> Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')

            # Validation phase specific actions
            if phase == 'val':
                epoch_auc = 0.0 # Default AUC
                # Calculate AUC if possible
                try:
                    # Need at least 2 unique classes and some positive probabilities calculated
                    if positive_class_idx != -1 and len(np.unique(all_labels_indices)) > 1 and len(all_positive_probs) > 0:
                        # Ensure probabilities and labels lists have the same length
                        if len(all_positive_probs) == len(all_labels_indices):
                            epoch_auc = roc_auc_score(all_labels_indices, all_positive_probs)
                            history['val_auc'].append(epoch_auc)
                            print(f'  Validation AUC: {epoch_auc:.4f} (Best AUC: {best_auc:.4f})')
                        else:
                            print(f"  Warning: Mismatch in lengths for AUC calculation (labels: {len(all_labels_indices)}, probs: {len(all_positive_probs)}). Skipping AUC for epoch {epoch}.")
                            history['val_auc'].append(0.0) # Append a placeholder
                    elif positive_class_idx == -1:
                        print("  Skipping AUC calculation: Positive class index invalid.")
                        history['val_auc'].append(0.0) # Append a placeholder
                    else:
                        print("  Skipping AUC calculation: Not enough classes or no positive samples with probability collected.")
                        history['val_auc'].append(0.0) # Append a placeholder


                except ValueError as e_auc:
                    # This can happen if there's only one class present in the labels/predictions for the epoch
                    print(f"  Warning: Could not calculate AUC for epoch {epoch} ({e_auc}). Setting to 0.")
                    traceback.print_exc()
                    history['val_auc'].append(0.0) # AUC is 0 or undefined
                except Exception as e_auc_general:
                    print(f"  Error calculating AUC for epoch {epoch}: {e_auc_general}")
                    traceback.print_exc()
                    history['val_auc'].append(0.0) # AUC is 0 or undefined


                # Print classification report
                print('  Classification Report:')
                try:
                    # Need at least one sample processed
                    if len(all_labels_indices) > 0:
                       # Use zero_division=0 or 1 to handle warnings/errors when a class has no predictions/labels
                       print(classification_report(all_labels_indices, all_preds_indices, target_names=class_names, zero_division=0))
                    else:
                       print("    No data to generate classification report.")

                except Exception as e_report:
                    print(f"    Warning: Could not generate classification report: {e_report}")
                    traceback.print_exc()


                # Check if this is the best model based on AUC
                if history['val_auc'] and history['val_auc'][-1] > best_auc: # Check if AUC was successfully appended
                    print(f'  🔥 New best model found! AUC improved from {best_auc:.4f} to {history["val_auc"][-1]:.4f}')
                    best_auc = history['val_auc'][-1]
                    best_model_wts = copy.deepcopy(model.state_dict())
                    try:
                        torch.save(model.state_dict(), MODEL_SAVE_PATH)
                        print(f'      Best model weights saved to {MODEL_SAVE_PATH}')
                    except Exception as e_save:
                        print(f"    ❌ Error saving best model weights: {e_save}")
                        traceback.print_exc()


                # Step the learning rate scheduler based on validation AUC
                # Only step if AUC calculation was successful and valid (history has AUC for this epoch)
                if history['val_auc'] and len(history['val_auc']) == epoch + 1 and positive_class_idx != -1 and len(np.unique(all_labels_indices)) > 1:
                    scheduler.step(history['val_auc'][-1]) # Step based on the last calculated AUC
                else:
                    # If AUC is not available or invalid, scheduler won't step effectively.
                    # Consider stepping based on val_loss if AUC is not suitable.
                    # print("  Skipping scheduler step (AUC not available or valid).") # Suppress frequent print
                    pass # Add pass to make the else block non-empty


        # --- End of Epoch ---
        # Generate Grad-CAM examples using validation data
        # Only generate if Grad-CAM was initialized successfully and NUM_GRAD_CAM_IMAGES > 0
        # This block should be at the same indentation level as the 'for phase in [...]' loop
        if grad_cam_generator is not None and NUM_GRAD_CAM_IMAGES > 0:
            try:
                generate_and_save_grad_cam_epoch(epoch, num_images=NUM_GRAD_CAM_IMAGES)
            except Exception as e_cam_epoch:
                print(f"    ❌ Error during Grad-CAM generation for epoch {epoch}: {e_cam_epoch}")
                traceback.print_exc()


        epoch_time_elapsed = time.time() - epoch_start_time
        print(f'Epoch {epoch} completed in {epoch_time_elapsed // 60:.0f}m {epoch_time_elapsed % 60:.1f}s')
        print('-' * 15)

    # --- End of Training ---
    time_elapsed = time.time() - start_time
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.1f}s')
    print(f'Best Validation AUC achieved: {best_auc:.4f}')
    print(f"Best model weights saved to: {MODEL_SAVE_PATH}")

    # Load best model weights back into the model
    try:
        # Check if the best model file exists before loading
        if os.path.exists(MODEL_SAVE_PATH):
            model.load_state_dict(best_model_wts)
            print("Loaded best model weights.")
        else:
            # This can happen if saving failed for all epochs
            print(f"Warning: Best model weights file not found at {MODEL_SAVE_PATH}. Skipping loading.")
    except Exception as e_load_best:
        print(f"    ❌ Error loading best model weights back: {e_load_best}")
        traceback.print_exc()


    return model, history

# ==============================================================================
# 8. Grad-CAM Generation Function (Called each epoch)
# ==============================================================================
def generate_and_save_grad_cam_epoch(epoch, num_images=NUM_GRAD_CAM_IMAGES):
    """ Generates and saves Grad-CAM images for a few validation samples. """
    if grad_cam_generator is None:
        # print("    Skipping Grad-CAM generation (generator not initialized).") # Suppress frequent print
        return # Silently skip if Grad-CAM is disabled

    print(f"\n🔍 Generating Grad-CAM examples for epoch {epoch}...")
    model.eval() # Ensure model is in eval mode

    count = 0
    # Get a few examples from the validation set *without shuffling* for consistency
    # Use num_workers=0 to avoid issues with multiprocessing and hooks/state
    # Make sure to use the actual validation dataset object
    if 'val' not in image_datasets or len(image_datasets['val']) == 0:
        print("    Warning: Validation dataset not available or empty. Skipping Grad-CAM generation.")
        return

    temp_val_loader = DataLoader(image_datasets['val'], batch_size=1, shuffle=False, num_workers=0) # Batch size 1 for simplicity
    val_loader_iter = iter(temp_val_loader)

    # Check if Grad-CAM output directory was successfully created
    if not os.path.isdir(GRAD_CAM_DIR):
        print(f"    Error: Grad-CAM output directory '{GRAD_CAM_DIR}' does not exist or could not be created. Skipping save.")
        return


    while count < num_images:
        try:
            try:
                # Get one image and label. Need to handle StopIteration explicitly.
                batch = next(val_loader_iter)
                inputs, labels = batch
            except StopIteration:
                print("    Warning: Reached end of validation set for Grad-CAM examples.")
                break # Exit the loop if no more images


            img_tensor = inputs # Shape [1, C, H, W]
            true_label_idx = labels[0].item()
            # Ensure true_label_idx is within bounds of class_names
            if true_label_idx < 0 or true_label_idx >= len(class_names):
                print(f"    Warning: True label index {true_label_idx} out of bounds for class_names. Skipping image {count}.")
                count += 1 # Increment count to move to the next requested image slot
                continue
            true_label_name = class_names[true_label_idx]

            # Generate CAM heatmap
            # We request CAM for the predicted class (target_class_idx=None)
            heatmap, predicted_class_idx = grad_cam_generator.generate(img_tensor) # Tensor already on DEVICE within generate

            if heatmap is not None and predicted_class_idx is not None:
                # Ensure predicted_class_idx is within bounds of class_names
                if predicted_class_idx < 0 or predicted_class_idx >= len(class_names):
                     print(f"    Warning: Predicted label index {predicted_class_idx} out of bounds for class_names. Skipping image {count}.")
                     count += 1 # Increment count to move to the next requested image slot
                     continue
                predicted_label_name = class_names[predicted_class_idx]

                # Convert tensor to OpenCV BGR image
                img_bgr = tensor_to_cv2_img(img_tensor.squeeze(0)) # Remove batch dim
                # Overlay CAM
                overlayed_image = overlay_grad_cam(img_bgr, heatmap)

                if overlayed_image is not None:
                    save_filename = f"epoch_{epoch:02d}_img_{count}_true_{true_label_name}_pred_{predicted_label_name}.png"
                    save_path = os.path.join(GRAD_CAM_DIR, save_filename)
                    try:
                        # Check if directory exists again just in case (unlikely with initial check)
                        if os.path.isdir(GRAD_CAM_DIR):
                            cv2.imwrite(save_path, overlayed_image)
                            # print(f"    Saved Grad-CAM: {save_path}") # Suppress frequent print
                            count += 1
                        else:
                            print(f"    Error: Grad-CAM output directory '{GRAD_CAM_DIR}' not found during save attempt. Skipping save.")
                            # Still increment count to try the next image slot
                            count += 1
                    except Exception as e_write:
                        print(f"    ❌ Error writing Grad-CAM image {save_path}: {e_write}")
                        traceback.print_exc()
                        # Still increment count to try the next image slot
                        count += 1 # Decide whether to increment on save error. Incrementing prevents infinite loop on a bad save.

                else:
                    # If heatmap or overlay failed, we printed error inside those functions
                    print(f"    Skipping Grad-CAM save for image {count} due to prior error in generation or overlay.")
                    count += 1 # Try next image slot

            else:
                 # If generate failed, an error was likely printed inside generate()
                 print(f"    Skipping image {count}: Grad-CAM generation returned None.")
                 count += 1 # Try next image slot


        except Exception as e_process_image:
            # This catches errors that weren't caught inside generate, overlay, or batch loading
            print(f"    ❌ Uncaught error processing image for Grad-CAM {count}: {e_process_image}")
            traceback.print_exc()
            count += 1 # Move to the next image slot


    if num_images > 0: # Only print summary if Grad-CAM was attempted
        if count >= num_images:
            print(f"✅ Successfully attempted to generate and save {num_images} Grad-CAM images.")
        elif count > 0:
            print(f"⚠️ Finished Grad-CAM attempt, saved {count} images (requested {num_images}). Errors occurred for some images.")
        else:
            print(f"❌ Grad-CAM generation/save failed for all attempted images this epoch.")


# ==============================================================================
# 9. Plotting Function
# ==============================================================================
def plot_training_history(history, save_path=HISTORY_PLOT_PATH):
    """ Plots training and validation loss, accuracy, and AUC. """
    try:
        # Ensure the save directory exists
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            # This should have been created by OUTPUT_BASE_DIR logic, but double check
            os.makedirs(save_dir, exist_ok=True)
            print(f"Plot save directory created: {save_dir}")

        epochs = range(len(history.get('train_loss', []))) # Use .get for safety

        if not epochs:
            print("Warning: No training history data to plot.")
            return # Nothing to plot if no epochs ran

        plt.figure(figsize=(18, 6))

        # Plot Loss
        plt.subplot(1, 3, 1)
        plt.plot(epochs, history.get('train_loss', []), 'bo-', label='Training Loss')
        plt.plot(epochs, history.get('val_loss', []), 'ro-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Plot Accuracy
        plt.subplot(1, 3, 2)
        plt.plot(epochs, history.get('train_acc', []), 'bo-', label='Training Accuracy')
        plt.plot(epochs, history.get('val_acc', []), 'ro-', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        # Plot Validation AUC
        plt.subplot(1, 3, 3)
        # Only plot if AUC data exists and matches epoch length
        if history.get('val_auc') and len(history['val_auc']) == len(epochs):
            plt.plot(epochs, history['val_auc'], 'go-', label='Validation AUC')
            plt.title('Validation AUC')
            plt.xlabel('Epoch')
            plt.ylabel('AUC')
            plt.legend()
            plt.grid(True)
        else:
            # print("Warning: AUC history not available or length mismatch. Skipping AUC plot.") # Suppress frequent print
            plt.title('Validation AUC (Not Available)')


        plt.tight_layout()
        plt.savefig(save_path)
        print(f"\nTraining history plot saved to: {save_path}")
        # plt.show() # Optionally display the plot directly
        plt.close() # Close the plot figure
    except Exception as e:
        print(f"Error plotting training history: {e}")
        traceback.print_exc()


# ==============================================================================
# 10. Main Execution Block
# ==============================================================================
if __name__ == '__main__':
    # Ensure the dataset path check passed earlier
    # This check is already done at the top and in data loading,
    # adding a final check here before starting training is safer
    # if you uncommented the exit() calls earlier.
    # Note: Data directory check happens before output directory check
    if not os.path.isdir(DATA_DIR) or not os.path.isdir(TRAIN_DIR) or not os.path.isdir(VAL_DIR):
         print("\nData directories are missing. Please check the DATA_DIR configuration and earlier logs.")
         # Exit if data is strictly required
         # import sys; sys.exit("Data directories missing.")
    # Also check if output directory was successfully created
    elif not os.path.isdir(OUTPUT_BASE_DIR):
        print("\nOutput base directory could not be created. Please check permissions and earlier logs.")
        # Exit if output directory is strictly required
        # import sys; sys.exit("Output directory cannot be created.")
    else:
        try: # Wrap the main training process in a try block
            # Start the training process
            trained_model, training_history = train_model(
                model,
                criterion,
                optimizer,
                scheduler,
                num_epochs=NUM_EPOCHS
            )

            print("\n✅ Training script finished successfully.")

            # Plot the training history
            plot_training_history(training_history, save_path=HISTORY_PLOT_PATH)

        except Exception as main_e:
            print(f"\n❌ An unexpected error occurred during the main training process: {main_e}")
            traceback.print_exc() # Print the full traceback

        finally:
            # Ensure Grad-CAM hooks are removed even if training crashed
            # Use the global grad_cam_generator variable
            # global grad_cam_generator # Removed this line
            if 'grad_cam_generator' in globals() and grad_cam_generator: # Safer check for global variable existence
                print("\nAttempting to remove Grad-CAM hooks...")
                try:
                    grad_cam_generator.remove_hooks()
                    print("Grad-CAM hooks removed.")
                except Exception as e_cleanup:
                    print(f"Error during Grad-CAM hook cleanup: {e_cleanup}")
                    traceback.print_exc()