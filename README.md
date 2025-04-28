# Chest X-Ray Classification with Custom CNN
<center><img src="https://github.com/user-attachments/assets/a7cca9ee-fd0b-4d63-925b-501ccaad4927" alt="Project Logo" width="250"></center>

## Project Overview

This project implements a custom Convolutional Neural Network (CNN) using PyTorch to classify Chest X-Ray images as either Normal or Pneumonia. The codebase includes advanced features such as Grad-CAM for model interpretability, handling of class imbalance, and configurable output directories for organizing results.

## Principle

The following principles underpin the design and functionality of this project:

- **Custom CNN Architecture:** The CNN consists of multiple convolutional blocks, each featuring convolution layers, batch normalization, ReLU activation, and max pooling. This structure is tailored to effectively extract spatial features from X-Ray images, culminating in fully connected layers for classification.

- **Grad-CAM (Gradient-weighted Class Activation Mapping):** Grad-CAM is integrated to provide visual explanations of the model's predictions by highlighting the image regions most influential to the classification outcome. This enhances interpretability, allowing users to verify the model's focus areas.

- **Class Imbalance Handling:** To address potential disparities in the dataset (e.g., more Normal than Pneumonia images), class weights are calculated based on class frequencies and applied to the loss function. This ensures the model pays adequate attention to the minority class.

- **Output Directory Configuration:** The script supports customizable output directories for saving model weights, training history plots, and Grad-CAM visualizations. This modularity facilitates result management and reproducibility.

## Framework

The project is structured around the following components:

- **Data Loading and Preprocessing:**
    * Utilizes torchvision.datasets.ImageFolder to load the dataset from directories structured as train and test.
    * Applies data augmentation (e.g., random cropping, flipping) for training and normalization for both training and validation using ImageNet statistics.

- **Model Definition:**
    * A CustomCNN class defines the network with four convolutional blocks followed by fully connected layers.
    * The architecture dynamically calculates the flattened feature size to adapt to input dimensions.

- **Training Loop:**
    * Implements training and validation phases per epoch, tracking metrics such as loss, accuracy, and AUC.
    * Saves the best model based on validation AUC, with a learning rate scheduler to optimize training.

- **Grad-CAM Implementation:**
    * A GradCAM class generates heatmaps by capturing gradients and activations from a specified target layer (e.g., the last convolutional layer).
    * Visualizations are overlaid on original images and saved for analysis.

- **Visualization and Result Saving:**
    * Functions plot training history (loss, accuracy, AUC) and save them as images.
    * Grad-CAM outputs are stored in a dedicated directory, with filenames indicating epoch, true label, and predicted label.

## Training Record (Overfitting Issue Still Present)

The training history plot illustrates the model's performance over multiple epochs, highlighting an ongoing overfitting issue:

- **Observation:** The plot shows training and validation loss, accuracy, and AUC. Overfitting is evident, as the model performs well on training data but struggles to generalize to validation data.
- **Evidence of Overfitting:** A decreasing training loss with a stagnating or increasing validation loss, or a widening gap between training and validation accuracy/AUC, indicates overfitting.
- **Potential Solutions:**
    * Increase Data Augmentation: Apply more aggressive transformations (e.g., random rotations, flips).
    * Regularization: Implement dropout or weight decay to penalize complex models.
    * Early Stopping: Stop training when validation performance plateaus.
    * Adjust Model Complexity: Simplify the CNN architecture if it is too complex for the dataset.

## Heatmap Effect Demonstration

The Grad-CAM heatmap demonstrates the model's focus areas in X-Ray images:

- **Purpose:** The heatmap highlights regions of the X-Ray image that most influence the model's prediction (e.g., "Normal" or "Pneumonia").
- **Interpretation:** The colored overlay (red for high importance) indicates focus areas, such as abnormal opacities in pneumonia cases.
- **Significance:** Enhances interpretability by verifying if the model targets clinically relevant regions (e.g., lungs) rather than irrelevant areas (e.g., background).
- **Usage:** Generated for a configurable number of images per epoch, saved with filenames indicating epoch, true label, and predicted label.

## Usage

1.  **Set Data Directory:** Modify the `DATA_DIR` variable to point to your Chest X-Ray dataset (e.g., `D:\archive\chest_xray`).
2.  **Configure Output Directory:** Update `OUTPUT_BASE_DIR` to your desired output location (e.g., `C:\Users\LRQ\Desktop\output`).
3.  **Run the Script:** Execute the Python script to initiate data loading, model training, and result generation.

## Requirements

-   **Python Libraries:**
    * PyTorch
    * Torchvision
    * OpenCV (cv2)
    * NumPy
    * Matplotlib
    * Scikit-learn

## Notes

-   The dataset should be organized with `train` and `test` subfolders, each containing class-specific subdirectories (`NORMAL` and `PNEUMONIA`).
-   Error handling and logging are implemented to troubleshoot issues during execution.
-   Grad-CAM visualizations are generated for a configurable number of images per epoch, stored in the output directory.
  
## Train Demonstration
![training_history](https://github.com/user-attachments/assets/8cb3276a-5054-492b-a93f-9b5af50284ff)

## Heatmap Effect Demonstraion
![Heatmap Effect Demonstration](https://github.com/user-attachments/assets/4f103722-188f-4025-9a30-4fea2c4962fb)
