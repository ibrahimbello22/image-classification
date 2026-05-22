# train.py - Training script for image classification model
# This script trains a CNN model on images organized in class folders

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# Configuration constants
IMG_SIZE = (224, 224)          # Image resolution
BATCH_SIZE = 32                # Number of images per batch
EPOCHS = 10                    # Number of training epochs
TRAIN_DIR = 'data/train'       # Training data directory
MODEL_SAVE_PATH = 'models/image_classifier_model.h5'  # Model save location

def create_model(num_classes):
    """
    Create a Convolutional Neural Network (CNN) model.
    
    Args:
        num_classes: Number of image classes to classify
        
    Returns:
        Compiled Keras model
    """
    print("[INFO] Building CNN model...")
    
    # Create Sequential model (layers stacked on top of each other)
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Flatten layer (convert 3D to 1D)
        Flatten(),
        
        # Fully Connected (Dense) layers
        Dense(128, activation='relu'),
        Dropout(0.5),
        
        # Output layer (softmax for multi-class classification)
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("[SUCCESS] Model created successfully!")
    model.summary()
    return model

def load_and_prepare_data():
    """
    Load and prepare training data using ImageDataGenerator for augmentation.
    
    Returns:
        train_generator: Data generator for training images
        num_classes: Number of classes found
    """
    print("[INFO] Loading and preparing training data...")
    
    # Data augmentation to improve model generalization
    # Augmentation applies random transformations to images
    train_datagen = ImageDataGenerator(
        rescale=1./255,                    # Normalize pixel values (0-1)
        rotation_range=20,                 # Random rotation (0-20 degrees)
        width_shift_range=0.2,             # Random horizontal shift (20%)
        height_shift_range=0.2,            # Random vertical shift (20%)
        zoom_range=0.2,                    # Random zoom (80-120%)
        horizontal_flip=True,              # Random horizontal flip
        shear_range=0.2                    # Random shear transformation
    )
    
    # Check if training directory exists
    if not os.path.exists(TRAIN_DIR):
        raise FileNotFoundError(f"Training directory not found: {TRAIN_DIR}")
    
    # Load images from directory
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'  # One-hot encoding for classes
    )
    
    num_classes = len(train_generator.classes)
    print(f"[SUCCESS] Found {len(train_generator.filenames)} training images")
    print(f"[INFO] Number of classes: {num_classes}")
    print(f"[INFO] Classes: {train_generator.class_indices}")
    
    return train_generator, num_classes

def train_model(model, train_generator):
    """
    Train the CNN model on training data.
    
    Args:
        model: Compiled Keras model
        train_generator: Data generator with training images
        
    Returns:
        history: Training history with metrics
    """
    print("[INFO] Starting model training...")
    
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        steps_per_epoch=len(train_generator),
        verbose=1
    )
    
    print("[SUCCESS] Model training completed!")
    return history

def plot_training_history(history):
    """
    Plot training accuracy and loss graphs.
    
    Args:
        history: Training history from model.fit()
    """
    print("[INFO] Generating accuracy and loss graphs...")
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Model Accuracy During Training', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2, color='orange')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Model Loss During Training', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs('images', exist_ok=True)
    plt.savefig('images/accuracy_loss_graph.png', dpi=300, bbox_inches='tight')
    print("[SUCCESS] Graph saved to: images/accuracy_loss_graph.png")
    plt.close()

def save_model(model):
    """
    Save the trained model to disk.
    
    Args:
        model: Trained Keras model
    """
    print("[INFO] Saving trained model...")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model in HDF5 format
    model.save(MODEL_SAVE_PATH)
    print(f"[SUCCESS] Model saved to: {MODEL_SAVE_PATH}")

def main():
    """Main training pipeline."""
    print("="*60)
    print("IMAGE CLASSIFICATION - MODEL TRAINING")
    print("="*60)
    
    try:
        # Step 1: Load and prepare data
        train_generator, num_classes = load_and_prepare_data()
        
        # Step 2: Create model
        model = create_model(num_classes)
        
        # Step 3: Train model
        history = train_model(model, train_generator)
        
        # Step 4: Plot results
        plot_training_history(history)
        
        # Step 5: Save model
        save_model(model)
        
        print("="*60)
        print("[SUCCESS] Training pipeline completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"[ERROR] An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
