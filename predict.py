# predict.py - Prediction script to classify new images
# This script loads a trained model and makes predictions on new images

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import warnings

warnings.filterwarnings('ignore')

# Configuration constants
IMG_SIZE = (224, 224)                          # Must match training image size
MODEL_PATH = 'models/image_classifier_model.h5'  # Path to trained model
PREDICT_DIR = 'data/sample_images'             # Directory with images to predict

def load_trained_model():
    """
    Load the trained model from disk.
    
    Returns:
        Loaded Keras model
    """
    print("[INFO] Loading trained model...")
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
    
    model = load_model(MODEL_PATH)
    print(f"[SUCCESS] Model loaded from: {MODEL_PATH}")
    return model

def get_class_names():
    """
    Get class names from training directory structure.
    
    Returns:
        Dictionary mapping class indices to class names
    """
    train_dir = 'data/train'
    
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    
    # Get class names from subdirectories
    class_names = sorted([d for d in os.listdir(train_dir) 
                         if os.path.isdir(os.path.join(train_dir, d))])
    
    print(f"[INFO] Classes found: {class_names}")
    return class_names

def load_and_preprocess_image(image_path):
    """
    Load and preprocess a single image for prediction.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Preprocessed image array ready for model
    """
    # Load image
    img = load_img(image_path, target_size=IMG_SIZE)
    
    # Convert image to numpy array
    img_array = img_to_array(img)
    
    # Normalize pixel values (0-1)
    img_array = img_array / 255.0
    
    # Add batch dimension (model expects batch of images)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array, img

def predict_single_image(model, image_path, class_names):
    """
    Make prediction on a single image.
    
    Args:
        model: Trained Keras model
        image_path: Path to image file
        class_names: List of class names
        
    Returns:
        Predicted class name and confidence score
    """
    # Load and preprocess image
    img_array, original_img = load_and_preprocess_image(image_path)
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    
    # Get predicted class and confidence
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_idx]
    confidence = float(predictions[0][predicted_class_idx]) * 100
    
    return predicted_class, confidence, predictions[0], original_img

def predict_all_images(model, class_names):
    """
    Make predictions on all images in prediction directory.
    
    Args:
        model: Trained Keras model
        class_names: List of class names
    """
    print("[INFO] Making predictions on images...")
    
    if not os.path.exists(PREDICT_DIR):
        print(f"[WARNING] Prediction directory not found: {PREDICT_DIR}")
        print("[INFO] Create a 'data/sample_images' folder with images to predict")
        return
    
    # Get list of image files
    image_files = [f for f in os.listdir(PREDICT_DIR) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not image_files:
        print(f"[WARNING] No images found in: {PREDICT_DIR}")
        return
    
    print(f"[INFO] Found {len(image_files)} images to predict")
    
    # Store results
    results = []
    
    # Make predictions on each image
    for image_file in image_files:
        image_path = os.path.join(PREDICT_DIR, image_file)
        
        try:
            predicted_class, confidence, all_predictions, original_img = \
                predict_single_image(model, image_path, class_names)
            
            results.append({
                'filename': image_file,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'all_predictions': all_predictions,
                'image': original_img
            })
            
            print(f"[PREDICTION] {image_file}")
            print(f"  → Class: {predicted_class}")
            print(f"  → Confidence: {confidence:.2f}%")
            
        except Exception as e:
            print(f"[ERROR] Failed to predict {image_file}: {str(e)}")
    
    # Visualize results
    if results:
        visualize_predictions(results)
    
    return results

def visualize_predictions(results):
    """
    Visualize predictions with images and results.
    
    Args:
        results: List of prediction results
    """
    print("[INFO] Generating prediction visualization...")
    
    num_images = min(len(results), 9)  # Show max 9 images
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx in range(num_images):
        result = results[idx]
        
        # Plot image
        axes[idx].imshow(result['image'])
        axes[idx].set_title(f"Predicted: {result['predicted_class']}\n"
                           f"Confidence: {result['confidence']:.2f}%",
                           fontsize=12, fontweight='bold')
        axes[idx].axis('off')
    
    # Hide empty subplots
    for idx in range(num_images, 9):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    os.makedirs('images', exist_ok=True)
    plt.savefig('images/predictions_results.png', dpi=300, bbox_inches='tight')
    print("[SUCCESS] Predictions visualization saved to: images/predictions_results.png")
    plt.close()

def main():
    """Main prediction pipeline."""
    print("="*60)
    print("IMAGE CLASSIFICATION - PREDICTION")
    print("="*60)
    
    try:
        # Load trained model
        model = load_trained_model()
        
        # Get class names
        class_names = get_class_names()
        
        # Make predictions on all images
        results = predict_all_images(model, class_names)
        
        if results:
            print("="*60)
            print("[SUCCESS] Predictions completed successfully!")
            print(f"[INFO] Processed {len(results)} images")
            print("="*60)
        
    except Exception as e:
        print(f"[ERROR] An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
