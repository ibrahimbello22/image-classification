# main.py - Main entry point for the image classification project
# This script orchestrates the complete workflow

import os
import sys

def check_environment():
    """Check if all required modules are installed."""
    print("[INFO] Checking environment and dependencies...")
    
    required_packages = ['tensorflow', 'keras', 'numpy', 'matplotlib', 'sklearn', 'PIL']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            elif package == 'PIL':
                __import__('PIL')
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"[ERROR] Missing packages: {', '.join(missing_packages)}")
        print("[INFO] Install with: pip install -r requirements.txt")
        return False
    
    print("[SUCCESS] All dependencies are installed!")
    return True

def check_data_structure():
    """Check if required data folders exist."""
    print("[INFO] Checking data structure...")
    
    required_dirs = ['data/train', 'data/test', 'models', 'images']
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"[WARNING] Creating missing directory: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
    
    # Check if training data exists
    train_dir = 'data/train'
    if os.path.exists(train_dir):
        subdirs = [d for d in os.listdir(train_dir) 
                  if os.path.isdir(os.path.join(train_dir, d))]
        if not subdirs:
            print("[WARNING] No class folders found in data/train")
            print("[INFO] Create folders like: data/train/class1, data/train/class2, ...")
            print("[INFO] Add images to these folders before training")
            return False
        else:
            total_images = 0
            for subdir in subdirs:
                images = [f for f in os.listdir(os.path.join(train_dir, subdir))
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                total_images += len(images)
                print(f"    {subdir}: {len(images)} images")
            
            if total_images == 0:
                print("[WARNING] No images found in training folders")
                return False
    else:
        print("[WARNING] Training directory not found: data/train")
        return False
    
    print("[SUCCESS] Data structure is valid!")
    return True

def display_menu():
    """Display main menu."""
    print("\n" + "="*60)
    print("IMAGE CLASSIFICATION PROJECT - MAIN MENU")
    print("="*60)
    print("1. Train the model (requires data)")
    print("2. Make predictions (requires trained model)")
    print("3. Run full pipeline (train + predict)")
    print("4. Check status")
    print("5. Exit")
    print("="*60)

def check_status():
    """Check project status."""
    print("\n[INFO] Checking project status...\n")
    
    # Check for training data
    train_exists = os.path.exists('data/train')
    train_images = 0
    if train_exists:
        for root, dirs, files in os.walk('data/train'):
            train_images += len([f for f in files 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    
    print(f"Training data: {'✓ Available' if train_exists and train_images > 0 else '✗ Not available'}")
    if train_images > 0:
        print(f"  ({train_images} images found)")
    
    # Check for trained model
    model_exists = os.path.exists('models/image_classifier_model.h5')
    print(f"Trained model: {'✓ Available' if model_exists else '✗ Not available'}")
    
    # Check for prediction images
    predict_dir = 'data/sample_images'
    predict_images = 0
    if os.path.exists(predict_dir):
        predict_images = len([f for f in os.listdir(predict_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    
    print(f"Sample images: {'✓ Available' if predict_images > 0 else '✗ Not available'}")
    if predict_images > 0:
        print(f"  ({predict_images} images in data/sample_images)")
    
    print()

def run_training():
    """Run training script."""
    print("\n[INFO] Starting training pipeline...")
    
    if not check_data_structure():
        print("[ERROR] Cannot proceed with training")
        return
    
    try:
        import train
        train.main()
    except Exception as e:
        print(f"[ERROR] Training failed: {str(e)}")

def run_prediction():
    """Run prediction script."""
    print("\n[INFO] Starting prediction pipeline...")
    
    if not os.path.exists('models/image_classifier_model.h5'):
        print("[ERROR] Trained model not found")
        print("[INFO] Please train the model first")
        return
    
    try:
        import predict
        predict.main()
    except Exception as e:
        print(f"[ERROR] Prediction failed: {str(e)}")

def run_full_pipeline():
    """Run complete training and prediction pipeline."""
    print("\n[INFO] Starting full pipeline...")
    
    if not check_data_structure():
        print("[ERROR] Cannot proceed with training")
        return
    
    # Run training
    run_training()
    
    # Run prediction
    print("\n[INFO] Starting prediction on sample images...")
    run_prediction()

def main():
    """Main application loop."""
    print("="*60)
    print("WELCOME TO IMAGE CLASSIFICATION PROJECT")
    print("="*60)
    
    # Check environment
    if not check_environment():
        print("\n[ERROR] Please install missing dependencies first")
        print("Run: pip install -r requirements.txt")
        return
    
    # Main loop
    while True:
        display_menu()
        
        try:
            choice = input("Enter your choice (1-5): ").strip()
            
            if choice == '1':
                run_training()
            elif choice == '2':
                run_prediction()
            elif choice == '3':
                run_full_pipeline()
            elif choice == '4':
                check_status()
            elif choice == '5':
                print("\n[INFO] Thank you for using Image Classification Project!")
                break
            else:
                print("[ERROR] Invalid choice. Please try again.")
        
        except KeyboardInterrupt:
            print("\n[INFO] Exiting...")
            break
        except Exception as e:
            print(f"[ERROR] An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
