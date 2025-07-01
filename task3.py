import os
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle

class CatDogSVMClassifier:
    def __init__(self, train_dir, test_dir, img_size=(64, 64)):
        """
        Initialize the SVM classifier for cats vs dogs
        
        Args:
            train_dir: Path to training directory
            test_dir: Path to test directory  
            img_size: Size to resize images (smaller for faster processing)
        """
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.img_size = img_size
        self.svm_model = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=300)  # Reduced components for efficiency
        
        # Verify directories exist
        if not os.path.exists(train_dir):
            raise ValueError(f"Training directory not found: {train_dir}")
        if not os.path.exists(test_dir):
            raise ValueError(f"Test directory not found: {test_dir}")
            
        print(f"Initialized classifier with image size: {img_size}")
        print(f"Training directory: {train_dir}")
        print(f"Test directory: {test_dir}")
        
    def extract_features(self, image_path):
        """
        Extract features from an image using multiple techniques
        
        Args:
            image_path: Path to the image
            
        Returns:
            Feature vector combining multiple feature extraction methods
        """
        try:
            # Read and preprocess image
            img = cv2.imread(image_path)
            if img is None:
                return None
                
            # Resize image
            img = cv2.resize(img, self.img_size)
            
            # Convert to different color spaces
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Feature 1: Pixel intensities (flattened and normalized)
            pixel_features = img.flatten().astype(np.float32) / 255.0
            
            # Feature 2: Histogram of oriented gradients (HOG) with custom parameters
            try:
                # Create HOG descriptor with smaller window and cell size
                hog = cv2.HOGDescriptor(
                    _winSize=(64, 64),
                    _blockSize=(16, 16),
                    _blockStride=(8, 8),
                    _cellSize=(8, 8),
                    _nbins=9
                )
                hog_features = hog.compute(gray)
                if hog_features is not None:
                    hog_features = hog_features.flatten()
                else:
                    hog_features = np.zeros(1764)  # Expected HOG size for 64x64
            except:
                # Fallback: use gradient-based features
                grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                magnitude = np.sqrt(grad_x**2 + grad_y**2)
                hog_features = magnitude.flatten()[:500]  # Limit size
                
            # Feature 3: Local Binary Pattern (simplified)
            lbp_features = self.calculate_lbp(gray)
            
            # Feature 4: Color histogram
            hist_b = cv2.calcHist([img], [0], None, [16], [0, 256])
            hist_g = cv2.calcHist([img], [1], None, [16], [0, 256])
            hist_r = cv2.calcHist([img], [2], None, [16], [0, 256])
            color_hist = np.concatenate([hist_b.flatten(), hist_g.flatten(), hist_r.flatten()])
            
            # Feature 5: Statistical features
            mean_vals = np.mean(img, axis=(0, 1))
            std_vals = np.std(img, axis=(0, 1))
            stat_features = np.concatenate([mean_vals, std_vals])
            
            # Ensure consistent feature sizes
            pixel_features = pixel_features[:500]  # Limit pixel features
            hog_features = hog_features[:500] if len(hog_features) > 500 else np.pad(hog_features, (0, max(0, 500-len(hog_features))))
            
            # Combine all features
            features = np.concatenate([
                pixel_features,
                hog_features,
                lbp_features,
                color_hist,
                stat_features
            ])
            
            return features.astype(np.float32)
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def calculate_lbp(self, gray_img):
        """
        Calculate Local Binary Pattern features (optimized)
        """
        try:
            h, w = gray_img.shape
            # Use smaller sampling for efficiency
            step = max(1, min(h, w) // 32)  # Adaptive step size
            lbp_values = []
            
            for i in range(1, h-1, step):
                for j in range(1, w-1, step):
                    center = gray_img[i, j]
                    binary_val = 0
                    
                    # 8-connectivity with bit shifting
                    neighbors = [
                        gray_img[i-1, j-1], gray_img[i-1, j], gray_img[i-1, j+1],
                        gray_img[i, j+1], gray_img[i+1, j+1], gray_img[i+1, j],
                        gray_img[i+1, j-1], gray_img[i, j-1]
                    ]
                    
                    for k, neighbor in enumerate(neighbors):
                        if neighbor >= center:
                            binary_val |= (1 << k)
                    
                    lbp_values.append(binary_val)
            
            # Calculate histogram of LBP values
            lbp_hist, _ = np.histogram(lbp_values, bins=32, range=(0, 256))
            return lbp_hist.astype(np.float32)
            
        except Exception as e:
            print(f"Error in LBP calculation: {e}")
            return np.zeros(32, dtype=np.float32)
    
    def load_training_data(self, sample_size=5000):
        """
        Load and preprocess training data
        
        Args:
            sample_size: Number of samples to use (reduce for faster training)
        """
        print("Loading training data...")
        
        # Get all image files
        image_files = [f for f in os.listdir(self.train_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Limit sample size for faster processing
        if len(image_files) > sample_size:
            np.random.seed(42)
            image_files = np.random.choice(image_files, sample_size, replace=False)
        
        X = []
        y = []
        
        for filename in tqdm(image_files, desc="Processing training images"):
            # Extract label from filename
            if filename.startswith('cat'):
                label = 0  # Cat
            elif filename.startswith('dog'):
                label = 1  # Dog
            else:
                continue
            
            # Extract features
            image_path = os.path.join(self.train_dir, filename)
            features = self.extract_features(image_path)
            
            if features is not None:
                X.append(features)
                y.append(label)
        
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        
        print(f"Loaded {len(self.X_train)} training samples")
        print(f"Feature vector size: {self.X_train.shape[1]}")
        print(f"Class distribution: {np.bincount(self.y_train)}")
        
        return self.X_train, self.y_train
    
    def preprocess_features(self, X_train, X_test=None):
        """
        Preprocess features using scaling and PCA
        """
        print("Preprocessing features...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Apply PCA for dimensionality reduction
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            X_test_pca = self.pca.transform(X_test_scaled)
            return X_train_pca, X_test_pca
        
        return X_train_pca
    
    def train_svm(self, X_train, y_train):
        """
        Train SVM classifier
        """
        print("Training SVM classifier...")
        
        # Split training data for validation
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Preprocess features
        X_train_processed = self.preprocess_features(X_train_split)
        X_val_processed = self.scaler.transform(X_val_split)
        X_val_processed = self.pca.transform(X_val_processed)
        
        # Train SVM with RBF kernel
        self.svm_model = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42
        )
        
        self.svm_model.fit(X_train_processed, y_train_split)
        
        # Validate model
        y_val_pred = self.svm_model.predict(X_val_processed)
        val_accuracy = accuracy_score(y_val_split, y_val_pred)
        
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print("\nValidation Classification Report:")
        print(classification_report(y_val_split, y_val_pred, 
                                  target_names=['Cat', 'Dog']))
        
        return self.svm_model
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """
        Plot confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Cat', 'Dog'],
                    yticklabels=['Cat', 'Dog'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
    
    def predict_test_images(self, output_file='test_predictions.csv'):
        """
        Make predictions on test images
        """
        print("Making predictions on test images...")
        
        # Get test image files
        test_files = [f for f in os.listdir(self.test_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        predictions = []
        probabilities = []
        
        for filename in tqdm(test_files, desc="Processing test images"):
            image_path = os.path.join(self.test_dir, filename)
            features = self.extract_features(image_path)
            
            if features is not None:
                # Preprocess features
                features_scaled = self.scaler.transform([features])
                features_pca = self.pca.transform(features_scaled)
                
                # Make prediction
                pred = self.svm_model.predict(features_pca)[0]
                prob = self.svm_model.predict_proba(features_pca)[0]
                
                predictions.append({
                    'filename': filename,
                    'prediction': 'dog' if pred == 1 else 'cat',
                    'confidence': max(prob)
                })
                probabilities.append(prob)
        
        # Save predictions
        import pandas as pd
        df = pd.DataFrame(predictions)
        df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
        
        return predictions
    
    def save_model(self, model_path='svm_cat_dog_model.pkl'):
        """
        Save trained model and preprocessors
        """
        model_data = {
            'svm_model': self.svm_model,
            'scaler': self.scaler,
            'pca': self.pca,
            'img_size': self.img_size
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path='svm_cat_dog_model.pkl'): 
        """
        Load trained model and preprocessors
        """
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.svm_model = model_data['svm_model']
        self.scaler = model_data['scaler']
        self.pca = model_data['pca']
        self.img_size = model_data['img_size']
        
        print(f"Model loaded from {model_path}")

# Main execution
if __name__ == "__main__":
    # Define directories
    TRAIN_DIR = "C:/Users/Yatin Singh/Downloads/dogs-vs-cats/train/train"
    TEST_DIR = "C:/Users/Yatin Singh/Downloads/dogs-vs-cats/test1/test1"
    
    # Initialize classifier
    classifier = CatDogSVMClassifier(TRAIN_DIR, TEST_DIR, img_size=(64, 64))
    
    # Load and prepare training data
    X_train, y_train = classifier.load_training_data(sample_size=25000)  # Reduced for faster training
    
    # Train the model
    model = classifier.train_svm(X_train, y_train)
    
    # Save the model
    classifier.save_model('svm_cat_dog_model.pkl')
    
    # Make predictions on test set
    predictions = classifier.predict_test_images('test_predictions.csv')
    
    print(f"\nPrediction completed! Made predictions for {len(predictions)} test images.")
    print("First 5 predictions:")
    for i, pred in enumerate(predictions[:5]):
        print(f"{pred['filename']}: {pred['prediction']} (confidence: {pred['confidence']:.3f})")