# C:\SEED-VII_Project\utils\eye_preprocessor.py
import numpy as np
import pandas as pd
import scipy.io
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pickle

class EyePreprocessor:
    def __init__(self, data_path):
        """
        Eye movement feature preprocessing pipeline
        
        Args:
            data_path: Path to SEED-VII data directory
        """
        self.data_path = Path(data_path)
        self.scaler = StandardScaler()
        
    def load_single_subject_eye(self, subject_id):
        """Load eye movement features for a single subject - FIXED"""
        # Corrected: files are named {subject_id}.mat
        eye_file = self.data_path / 'EYE_features' / f'{subject_id}.mat'
        
        if not eye_file.exists():
            raise FileNotFoundError(f"No eye movement file found: {eye_file}")
        
        data = scipy.io.loadmat(str(eye_file))
        eye_keys = [k for k in data.keys() if not k.startswith('__')]
        
        # Extract features with robust dimension handling
        processed_features = []
        for key in eye_keys:
            feature_data = data[key]
            if isinstance(feature_data, np.ndarray) and feature_data.size > 0:
                # Convert to statistical features to handle dimension mismatches
                if feature_data.ndim > 1:
                    # Extract statistics from each column
                    for col in range(min(feature_data.shape[1], 10)):  # Limit columns
                        col_data = feature_data[:, col]
                        stats = [
                            np.mean(col_data), np.std(col_data), 
                            np.median(col_data), np.min(col_data), np.max(col_data)
                        ]
                        processed_features.extend(stats)
                else:
                    # 1D array statistics
                    stats = [
                        np.mean(feature_data), np.std(feature_data),
                        np.median(feature_data), np.min(feature_data), np.max(feature_data)
                    ]
                    processed_features.extend(stats)
        
        return np.array(processed_features).reshape(1, -1) if processed_features else np.array([])
    
    def extract_eye_statistics(self, eye_data):
        """Extract statistical features from eye movement data"""
        if eye_data.size == 0:
            return np.array([])
        
        stats_features = []
        
        # Basic statistics
        stats_features.extend([
            np.mean(eye_data),
            np.std(eye_data),
            np.median(eye_data),
            np.min(eye_data),
            np.max(eye_data),
            np.percentile(eye_data, 25),
            np.percentile(eye_data, 75),
        ])
        
        # Additional features
        stats_features.extend([
            np.var(eye_data),
            len(eye_data),  # Number of samples
            np.sum(eye_data > np.mean(eye_data)),  # Above-average count
        ])
        
        return np.array(stats_features)
    
    def load_all_subjects_eye(self):
        """Load eye movement features for all subjects"""
        all_features = []
        all_metadata = []
        
        for subject_id in range(1, 21):
            print(f"Loading eye data for subject {subject_id}...")
            try:
                eye_data = self.load_single_subject_eye(subject_id)
                
                if eye_data.size > 0:
                    all_features.append(eye_data)
                    all_metadata.append(subject_id)
                    print(f"✓ Subject {subject_id}: {eye_data.shape[1]} eye features")
                
            except Exception as e:
                print(f"Warning: Could not load eye data for subject {subject_id}: {e}")
        
        if all_features:
            combined_features = np.vstack(all_features)
            print(f"✓ Successfully loaded eye features: {combined_features.shape}")
        else:
            combined_features = np.array([])
            print("⚠ No eye movement features loaded")
        
        metadata = {
            'subject_ids': all_metadata,
            'n_eye_features': combined_features.shape[1] if combined_features.size > 0 else 0
        }
        
        return combined_features, metadata
    
    def save_eye_data(self, features, metadata, scaler, output_path):
        """Save preprocessed eye movement data"""
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True)
        
        if features.size > 0:
            np.save(output_path / 'eye_features.npy', features)
            
            with open(output_path / 'eye_metadata.pkl', 'wb') as f:
                pickle.dump(metadata, f)
            
            with open(output_path / 'eye_scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)
            
            print(f"✓ Eye movement data saved to {output_path}")
        else:
            print("⚠ No eye movement features to save")

if __name__ == "__main__":
    preprocessor = EyePreprocessor('C:/SEED-VII_Project/data/SEED-VII')
    features, metadata = preprocessor.load_all_subjects_eye()
    
    if features.size > 0:
        print(f"Loaded eye features: {features.shape}")
        
        # Normalize
        normalized_features = preprocessor.scaler.fit_transform(features)
        
        # Save
        preprocessor.save_eye_data(
            normalized_features, metadata, preprocessor.scaler,
            'C:/SEED-VII_Project/data/preprocessed'
        )
    else:
        print("No eye movement features found")
