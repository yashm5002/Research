# C:\SEED-VII_Project\utils\eeg_preprocessor.py
import numpy as np
import pandas as pd
import scipy.io
from sklearn.preprocessing import StandardScaler, RobustScaler
from pathlib import Path
import pickle

class EEGPreprocessor:
    def __init__(self, data_path, use_smoothed=True):
        """
        EEG feature preprocessing pipeline
        
        Args:
            data_path: Path to SEED-VII data directory
            use_smoothed: Whether to use LDS-smoothed features
        """
        self.data_path = Path(data_path)
        self.use_smoothed = use_smoothed
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_single_subject(self, subject_id):
        """Load EEG features for a single subject"""
        eeg_file = self.data_path / 'EEG_features' / f'{subject_id}.mat'
        data = scipy.io.loadmat(str(eeg_file))
        
        # Get appropriate feature keys
        if self.use_smoothed:
            de_keys = [k for k in data.keys() if k.startswith('de_LDS')]
        else:
            de_keys = [k for k in data.keys() if k.startswith('de_') and 'LDS' not in k]
        
        de_keys.sort()  # Ensure consistent ordering
        
        # Extract features
        features = []
        for key in de_keys:
            feature_matrix = data[key]  # Shape: (samples, channels, bands)
            # Flatten to (samples, channels*bands)
            flattened = feature_matrix.reshape(feature_matrix.shape[0], -1)
            features.append(flattened)
        
        return np.vstack(features), de_keys
    
    def load_all_subjects(self):
        """Load EEG features for all subjects"""
        all_features = []
        all_labels = []
        subject_ids = []
        trial_ids = []
        
        for subject_id in range(1, 21):
            print(f"Loading subject {subject_id}...")
            try:
                features, trial_keys = self.load_single_subject(subject_id)
                all_features.append(features)
                
                # Create subject and trial identifiers
                n_samples = features.shape[0]
                subject_ids.extend([subject_id] * n_samples)
                trial_ids.extend(list(range(1, n_samples + 1)))
                
            except Exception as e:
                print(f"Error loading subject {subject_id}: {e}")
        
        # Combine all features
        combined_features = np.vstack(all_features)
        
        # Store feature information
        if combined_features.shape[1] == 310:  # 62 channels × 5 bands
            self.feature_names = self._generate_feature_names()
        
        metadata = {
            'subject_ids': subject_ids,
            'trial_ids': trial_ids,
            'n_subjects': 20,
            'n_features': combined_features.shape[1]
        }
        
        return combined_features, metadata
    
    def _generate_feature_names(self):
        """Generate systematic feature names"""
        # Load channel names
        channel_file = self.data_path / 'Channel Order.xlsx'
        channels_df = pd.read_excel(channel_file)
        channel_names = channels_df.iloc[:, 0].tolist()
        
        bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        
        feature_names = []
        for channel in channel_names:
            for band in bands:
                feature_names.append(f"{channel}_{band}")
        
        return feature_names
    
    def normalize_features(self, features, method='standard'):
        """Normalize features using specified method"""
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError("Method must be 'standard' or 'robust'")
        
        normalized = scaler.fit_transform(features)
        return normalized, scaler
    
    def remove_outliers(self, features, threshold=3.0):
        """Remove outliers using z-score method"""
        z_scores = np.abs((features - np.mean(features, axis=0)) / np.std(features, axis=0))
        mask = np.all(z_scores < threshold, axis=1)
        return features[mask], mask
    
    def save_preprocessed_data(self, features, metadata, scaler, output_path):
        """Save preprocessed data and metadata"""
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True)
        
        # Save features
        np.save(output_path / 'eeg_features.npy', features)
        
        # Save metadata
        with open(output_path / 'eeg_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        # Save scaler
        with open(output_path / 'eeg_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        # Save feature names
        if self.feature_names:
            pd.DataFrame({'feature_name': self.feature_names}).to_csv(
                output_path / 'eeg_feature_names.csv', index=False
            )
        
        print(f"✓ EEG data saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    preprocessor = EEGPreprocessor('C:/SEED-VII_Project/data/SEED-VII')
    features, metadata = preprocessor.load_all_subjects()
    
    print(f"Loaded EEG features: {features.shape}")
    print(f"Feature range: {features.min():.3f} to {features.max():.3f}")
    
    # Normalize features
    normalized_features, scaler = preprocessor.normalize_features(features)
    print(f"Normalized feature range: {normalized_features.min():.3f} to {normalized_features.max():.3f}")
    
    # Save preprocessed data
    preprocessor.save_preprocessed_data(
        normalized_features, metadata, scaler, 
        'C:/SEED-VII_Project/data/preprocessed'
    )
