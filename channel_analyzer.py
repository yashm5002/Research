# C:\SEED-VII_Project\utils\channel_analyzer.py
import pandas as pd
import os

def analyze_channels():
    """Analyze EEG channel configuration"""
    
    # Load channel order information
    channel_file = 'C:/SEED-VII_Project/data/SEED-VII/Channel Order.xlsx'
    if os.path.exists(channel_file):
        channels_df = pd.read_excel(channel_file)
        print("=== Channel Configuration ===")
        print(f"Total channels: {len(channels_df)}")
        print(f"First 10 channels: {channels_df.iloc[:10, 0].tolist()}")
        
        # Frequency bands analysis
        print("\n=== Frequency Bands (DE Features) ===")
        bands = {
            'Delta': '1-4 Hz',
            'Theta': '4-8 Hz', 
            'Alpha': '8-14 Hz',
            'Beta': '14-31 Hz',
            'Gamma': '31-50 Hz'
        }
        
        for band, freq_range in bands.items():
            print(f"  {band}: {freq_range}")
            
        total_features = len(channels_df) * len(bands)
        print(f"\nTotal DE features per sample: {total_features}")
        print(f"Features per band: {len(channels_df)}")
    
    # Analyze emotion labels
    emotion_file = 'C:/SEED-VII_Project/data/SEED-VII/emotion_label_and_stimuli_order.xlsx'
    if os.path.exists(emotion_file):
        emotion_df = pd.read_excel(emotion_file)
        print(f"\n=== Emotion Distribution ===")
        print(f"Total video trials: {len(emotion_df)}")
        
        # Count emotions (excluding VideoID column)
        for col in emotion_df.columns:
            if col != 'VideoID' and emotion_df[col].dtype == 'int64':
                count = emotion_df[col].sum()
                print(f"{col}: {count} videos")

if __name__ == "__main__":
    analyze_channels()
