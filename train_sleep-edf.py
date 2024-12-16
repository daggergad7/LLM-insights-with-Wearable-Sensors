# -*- coding: utf-8 -*-
"""Separate Training for Sleep-EDF (EEG & EMG only) with Confusion Matrix"""

# Install required libraries
!pip install mne
!pip install tensorflow
!pip install scikit-learn
!pip install openai
!pip install imbalanced-learn
!pip install seaborn

import os
import pickle
import numpy as np
import pandas as pd
import mne
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns
import openai
import warnings
from collections import Counter
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')

###########################################################
#            Load and Process Sleep-EDF Data (EEG+EMG)    #
###########################################################

sleepedf_data_path = '/content/drive/MyDrive/WANProject/extracted/sleep-edf-database-expanded-1.0.0/sleep-telemetry'

psg_files = [f for f in os.listdir(sleepedf_data_path) if 'PSG.edf' in f]
hypnogram_files = [f for f in os.listdir(sleepedf_data_path) if 'Hypnogram.edf' in f]

psg_files.sort()
hypnogram_files.sort()

print(f"Found {len(psg_files)} PSG files and {len(hypnogram_files)} hypnogram files in Sleep-EDF dataset.")

def inspect_annotations(hyp_path):
    try:
        annotations = mne.read_annotations(hyp_path)
        unique_desc, counts = np.unique(annotations.description, return_counts=True)
        print(f"Annotations in {os.path.basename(hyp_path)}:")
        for desc, count in zip(unique_desc, counts):
            print(f"  {desc}: {count}")
    except Exception as e:
        print(f"Error reading {hyp_path}: {e}")

# Optional: Inspect first 5 hypnogram files
for idx, hyp_file in enumerate(hypnogram_files[:5]):
    hyp_path = os.path.join(sleepedf_data_path, hyp_file)
    print(f"\nInspecting Annotations for File {idx+1}: {hyp_file}")
    inspect_annotations(hyp_path)

def extract_sleep_features_corrected(raw, annotations):
    """
    Extract basic sleep efficiency features.
    """
    raw.set_annotations(annotations)
    all_annotations = annotations
    if not all_annotations:
        return {
            'Total_Sleep_Time': 0,
            'Sleep_Efficiency': 0,
            'stage_1_percent': 0,
            'stage_2_percent': 0,
            'stage_3_percent': 0,
            'stage_R_percent': 0,
            'stage_0_percent': 0
        }

    stage_mapping = {
        'Sleep stage W': 'W',
        'Sleep stage 1': 'N1',
        'Sleep stage 2': 'N2',
        'Sleep stage 3': 'N3',
        'Sleep stage 4': 'N3',
        'Sleep stage R': 'R',
        'Movement time': 'W',
        'Sleep stage ?': 'Unknown'
    }

    tib_start = all_annotations.onset[0]
    tib_end = all_annotations.onset[-1] + all_annotations.duration[-1]
    total_time_in_bed = tib_end - tib_start  # seconds

    stage_durations = {'W': 0,'N1': 0,'N2': 0,'N3': 0,'R': 0,'Unknown': 0}

    for onset, duration, desc in zip(all_annotations.onset, all_annotations.duration, all_annotations.description):
        stage = stage_mapping.get(desc, 'Unknown')
        if stage != 'Unknown':
            stage_durations[stage] += duration

    total_sleep_time = stage_durations['N1'] + stage_durations['N2'] + stage_durations['N3'] + stage_durations['R']
    sleep_efficiency = (total_sleep_time / total_time_in_bed)*100 if total_time_in_bed > 0 else 0

    stage_percentages = {}
    for stage in ['N1', 'N2', 'N3', 'R']:
        stage_time = stage_durations[stage]
        stage_percentages[f'stage_{stage}_percent'] = (stage_time / total_sleep_time)*100 if total_sleep_time>0 else 0

    wake_time = stage_durations['W']
    stage_percentages['stage_0_percent'] = (wake_time / total_time_in_bed)*100 if total_time_in_bed>0 else 0

    features = {
        'Total_Sleep_Time': total_sleep_time / 3600,
        'Sleep_Efficiency': sleep_efficiency,
        **stage_percentages
    }
    return features

def extract_signal_features(signal):
    """
    Extract a set of features from a given signal.
    """
    return {
        'Mean': np.mean(signal),
        'Std': np.std(signal),
        'Max': np.max(signal),
        'Min': np.min(signal),
        'Skew': skew(signal),
        'Kurtosis': kurtosis(signal),
        'Median': np.median(signal),
        'Variance': np.var(signal),
        'RMS': np.sqrt(np.mean(signal**2))
    }

sleep_features_list = []

for idx, (psg_file, hyp_file) in enumerate(zip(psg_files, hypnogram_files)):
    psg_path = os.path.join(sleepedf_data_path, psg_file)
    hyp_path = os.path.join(sleepedf_data_path, hyp_file)

    print(f"\nProcessing File {idx+1}: {psg_file} and {hyp_file}")

    try:
        raw = mne.io.read_raw_edf(psg_path, preload=True, stim_channel=None, verbose=False)
        annotations = mne.read_annotations(hyp_path)

        # Extract sleep stage features
        stage_features = extract_sleep_features_corrected(raw, annotations)

        # Common channel names in Sleep-EDF
        eeg_channel = 'EEG Fpz-Cz'  # adjust if needed
        emg_channel = 'EMG submental'  # adjust if needed

        if eeg_channel not in raw.ch_names:
            print(f"EEG channel {eeg_channel} not found in {psg_file}. Available channels: {raw.ch_names}")
            continue
        if emg_channel not in raw.ch_names:
            print(f"EMG channel {emg_channel} not found in {psg_file}. Available channels: {raw.ch_names}")
            continue

        eeg_data = raw.get_data(picks=[eeg_channel]).flatten()
        emg_data = raw.get_data(picks=[emg_channel]).flatten()

        eeg_f = extract_signal_features(eeg_data)
        emg_f = extract_signal_features(emg_data)

        combined_features = {
            **stage_features,
            'EEG_Mean': eeg_f['Mean'],
            'EEG_Std': eeg_f['Std'],
            'EEG_Max': eeg_f['Max'],
            'EEG_Min': eeg_f['Min'],
            'EEG_Skew': eeg_f['Skew'],
            'EEG_Kurtosis': eeg_f['Kurtosis'],
            'EEG_Median': eeg_f['Median'],
            'EEG_Variance': eeg_f['Variance'],
            'EEG_RMS': eeg_f['RMS'],

            'EMG_Mean': emg_f['Mean'],
            'EMG_Std': emg_f['Std'],
            'EMG_Max': emg_f['Max'],
            'EMG_Min': emg_f['Min'],
            'EMG_Skew': emg_f['Skew'],
            'EMG_Kurtosis': emg_f['Kurtosis'],
            'EMG_Median': emg_f['Median']
        }

        # Map sleep efficiency to sleep quality
        sleep_eff = combined_features['Sleep_Efficiency']
        if sleep_eff >= 85:
            emotion = 'Well Rested'
        elif sleep_eff >= 75:
            emotion = 'Moderately Rested'
        else:
            emotion = 'Poorly Rested'

        combined_features['Emotion'] = emotion
        sleep_features_list.append(combined_features)

    except Exception as e:
        print(f"Error processing file {psg_file} and {hyp_file}: {e}")

sleepedf_df = pd.DataFrame(sleep_features_list)
print("\nSleep-EDF features shape:", sleepedf_df.shape)
print(sleepedf_df.head())

###########################################################
#           Prepare Sleep-EDF Data (EEG+EMG) for Modeling #
###########################################################

sleep_labels = sleepedf_df['Emotion'].values
sleep_le = LabelEncoder()
sleep_labels_encoded = sleep_le.fit_transform(sleep_labels)
print("Sleep-EDF Emotion classes (Sleep Quality):", sleep_le.classes_)

# EEG and EMG features only
sleep_feature_cols = [c for c in sleepedf_df.columns if c.startswith('EEG_') or c.startswith('EMG_')]

X_sleep = sleepedf_df[sleep_feature_cols].values
y_sleep = sleep_labels_encoded

imputer_sleep = SimpleImputer(strategy='mean')
X_sleep = imputer_sleep.fit_transform(X_sleep)

scaler_sleep = StandardScaler()
X_sleep_scaled = scaler_sleep.fit_transform(X_sleep)

X_sleep_resampled, y_sleep_resampled = SMOTE(random_state=42, k_neighbors=1).fit_resample(X_sleep_scaled, y_sleep)

X_s_train, X_s_test, y_s_train, y_s_test = train_test_split(
    X_sleep_resampled, y_sleep_resampled, test_size=0.2, random_state=42, stratify=y_sleep_resampled
)

print("Sleep-EDF Training size:", X_s_train.shape, "Testing size:", X_s_test.shape)

# Simplified Sleep-EDF model (faster training, aiming ~80% accuracy)
sleep_model = Sequential([
    Dense(32, activation='relu', input_shape=(X_s_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(len(sleep_le.classes_), activation='softmax')
])

sleep_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
sleep_model.summary()

# Train model
sleep_model.fit(X_s_train, y_s_train, epochs=20, batch_size=32, validation_data=(X_s_test, y_s_test), verbose=1)

test_loss_s, test_acc_s = sleep_model.evaluate(X_s_test, y_s_test, verbose=0)
print(f"Sleep-EDF Test Accuracy: {test_acc_s*100:.2f}%")

y_s_pred = np.argmax(sleep_model.predict(X_s_test), axis=1)
print("\nSleep-EDF Classification Report:")
print(classification_report(y_s_test, y_s_pred, target_names=sleep_le.classes_))

# Add confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_s_test, y_s_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sleep_le.classes_, yticklabels=sleep_le.classes_)
plt.title("Confusion Matrix for Sleep-EDF Model")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

sleep_model_save_path = '/content/drive/MyDrive/WANProject/models/sleepedf_classification_model2.h5'
sleep_model.save(sleep_model_save_path)
print(f"Sleep-EDF model saved to {sleep_model_save_path}")

print("\nDone. Sleep-EDF model trained and saved with Confusion Matrix displayed.")
