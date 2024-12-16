##!curl -o wesad.zip "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download"
##!unzip wesad.zip
##!rm wesad.zip

# -*- coding: utf-8 -*-
"""Separate Training for WESAD (Faster and ~80% Accuracy) with Confusion Matrix and Additional Evaluations"""

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
#                 Load and Process WESAD Data             #
###########################################################

wesad_data_path = '/content/WESAD'

# Reduced subjects for faster training
subjects = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']

wesad_dfs = []
for subj in subjects:
    subj_file = os.path.join(wesad_data_path, subj, f'{subj}.pkl')
    if not os.path.exists(subj_file):
        print(f"File not found: {subj_file}")
        continue
    with open(subj_file, 'rb') as file:
        data = pickle.load(file, encoding='latin1')
        chest_data = data['signal']['chest']
        labels = data['label']

        eda_signal = chest_data['EDA']
        ecg_signal = chest_data['ECG']
        emg_signal = chest_data['EMG']

        df = pd.DataFrame({
            'EDA': eda_signal.flatten(),
            'ECG': ecg_signal.flatten(),
            'EMG': emg_signal.flatten(),
            'Label': labels.flatten()
        })
        wesad_dfs.append(df)

print(f"Loaded WESAD data for {len(wesad_dfs)} subjects.")

wesad_df = pd.concat(wesad_dfs, axis=0).reset_index(drop=True)
print("WESAD DataFrame shape:", wesad_df.shape)
print("WESAD label distribution:", Counter(wesad_df['Label']))

label_mapping_wesad = {0:'Baseline', 1:'Stress', 2:'Amusement', 3:'Meditation'}

valid_labels = set(label_mapping_wesad.keys())
wesad_df = wesad_df[wesad_df['Label'].isin(valid_labels)].copy()
wesad_df['Emotion'] = wesad_df['Label'].map(label_mapping_wesad)

if wesad_df['Emotion'].isnull().any():
    print("Warning: Some rows have invalid labels not in {0,1,2,3}.")
else:
    print("WESAD unique emotions:", wesad_df['Emotion'].unique())

###########################################################
#             Prepare WESAD Data for Modeling             #
###########################################################

wesad_features = wesad_df[['EDA','ECG','EMG']]
wesad_labels = wesad_df['Emotion'].values

wesad_le = LabelEncoder()
wesad_labels_encoded = wesad_le.fit_transform(wesad_labels)
print("WESAD Emotion classes:", wesad_le.classes_)

X_wesad = wesad_features.values
y_wesad = wesad_labels_encoded

imputer_wesad = SimpleImputer(strategy='mean')
X_wesad = imputer_wesad.fit_transform(X_wesad)

scaler_wesad = StandardScaler()
X_wesad_scaled = scaler_wesad.fit_transform(X_wesad)

smote = SMOTE(random_state=42, k_neighbors=1)
X_wesad_resampled, y_wesad_resampled = smote.fit_resample(X_wesad_scaled, y_wesad)

X_w_train, X_w_test, y_w_train, y_w_test = train_test_split(
    X_wesad_resampled, y_wesad_resampled, test_size=0.2, random_state=42, stratify=y_wesad_resampled
)

print("WESAD Training size:", X_w_train.shape, "Testing size:", X_w_test.shape)

# Simplified WESAD model: smaller architecture, no dropout
wesad_model = Sequential([
    Dense(32, activation='relu', input_shape=(X_w_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(len(wesad_le.classes_), activation='softmax')
])

wesad_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
wesad_model.summary()

# Train for a few more epochs for better accuracy, larger batch_size for speed
wesad_model.fit(X_w_train, y_w_train, epochs=10, batch_size=256, validation_data=(X_w_test, y_w_test), verbose=1)

test_loss_w, test_acc_w = wesad_model.evaluate(X_w_test, y_w_test, verbose=0)
print(f"WESAD Test Accuracy: {test_acc_w*100:.2f}%")

y_w_pred = np.argmax(wesad_model.predict(X_w_test), axis=1)

# Print classification report
print("\nWESAD Classification Report:")
print(classification_report(y_w_test, y_w_pred, target_names=wesad_le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_w_test, y_w_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wesad_le.classes_, yticklabels=wesad_le.classes_)
plt.title("Confusion Matrix for WESAD Model")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

wesad_model_save_path = '/content/drive/MyDrive/WANProject/models/wesad_emotion_classification_model.h5'
wesad_model.save(wesad_model_save_path)
print(f"WESAD model saved to {wesad_model_save_path}")
