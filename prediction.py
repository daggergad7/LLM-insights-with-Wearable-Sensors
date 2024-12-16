#sensor data prediction and getting music recommendation and insights

# -*- coding: utf-8 -*-
"""Use Pre-Trained Models for Predictions and LLM Insights"""

# Install required libraries
!pip install -q -U datasets
!pip install bitsandbytes
!pip install -q -U git+https://github.com/huggingface/transformers.git
!pip install -q -U git+https://github.com/huggingface/peft.git
!pip install -q -U git+https://github.com/huggingface/accelerate.git
!pip install -q -U einops
!pip install -q -U safetensors
!pip install -q -U torch
!pip install -q -U tensorflow
!pip install mne
!pip install scikit-learn
!pip install openai
!pip install imbalanced-learn
!pip install seaborn

import os
import numpy as np
import pandas as pd
import mne
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
from tensorflow.keras.models import load_model
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    set_seed,
    DataCollatorForLanguageModeling
)
from peft import PeftModel
import matplotlib.pyplot as plt
import seaborn as sns
import openai
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# Configure TensorFlow to not use the GPU
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
print("TensorFlow is configured to use CPU only.")

###########################################################
#             Load Pre-Trained Models (H5 Files)          #
###########################################################

wesad_model_path = '/content/drive/MyDrive/WANProject/models/wesad_emotion_classification_model.h5'
sleep_model_path = '/content/drive/MyDrive/WANProject/models/sleepedf_classification_model2.h5'

emotion_model = load_model(wesad_model_path)
print("WESAD emotion classification model loaded successfully.")

sleep_model = load_model(sleep_model_path)
print("Sleep-EDF sleep quality model loaded successfully.")

###########################################################
#               Define Classes and Feature Names           #
###########################################################

# WESAD model assumed to output 4 emotion classes: Amusement, Baseline, Meditation, Stress
# Sleep model assumed to output 3 sleep classes: Well Rested, Moderately Rested, Poorly Rested
emotion_classes = ['Amusement','Baseline','Meditation','Stress']
sleep_classes = ['Well Rested', 'Moderately Rested', 'Poorly Rested']

wesad_feature_names = [
    'EDA_Mean', 'EDA_Std', 'EDA_Max', 'EDA_Min', 'EDA_Skew', 'EDA_Kurtosis', 'EDA_Median', 'EDA_Variance', 'EDA_RMS',
    'ECG_Mean', 'ECG_Std', 'ECG_Max', 'ECG_Min', 'ECG_Skew', 'ECG_Kurtosis', 'ECG_Median', 'ECG_Variance', 'ECG_RMS',
    'EMG_Mean', 'EMG_Std', 'EMG_Max', 'EMG_Min', 'EMG_Skew', 'EMG_Kurtosis', 'EMG_Median'
]

sleep_feature_cols = [f'EEG_{stat}' for stat in ['Mean', 'Std', 'Max', 'Min', 'Skew', 'Kurtosis', 'Median', 'Variance', 'RMS']] + \
                     [f'EMG_{stat}' for stat in ['Mean', 'Std', 'Max', 'Min', 'Skew', 'Kurtosis', 'Median']]

num_samples = 100

# Fit WESAD Imputer/Scaler
synthetic_data = []
for _ in range(num_samples):
    features = {}
    for signal_name in ['EDA', 'ECG', 'EMG']:
        signal = np.random.normal(size=1000)
        if signal_name in ['EDA', 'ECG']:
            feature_list = ['Mean','Std','Max','Min','Skew','Kurtosis','Median','Variance','RMS']
        else:
            feature_list = ['Mean','Std','Max','Min','Skew','Kurtosis','Median']

        if 'Mean' in feature_list:
            features[f'{signal_name}_Mean'] = np.mean(signal)
        if 'Std' in feature_list:
            features[f'{signal_name}_Std'] = np.std(signal)
        if 'Max' in feature_list:
            features[f'{signal_name}_Max'] = np.max(signal)
        if 'Min' in feature_list:
            features[f'{signal_name}_Min'] = np.min(signal)
        if 'Skew' in feature_list:
            features[f'{signal_name}_Skew'] = skew(signal)
        if 'Kurtosis' in feature_list:
            features[f'{signal_name}_Kurtosis'] = kurtosis(signal)
        if 'Median' in feature_list:
            features[f'{signal_name}_Median'] = np.median(signal)
        if 'Variance' in feature_list and signal_name in ['EDA','ECG']:
            features[f'{signal_name}_Variance'] = np.var(signal)
        if 'RMS' in feature_list and signal_name in ['EDA','ECG']:
            features[f'{signal_name}_RMS'] = np.sqrt(np.mean(signal**2))
    synthetic_data.append(features)

df_synthetic = pd.DataFrame(synthetic_data)
df_synthetic = df_synthetic[wesad_feature_names]

imputer = SimpleImputer(strategy='mean')
imputer.fit(df_synthetic)
print("WESAD Imputer fitted successfully.")

scaler = StandardScaler()
scaler.fit(imputer.transform(df_synthetic))
print("WESAD Scaler fitted successfully.")

# Fit Sleep Imputer/Scaler
sleep_synthetic_data = []
for _ in range(num_samples):
    features = {}
    eeg_features = ['Mean','Std','Max','Min','Skew','Kurtosis','Median','Variance','RMS']
    emg_features = ['Mean','Std','Max','Min','Skew','Kurtosis','Median']
    eeg_signal = np.random.normal(size=1000)
    emg_signal = np.random.normal(size=1000)

    for feat in eeg_features:
        if feat == 'Mean':
            features['EEG_Mean'] = np.mean(eeg_signal)
        elif feat == 'Std':
            features['EEG_Std'] = np.std(eeg_signal)
        elif feat == 'Max':
            features['EEG_Max'] = np.max(eeg_signal)
        elif feat == 'Min':
            features['EEG_Min'] = np.min(eeg_signal)
        elif feat == 'Skew':
            features['EEG_Skew'] = skew(eeg_signal)
        elif feat == 'Kurtosis':
            features['EEG_Kurtosis'] = kurtosis(eeg_signal)
        elif feat == 'Median':
            features['EEG_Median'] = np.median(eeg_signal)
        elif feat == 'Variance':
            features['EEG_Variance'] = np.var(eeg_signal)
        elif feat == 'RMS':
            features['EEG_RMS'] = np.sqrt(np.mean(eeg_signal**2))

    for feat in emg_features:
        if feat == 'Mean':
            features['EMG_Mean'] = np.mean(emg_signal)
        elif feat == 'Std':
            features['EMG_Std'] = np.std(emg_signal)
        elif feat == 'Max':
            features['EMG_Max'] = np.max(emg_signal)
        elif feat == 'Min':
            features['EMG_Min'] = np.min(emg_signal)
        elif feat == 'Skew':
            features['EMG_Skew'] = skew(emg_signal)
        elif feat == 'Kurtosis':
            features['EMG_Kurtosis'] = kurtosis(emg_signal)
        elif feat == 'Median':
            features['EMG_Median'] = np.median(emg_signal)

    sleep_synthetic_data.append(features)

df_sleep_synth = pd.DataFrame(sleep_synthetic_data)
df_sleep_synth = df_sleep_synth[sleep_feature_cols]

sleep_imputer = SimpleImputer(strategy='mean')
sleep_imputer.fit(df_sleep_synth)
print("Sleep-EDF Imputer fitted successfully.")

sleep_scaler = StandardScaler()
sleep_scaler.fit(sleep_imputer.transform(df_sleep_synth))
print("Sleep-EDF Scaler fitted successfully.")

def _compute_wesad_features(new_data, imputer, scaler):
    features_per_signal = {
        'EDA': ['Mean','Std','Max','Min','Skew','Kurtosis','Median','Variance','RMS'],
        'ECG': ['Mean','Std','Max','Min','Skew','Kurtosis','Median','Variance','RMS'],
        'EMG': ['Mean','Std','Max','Min','Skew','Kurtosis','Median']
    }
    features = {}
    for signal_name, signal in new_data.items():
        for feat in features_per_signal[signal_name]:
            if feat == 'Mean':
                features[f'{signal_name}_Mean'] = np.mean(signal)
            elif feat == 'Std':
                features[f'{signal_name}_Std'] = np.std(signal)
            elif feat == 'Max':
                features[f'{signal_name}_Max'] = np.max(signal)
            elif feat == 'Min':
                features[f'{signal_name}_Min'] = np.min(signal)
            elif feat == 'Skew':
                features[f'{signal_name}_Skew'] = skew(signal)
            elif feat == 'Kurtosis':
                features[f'{signal_name}_Kurtosis'] = kurtosis(signal)
            elif feat == 'Median':
                features[f'{signal_name}_Median'] = np.median(signal)
            elif feat == 'Variance' and signal_name in ['EDA','ECG']:
                features[f'{signal_name}_Variance'] = np.var(signal)
            elif feat == 'RMS' and signal_name in ['EDA','ECG']:
                features[f'{signal_name}_RMS'] = np.sqrt(np.mean(signal**2))

    df_new = pd.DataFrame([features])
    df_imputed = pd.DataFrame(imputer.transform(df_new), columns=wesad_feature_names)
    X_scaled = scaler.transform(df_imputed)
    return X_scaled

def process_wesad_data(new_data, imputer, scaler):
    X_scaled_full = _compute_wesad_features(new_data, imputer, scaler)
    # The WESAD model expects 3 features (based on error): Let's pick EDA_Mean, ECG_Mean, EMG_Mean
    idx_eda = wesad_feature_names.index('EDA_Mean')
    idx_ecg = wesad_feature_names.index('ECG_Mean')
    idx_emg = wesad_feature_names.index('EMG_Mean')
    X_three = X_scaled_full[:, [idx_eda, idx_ecg, idx_emg]]
    return X_three

def _compute_sleep_features(new_data, sleep_imputer, sleep_scaler):
    eeg_features = ['Mean','Std','Max','Min','Skew','Kurtosis','Median','Variance','RMS']
    emg_features = ['Mean','Std','Max','Min','Skew','Kurtosis','Median']
    features = {}

    eeg_signal = new_data['EEG']
    emg_signal = new_data['EMG']

    for feat in eeg_features:
        if feat == 'Mean':
            features['EEG_Mean'] = np.mean(eeg_signal)
        elif feat == 'Std':
            features['EEG_Std'] = np.std(eeg_signal)
        elif feat == 'Max':
            features['EEG_Max'] = np.max(eeg_signal)
        elif feat == 'Min':
            features['EEG_Min'] = np.min(eeg_signal)
        elif feat == 'Skew':
            features['EEG_Skew'] = skew(eeg_signal)
        elif feat == 'Kurtosis':
            features['EEG_Kurtosis'] = kurtosis(eeg_signal)
        elif feat == 'Median':
            features['EEG_Median'] = np.median(eeg_signal)
        elif feat == 'Variance':
            features['EEG_Variance'] = np.var(eeg_signal)
        elif feat == 'RMS':
            features['EEG_RMS'] = np.sqrt(np.mean(eeg_signal**2))

    for feat in emg_features:
        if feat == 'Mean':
            features['EMG_Mean'] = np.mean(emg_signal)
        elif feat == 'Std':
            features['EMG_Std'] = np.std(emg_signal)
        elif feat == 'Max':
            features['EMG_Max'] = np.max(emg_signal)
        elif feat == 'Min':
            features['EMG_Min'] = np.min(emg_signal)
        elif feat == 'Skew':
            features['EMG_Skew'] = skew(emg_signal)
        elif feat == 'Kurtosis':
            features['EMG_Kurtosis'] = kurtosis(emg_signal)
        elif feat == 'Median':
            features['EMG_Median'] = np.median(emg_signal)

    df_new = pd.DataFrame([features])
    df_imputed = pd.DataFrame(sleep_imputer.transform(df_new), columns=sleep_feature_cols)
    X_scaled = sleep_scaler.transform(df_imputed)
    return X_scaled

def process_sleep_data(new_data, sleep_imputer, sleep_scaler):
    X_scaled_full = _compute_sleep_features(new_data, sleep_imputer, sleep_scaler)
    # If the sleep model expects just these features in order (EEG, EMG) and was trained on them, no slicing needed.
    return X_scaled_full

def get_wesad_prediction(X_scaled, model):
    # WESAD model outputs exactly 4 emotion classes in order: Amusement, Baseline, Meditation, Stress
    # shape of y_probs: (1,4)
    y_probs = model.predict(X_scaled)
    pred_class = np.argmax(y_probs, axis=1)[0]
    return ['Amusement','Baseline','Meditation','Stress'][pred_class]

def get_sleep_prediction(X_scaled, model):
    # Sleep model outputs 3 classes: Well Rested, Moderately Rested, Poorly Rested in order
    y_probs = model.predict(X_scaled)
    pred_class = np.argmax(y_probs, axis=1)[0]
    return ['Well Rested','Moderately Rested','Poorly Rested'][pred_class]

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model_id = "vilsonrodrigues/falcon-7b-instruct-sharded"
tokenizer = AutoTokenizer.from_pretrained('/content/drive/MyDrive/WANProject/falcon_music_recommendation', trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
print("Set pad_token to eos_token.")

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True
)
print("Base Falcon 7B model loaded with 4-bit quantization.")

model = PeftModel.from_pretrained(base_model, './falcon_music_recommendation')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print("LoRA adapters loaded and model moved to device.")

def generate_music_recommendations_llm(current_emotion, sleep_quality, model, tokenizer, device):
    prompt = (
        f"User's current emotion: {current_emotion}\n"
        f"User's sleep quality last night: {sleep_quality}\n"
        "Provide 3 music recommendations that match the user's mood and sleep quality.\n"
        "Additionally, rovide some insights to improve users mood.\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    max_new_tokens = 150
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    recommendations = generated_text[len(prompt):].strip()
    return recommendations

def main():
    # Generate random sensor data for emotion (EDA, ECG, EMG)
    wesad_data_example = {
        'EDA': np.random.normal(size=1000),
        'ECG': np.random.normal(size=1000),
        'EMG': np.random.normal(size=1000)
    }

    X_wesad_scaled = process_wesad_data(wesad_data_example, imputer, scaler)
    current_emotion = get_wesad_prediction(X_wesad_scaled, emotion_model)

    # Generate random sensor data for sleep (EEG, EMG)
    sleep_data_example = {
        'EEG': np.random.normal(size=1000),
        'EMG': np.random.normal(size=1000)
    }

    X_sleep_scaled = process_sleep_data(sleep_data_example, sleep_imputer, sleep_scaler)
    sleep_quality = get_sleep_prediction(X_sleep_scaled, sleep_model)

    print(f"From WESAD model (emotion): {current_emotion}")
    print(f"From Sleep-EDF model (sleep quality): {sleep_quality}")

    music_recommendations = generate_music_recommendations_llm(current_emotion, sleep_quality, model, tokenizer, device)

    print("\nFinal Insights:")
    print(f"You are feeling {current_emotion.lower()} and were {sleep_quality.lower()} last night.")
    print("Recommended Music:")
    print(music_recommendations)

if __name__ == "__main__":
    main()
