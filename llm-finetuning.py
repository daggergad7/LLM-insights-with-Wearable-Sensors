#llm finetuning using spotify dataset

!pip install -q -U datasets
# -----------------------------
# 1. Install Required Libraries
# -----------------------------
# Uncomment and run the following lines if running in a new environment (e.g., Google Colab)

!pip install bitsandbytes
!pip install -q -U git+https://github.com/huggingface/transformers.git
!pip install -q -U git+https://github.com/huggingface/peft.git
!pip install -q -U git+https://github.com/huggingface/accelerate.git
!pip install -q -U einops
!pip install -q -U safetensors
!pip install -q -U torch
!pip install -q -U tensorflow


# -----------------------------
# 2. Import Necessary Libraries
# -----------------------------
# Remove the line that forces TensorFlow to use CPU by hiding the GPUs
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # This line is removed

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings
from tensorflow.keras.models import load_model
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    set_seed,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import os

warnings.filterwarnings('ignore')

# Configure TensorFlow to not use the GPU
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')
print("TensorFlow is configured to use CPU only.")

# -----------------------------
# 3. Load the Emotion Classification Model
# -----------------------------

# Define the path to your trained emotion classification model
model_save_path = '/content/drive/MyDrive/WANProject/models/emotion_classification_model.h5'  # Update this path

# Load the emotion classification model
emotion_model = load_model(model_save_path)
print("Emotion classification model loaded successfully.")

# -----------------------------
# 4. Define Label Encoder Classes
# -----------------------------

# Initialize LabelEncoder and define classes
le = LabelEncoder()
le.classes_ = np.array(['Amusement', 'Baseline', 'Meditation', 'Moderately Rested', 'Poorly Rested', 'Stress', 'Well Rested'])
print("Label encoder defined successfully. Classes:", le.classes_)

# -----------------------------
# 5. Define Feature Names Used During Training
# -----------------------------

feature_names = [
    'EDA_Mean', 'EDA_Std', 'EDA_Max', 'EDA_Min', 'EDA_Skew', 'EDA_Kurtosis', 'EDA_Median', 'EDA_Variance', 'EDA_RMS',
    'ECG_Mean', 'ECG_Std', 'ECG_Max', 'ECG_Min', 'ECG_Skew', 'ECG_Kurtosis', 'ECG_Median', 'ECG_Variance', 'ECG_RMS',
    'EMG_Mean', 'EMG_Std', 'EMG_Max', 'EMG_Min', 'EMG_Skew', 'EMG_Kurtosis', 'EMG_Median'
]
print("Feature names defined successfully.")

# -----------------------------
# 6. Fit Imputer and Scaler on Synthetic Data
# -----------------------------

# Generate synthetic data to fit the imputer and scaler
num_samples = 100
synthetic_data = []
for _ in range(num_samples):
    features = {}
    for signal_name in ['EDA', 'ECG', 'EMG']:
        signal = np.random.normal(size=1000)
        if signal_name in ['EDA', 'ECG']:
            feature_list = ['Mean', 'Std', 'Max', 'Min', 'Skew', 'Kurtosis', 'Median', 'Variance', 'RMS']
        else:
            feature_list = ['Mean', 'Std', 'Max', 'Min', 'Skew', 'Kurtosis', 'Median']

        # Extract features
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
        if 'Variance' in feature_list and signal_name in ['EDA', 'ECG']:
            features[f'{signal_name}_Variance'] = np.var(signal)
        if 'RMS' in feature_list and signal_name in ['EDA', 'ECG']:
            features[f'{signal_name}_RMS'] = np.sqrt(np.mean(signal**2))

    synthetic_data.append(features)

# Create DataFrame
df_synthetic = pd.DataFrame(synthetic_data)
df_synthetic = df_synthetic[feature_names]  # Ensure correct order

# Fit imputer
imputer = SimpleImputer(strategy='mean')
imputer.fit(df_synthetic)
print("Imputer fitted successfully.")

# Fit scaler
scaler = StandardScaler()
scaler.fit(imputer.transform(df_synthetic))
print("Scaler fitted successfully.")


import os
# -----------------------------
# 7. Load and Preprocess the Spotify Dataset
# -----------------------------

# Define the path to the Spotify dataset
spotify_dataset_path = '/content/drive/MyDrive/WANProject/spotifydataset/dataset.csv'  # Update this path

# Load the dataset
spotify_df = pd.read_csv(spotify_dataset_path)
print("Spotify dataset loaded successfully.")

# Select relevant columns
spotify_df = spotify_df[['track_name', 'artists', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
                         'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']]

# Remove duplicates and NaN values
spotify_df.drop_duplicates(inplace=True)
spotify_df.dropna(inplace=True)

# Reset index
spotify_df.reset_index(drop=True, inplace=True)

# Scale numerical features between 0 and 1
numerical_features = ['danceability', 'energy', 'loudness', 'speechiness',
                      'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
scaler_features = MinMaxScaler()
spotify_df[numerical_features] = scaler_features.fit_transform(spotify_df[numerical_features])

print("Spotify dataset preprocessed successfully.")

# -----------------------------
# 8. Define Functions for Mapping Emotions to Song Features
# -----------------------------

def map_emotion_to_features(emotion):
    """
    Maps user emotion to desired song features.

    Parameters:
    - emotion (str): User's current emotion.

    Returns:
    - feature_preferences (dict): Desired ranges for song features.
    """
    if emotion == 'Amusement':
        feature_preferences = {
            'valence': (0.6, 1.0),  # Positive mood
            'energy': (0.6, 1.0),   # High energy
            'danceability': (0.5, 1.0)
        }
    elif emotion == 'Stress':
        feature_preferences = {
            'valence': (0.0, 0.4),  # Negative mood
            'energy': (0.0, 0.5),   # Low energy
            'acousticness': (0.5, 1.0),
            'instrumentalness': (0.0, 0.5)
        }
    elif emotion == 'Meditation':
        feature_preferences = {
            'valence': (0.4, 0.7),
            'energy': (0.0, 0.4),
            'acousticness': (0.5, 1.0),
            'instrumentalness': (0.5, 1.0)
        }
    elif emotion == 'Baseline':
        feature_preferences = {
            'valence': (0.4, 0.6),
            'energy': (0.4, 0.6)
        }
    else:
        feature_preferences = {
            'valence': (0.3, 0.7),
            'energy': (0.3, 0.7)
        }
    return feature_preferences

def map_sleep_quality_to_features(sleep_quality):
    """
    Maps user's sleep quality to desired song features.

    Parameters:
    - sleep_quality (str): User's sleep quality last night.

    Returns:
    - feature_preferences (dict): Desired ranges for song features.
    """
    if sleep_quality == 'Well Rested':
        feature_preferences = {
            'tempo': (0.5, 1.0)
        }
    elif sleep_quality == 'Moderately Rested':
        feature_preferences = {
            'tempo': (0.3, 0.7)
        }
    elif sleep_quality == 'Poorly Rested':
        feature_preferences = {
            'tempo': (0.0, 0.5)
        }
    else:
        feature_preferences = {}
    return feature_preferences

def get_recommendations_from_dataset(emotion, sleep_quality, spotify_df, num_recommendations=3):
    """
    Retrieves song recommendations from the Spotify dataset based on emotion and sleep quality.

    Parameters:
    - emotion (str): User's current emotion.
    - sleep_quality (str): User's sleep quality last night.
    - spotify_df (pd.DataFrame): Spotify dataset.
    - num_recommendations (int): Number of recommendations to return.

    Returns:
    - recommendations (list): List of recommended songs with title and artist.
    """
    emotion_features = map_emotion_to_features(emotion)
    sleep_features = map_sleep_quality_to_features(sleep_quality)

    # Combine feature preferences
    combined_preferences = {**emotion_features, **sleep_features}

    # Filter songs based on preferences
    df_filtered = spotify_df.copy()
    for feature, (min_val, max_val) in combined_preferences.items():
        df_filtered = df_filtered[(df_filtered[feature] >= min_val) & (df_filtered[feature] <= max_val)]

    # If not enough songs, relax the filters
    if len(df_filtered) < num_recommendations:
        df_filtered = spotify_df.sample(n=num_recommendations)
    else:
        df_filtered = df_filtered.sample(n=num_recommendations)

    # Prepare recommendations
    recommendations = []
    for idx, row in df_filtered.iterrows():
        song_info = {
            'title': row['track_name'],
            'artist': row['artists']
        }
        recommendations.append(song_info)

    return recommendations

# -----------------------------
# 9. Define Functions for Processing and Prediction
# -----------------------------

def process_new_sensor_data(new_data, imputer, scaler):
    """
    Processes new sensor data into the feature format expected by the model.

    Parameters:
    - new_data (dict): Dictionary containing 'EDA', 'ECG', 'EMG' signals as numpy arrays.
    - imputer: Pre-fitted imputer.
    - scaler: Pre-fitted scaler.

    Returns:
    - X_new_scaled (numpy.ndarray): Scaled feature vector ready for prediction.
    """
    features_per_signal = {
        'EDA': ['Mean', 'Std', 'Max', 'Min', 'Skew', 'Kurtosis', 'Median', 'Variance', 'RMS'],
        'ECG': ['Mean', 'Std', 'Max', 'Min', 'Skew', 'Kurtosis', 'Median', 'Variance', 'RMS'],
        'EMG': ['Mean', 'Std', 'Max', 'Min', 'Skew', 'Kurtosis', 'Median'],
    }
    features = {}
    for signal_name in new_data.keys():
        signal = new_data[signal_name]
        feature_list = features_per_signal.get(signal_name, [])
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
        if 'Variance' in feature_list and signal_name in ['EDA', 'ECG']:
            features[f'{signal_name}_Variance'] = np.var(signal)
        if 'RMS' in feature_list and signal_name in ['EDA', 'ECG']:
            features[f'{signal_name}_RMS'] = np.sqrt(np.mean(signal**2))

    df_new = pd.DataFrame([features])
    df_new = df_new[feature_names]  # Ensure correct order

    # Impute missing values
    df_imputed = pd.DataFrame(imputer.transform(df_new), columns=feature_names)

    # Scale features
    X_new_scaled = scaler.transform(df_imputed)

    return X_new_scaled

def predict_emotion_and_sleep_quality(X_new_scaled, model, label_encoder):
    """
    Predicts the current emotion and sleep quality based on scaled features.

    Parameters:
    - X_new_scaled (numpy.ndarray): Scaled feature vector.
    - model (keras.Model): Trained emotion classification model.
    - label_encoder (sklearn.preprocessing.LabelEncoder): Encoder for emotion labels.

    Returns:
    - current_emotion (str): Predicted current emotion label.
    - sleep_quality (str): Predicted sleep quality label.
    """
    y_pred_probs = model.predict(X_new_scaled)
    emotion_labels = ['Amusement', 'Baseline', 'Meditation', 'Stress']
    sleep_labels = ['Well Rested', 'Moderately Rested', 'Poorly Rested']

    # Get indices for emotion and sleep quality
    emotion_indices = [np.where(label_encoder.classes_ == label)[0][0] for label in emotion_labels]
    sleep_indices = [np.where(label_encoder.classes_ == label)[0][0] for label in sleep_labels]

    # Predict emotion
    emotion_probs = y_pred_probs[0][emotion_indices]
    current_emotion_index = emotion_indices[np.argmax(emotion_probs)]
    current_emotion = label_encoder.classes_[current_emotion_index]

    # Predict sleep quality
    sleep_probs = y_pred_probs[0][sleep_indices]
    sleep_quality_index = sleep_indices[np.argmax(sleep_probs)]
    sleep_quality = label_encoder.classes_[sleep_quality_index]

    return current_emotion, sleep_quality

# -----------------------------
# 10. Generate Dataset for LLM Fine-Tuning
# -----------------------------

def prepare_llm_dataset(spotify_df):
    """
    Prepares a dataset for LLM fine-tuning based on the Spotify dataset.

    Parameters:
    - spotify_df (pd.DataFrame): Preprocessed Spotify dataset.

    Returns:
    - df_llm_dataset (pd.DataFrame): DataFrame containing prompts and completions.
    """
    emotions = ['Amusement', 'Stress', 'Meditation', 'Baseline']
    sleep_qualities = ['Well Rested', 'Moderately Rested', 'Poorly Rested']
    dataset_entries = []

    for emotion in emotions:
        for sleep_quality in sleep_qualities:
            for _ in range(50):  # Increase the number of examples per combination
                prompt = f"User's current emotion: {emotion}\nUser's sleep quality last night: {sleep_quality}\nProvide 3 music recommendations that match the user's mood and sleep quality."
                recommendations_list = get_recommendations_from_dataset(emotion, sleep_quality, spotify_df)
                completion = ""
                for idx, song in enumerate(recommendations_list, 1):
                    completion += f"{idx}. \"{song['title']}\" by {song['artist']}\n"
                completion = completion.strip()
                dataset_entries.append({'text': prompt + '\n' + completion})

    df_llm_dataset = pd.DataFrame(dataset_entries)
    print(f"LLM dataset prepared successfully with {len(df_llm_dataset)} entries.")
    return df_llm_dataset

# Prepare the dataset
df_llm_dataset = prepare_llm_dataset(spotify_df)

# -----------------------------
# 11. Fine-Tune the Falcon 7B Model with 4-bit Quantization and PEFT (LoRA)
# -----------------------------

# Define quantization configuration for 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Update the model ID
model_id = "vilsonrodrigues/falcon-7b-instruct-sharded"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Set pad_token to eos_token to avoid increasing vocab_size
tokenizer.pad_token = tokenizer.eos_token
print("Set pad_token to eos_token.")

# Load the model with 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True
)
print("Falcon 7B model loaded with 4-bit quantization.")

# No need to resize embeddings since vocab_size remains the same

# Apply PEFT (LoRA) for parameter-efficient fine-tuning
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query_key_value"],  # Adjusted for this model
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
print("LoRA applied to the model for fine-tuning.")

# Set use_cache to False
model.config.use_cache = False

# Prepare the dataset for training
dataset = Dataset.from_pandas(df_llm_dataset)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=512,
        padding='max_length',
    )

# Apply the tokenization
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # We're using a causal LM
)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./falcon_music_recommendation',
    overwrite_output_dir=True,
    num_train_epochs=3,  # Adjust as needed
    per_device_train_batch_size=1,  # Adjust based on GPU memory
    per_device_eval_batch_size=1,
    save_strategy='no',
    logging_steps=10,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=True,    # Enable FP16 if supported
    bf16=False,   # Enable BF16 if your GPU supports it
    gradient_checkpointing=False,  # Disable gradient checkpointing
    optim="adamw_torch",
    report_to=[],  # Disable reporting to third-party services
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Set seed for reproducibility
set_seed(42)

# Fine-tune the model


# Save the LoRA adapters and tokenizer
model.save_pretrained('./falcon_music_recommendation')
tokenizer.save_pretrained('./falcon_music_recommendation')
print("LoRA adapters and tokenizer saved successfully.")

# -----------------------------
# 12. Load the Fine-Tuned Model for Inference
# -----------------------------
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('/content/drive/MyDrive/WANProject/falcon_music_recommendation', trust_remote_code=True)

# Set pad_token to eos_token
tokenizer.pad_token = tokenizer.eos_token
print("Set pad_token to eos_token.")

# Load the base model with 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True
)

print("Base Falcon 7B model loaded with 4-bit quantization.")

# Load the LoRA adapters
model = PeftModel.from_pretrained(model, './falcon_music_recommendation')

print("LoRA adapters loaded into the model.")

# Move model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print("Model moved to device.")

# -----------------------------
# 13. Define the Recommendation Generation Function
# -----------------------------

def generate_music_recommendations_llm(current_emotion, sleep_quality, model, tokenizer, device):
    """
    Generates music recommendations using the fine-tuned LLM based on current emotion and sleep quality.

    Parameters:
    - current_emotion (str): User's current emotion.
    - sleep_quality (str): User's sleep quality last night.
    - model: Fine-tuned LLM.
    - tokenizer: Tokenizer corresponding to the model.
    - device: Torch device ('cuda' or 'cpu').

    Returns:
    - recommendations (str): Generated music recommendations.
    """
    prompt = (
        f"User's current emotion: {current_emotion}\n"
        f"User's sleep quality last night: {sleep_quality}\n"
        "Provide 3 music recommendations that match the user's mood and sleep quality.\n"
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
    # Extract the recommendations
    recommendations = generated_text[len(prompt):].strip()
    return recommendations

# -----------------------------
# 14. Define the Main Function to Analyze and Recommend Music
# -----------------------------

def analyze_emotion_and_recommend_music(
    new_sensor_data, emotion_model, model, tokenizer, device, label_encoder, imputer, scaler
):
    """
    Analyzes emotion from sensor data and provides music recommendations using the fine-tuned LLM.

    Parameters:
    - new_sensor_data (dict): Dictionary containing 'EDA', 'ECG', 'EMG' signals as numpy arrays.
    - emotion_model (keras.Model): Trained emotion classification model.
    - model: Fine-tuned LLM.
    - tokenizer (transformers.PreTrainedTokenizer): Tokenizer corresponding to the model.
    - device (torch.device): Device to run the model on.
    - label_encoder (sklearn.preprocessing.LabelEncoder): Encoder for emotion labels.
    - imputer (sklearn.preprocessing.SimpleImputer): Pre-fitted imputer.
    - scaler (sklearn.preprocessing.StandardScaler): Pre-fitted scaler.

    Returns:
    - message (str): Custom message including detected emotions.
    - music_recommendations (str): Generated music recommendations.
    """
    # Step 1: Process the new sensor data
    X_new_scaled = process_new_sensor_data(new_sensor_data, imputer, scaler)

    # Step 2: Predict emotions
    current_emotion, sleep_quality = predict_emotion_and_sleep_quality(X_new_scaled, emotion_model, label_encoder)
    print(f"Detected Emotion: {current_emotion}, Sleep Quality: {sleep_quality}")

    # Step 3: Generate personalized message
    message = f"You are feeling {current_emotion.lower()} and were {sleep_quality.lower()} last night. Here are some music recommendations:"

    # Step 4: Generate music recommendations using the fine-tuned LLM
    music_recommendations = generate_music_recommendations_llm(current_emotion, sleep_quality, model, tokenizer, device)

    return message, music_recommendations

# -----------------------------
# 15. Run an Example with Synthetic Sensor Data
# -----------------------------

def main():
    # Example new sensor data (replace with actual sensor readings)
    new_sensor_data_example = {
        'EDA': np.random.normal(size=1000),  # Simulated EDA signal
        'ECG': np.random.normal(size=1000),  # Simulated ECG signal
        'EMG': np.random.normal(size=1000)   # Simulated EMG signal
    }

    # Analyze emotion and get music recommendations
    custom_message, music_recs = analyze_emotion_and_recommend_music(
        new_sensor_data_example,
        emotion_model,
        model,
        tokenizer,
        device,
        le,
        imputer,
        scaler
    )

    print(f"\n{custom_message}\n")
    print("Music Recommendations:")
    print(music_recs)

if __name__ == "__main__":
    main()
