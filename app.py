import streamlit as st
import joblib
import numpy as np
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch

# Load models
svm_model = joblib.load("svm_model.joblib")
scaler = joblib.load("scaler.joblib")
lda_model = joblib.load("lda_model.joblib")

# Load Wav2Vec2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)

def extract_features(audio_path):
    audio, sr = torchaudio.load(audio_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        audio = resampler(audio)
    inputs = wav2vec_processor(audio.numpy(), sampling_rate=16000, return_tensors="pt")
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        wav2vec_features = wav2vec_model(**inputs).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return wav2vec_features

st.title("Audio Deepfake Detection")
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac"])

if uploaded_file:
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.write("Processing the file...")
    features = extract_features("temp_audio.wav")

    # Preprocess features
    features_scaled = scaler.transform([features])
    features_reduced = lda_model.transform(features_scaled)

    # Predict
    prediction = svm_model.predict(features_reduced)[0]
    result = "Fake Audio" if prediction == 1 else "Real Audio"

    st.success(f"Prediction: {result}")
