import warnings
import serial
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import pandas as pd
from scipy.signal import find_peaks
import serial
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import pickle
import os

warnings.filterwarnings("ignore", category=UserWarning)
# === Load models ===
with open(r'C:\Python_Env\ECE4179\TRC3500\Normal Mode\clf_material_set2.pkl', 'rb') as f:
    clf_material = pickle.load(f)
with open(r'C:\Python_Env\ECE4179\TRC3500\Normal Mode\clf_distance_set2.pkl', 'rb') as f:
    clf_distance = pickle.load(f)
with open(r'C:\Python_Env\ECE4179\TRC3500\Normal Mode\clf_height_set2.pkl', 'rb') as f:
    clf_height = pickle.load(f)
with open(r'C:\Python_Env\ECE4179\TRC3500\Normal Mode\scaler_set2.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open(r'C:\Python_Env\ECE4179\TRC3500\Normal Mode\material_encoder_set2.pkl', 'rb') as f:
    material_encoder = pickle.load(f)

# === Settings ===
SERIAL_PORT = 'COM4'
BAUDRATE = 115200
BASELINE = 1987
VREF = 3
ADC_RESOLUTION = 4096
SAMPLING_RATE = 1000
THRESHOLD = 100
QUIET_COUNT = 100
MAX_SAMPLES = 2000

def extract_features(samples):
    centered = (np.array(samples) - BASELINE) * (VREF / ADC_RESOLUTION)
    N = len(centered)

    peak_voltage = np.max(np.abs(centered))
    signal_energy = np.sum(centered ** 2)
    duration_ms = (N / SAMPLING_RATE) * 1000
    zcr = ((centered[:-1] * centered[1:]) < 0).sum() / duration_ms * 1000 if duration_ms > 0 else 0

    T = 1.0 / SAMPLING_RATE
    yf = fft(centered)
    xf = fftfreq(N, T)[:N//2]
    fft_mag = 2.0/N * np.abs(yf[0:N//2])

    dominant_freq = xf[np.argmax(fft_mag)] if np.any(fft_mag) else 0
    spectral_centroid = np.sum(xf * fft_mag) / np.sum(fft_mag) if np.sum(fft_mag) > 0 else 0
    spectral_energy = np.sum(fft_mag**2)

    return {
        "Peak Voltage (V)": peak_voltage,
        "Signal Energy (J)": signal_energy,
        "Duration (ms)": duration_ms,
        "ZCR": zcr,
        "Dominant Frequency (Hz)": dominant_freq,
        "Spectral Centroid (Hz)": spectral_centroid,
        "Spectral Energy": spectral_energy
    }

# === Serial Listen ===
ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
print("Listening for vibration...")

while True:
    try:
        line = ser.readline().decode('utf-8').strip()
        if not line:
            continue
        try:
            adc_val = int(line)
        except ValueError:
            continue

        if abs(adc_val - BASELINE) > THRESHOLD:
            print("Vibration detected!")
            samples = [adc_val]
            silent_count = 0

            while True:
                line = ser.readline().decode('utf-8').strip()
                if not line:
                    continue
                try:
                    val = int(line)
                    samples.append(val)
                    if abs(val - BASELINE) < 20:
                        silent_count += 1
                    else:
                        silent_count = 0
                    if silent_count > QUIET_COUNT or len(samples) > MAX_SAMPLES:
                        break
                except:
                    continue

            features_dict = extract_features(samples)
            features_array = np.array(list(features_dict.values())).reshape(1, -1)
            features_scaled = scaler.transform(features_array)

            material_pred = clf_material.predict(features_scaled)[0]
            distance_pred = clf_distance.predict(features_scaled)[0]
            height_pred = clf_height.predict(features_scaled)[0]

            material_name = material_encoder.inverse_transform([material_pred])[0]

            print(f"\nPrediction: {material_name}, {distance_pred}cm, {height_pred}cm")
            print("==================================")

            # Ask user if prediction is correct
            correct = input("Is the prediction correct? (y/n): ").lower()

            if correct != 'y':
                # Manual labeling
                true_material = input("Enter true material (e.g., Coin, Eraser): ").strip()
                true_distance = input("Enter true distance (cm): ").strip()
                true_height = input("Enter true height (cm): ").strip()

                # Save wrong prediction for future retraining
                wrong_entry = {
                    "Material": true_material,
                    "Distance (cm)": true_distance,
                    "Height (cm)": true_height,
                    **features_dict
                }

                # Append to file
                try:
                    df_wrong = pd.read_csv(r'C:\Python_Env\ECE4179\TRC3500\wrong_predictions_set1.csv')
                    df_wrong = pd.concat([df_wrong, pd.DataFrame([wrong_entry])], ignore_index=True)
                except FileNotFoundError:
                    df_wrong = pd.DataFrame([wrong_entry])

                df_wrong.to_csv('wrong_predictions_set1.csv', index=False)
                print("Wrong prediction logged for retraining!\n")

    except KeyboardInterrupt:
        print("Stopped.")
        ser.close()
        break
