import os
import logging
import gc
import numpy as np
import tensorflow as tf
import librosa
from keras.models import load_model

# Deshabilitar OneDNN y reducir logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Configurar logging
logging.basicConfig(level=logging.INFO)

# Parámetros de audio
SAMPLE_RATE = 22050
DURATION = 0.2
TARGET_SAMPLES = int(SAMPLE_RATE * DURATION)
HOP_LENGTH = 128 #128
N_FFT = 255 #255

# Clases de predicción
PHONEMES = [
    "a", "e", "i", "noise", "o", "u",
    "pa", "pe", "pi", "po", "pu",
    "papa", "pelo", "pie", "Palo", "pila",
    "pollo", "lupa", "pulpo", "mapa", "pino", "pan"
]
PRONUNS = ["correct", "incorrect", "noise"]

class PhonemeRecognitionService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_models()
        return cls._instance

    def _load_models(self):
        """Carga los modelos solo una vez en memoria."""
        try:
            logging.info("Cargando modelos en memoria...")
            self.models = {
                "vocal": load_model("./models/phoneme_vocal_model.h5"),
                "p": load_model("./models/phoneme_vocal_model.h5")
            }
        except Exception as e:
            logging.error(f"Error al cargar los modelos: {e}")
            self.models = {}

    def predict(self, spectrograms: np.ndarray, type_model: str):
        """Realiza la predicción usando un modelo pre-cargado."""
        model = self.models.get(type_model)
        if model is None:
            return {"error": f"Modelo '{type_model}' no disponible"}

        try:
            logging.info(f"Realizando predicción con el modelo {type_model}...")
            predicts = model.predict(spectrograms, verbose=0)

            # Obtener clases y probabilidades en una sola pasada
            results = [
                {"class": PHONEMES[idx], "percentage": round(np.max(probs) * 1, 1)}
                for idx, probs in zip(np.argmax(predicts, axis=-1), predicts)
            ]

            return results

        except Exception as e:
            logging.error(f"Error en la predicción: {e}")
            return {"error": "Error en la predicción"}

        finally:
            # Liberar memoria
            del spectrograms
            gc.collect()

# Funciones auxiliares
def read_audio_segments(file):
    signal = read_audio(file)
    return [
        np.pad(signal[i * TARGET_SAMPLES : (i + 1) * TARGET_SAMPLES], 
               (0, max(0, TARGET_SAMPLES - len(signal[i * TARGET_SAMPLES : (i + 1) * TARGET_SAMPLES]))), 
               "constant")
        for i in range((len(signal) + TARGET_SAMPLES - 1) // TARGET_SAMPLES)
    ]

def read_audio(file) -> np.ndarray:
    return librosa.load(file, sr=SAMPLE_RATE, dtype=np.float32)[0]

def get_spectrogram(signal: np.ndarray) -> np.ndarray:
    return np.expand_dims(np.abs(librosa.stft(signal, n_fft=N_FFT, hop_length=HOP_LENGTH)).T, axis=2)

def convert_audio_to_spectrograms(file) -> np.ndarray:
    return np.array([get_spectrogram(segment) for segment in read_audio_segments(file)])


