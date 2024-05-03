import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from keras.models import load_model
import numpy as np
import librosa

SAMPLE_RATE = 22050  # frequency with which instants of the audio signal
TARGET_SAMPLES = int(SAMPLE_RATE * 0.2)  # one second worth of audio
HOP_LENGTH = 128  # sliding window for FFT. Measured in number of samples
N_FFT = 255  # length of the windowed signal after padding with zeros

phonemes = ["a", "e", "i", "noise", "o", "u"]
pronuns = ["correct", "incorrect", "noise"]

def read_audio_segments(file):
    signal = read_audio(file)
    num_segments = (len(signal) + TARGET_SAMPLES - 1) // TARGET_SAMPLES

    segments = []

    for i in range(num_segments):
        start_sample = i * TARGET_SAMPLES
        end_sample = min((i + 1) * TARGET_SAMPLES, len(signal))
        segment = signal[start_sample:end_sample]
        segment = np.pad(segment, (0, TARGET_SAMPLES - len(segment)), "constant")
        segments.append(segment)

    return segments


def read_audio(file) -> np.ndarray:
    signal, _ = librosa.load(file, sr=SAMPLE_RATE, dtype=np.float32)
    return signal


def get_spectrogram(signal: np.ndarray) -> np.ndarray:
    spectrogram = librosa.stft(signal, n_fft=N_FFT, hop_length=HOP_LENGTH)
    spectrogram = np.abs(spectrogram.T)
    return np.expand_dims(spectrogram, axis=2)


def convert_audio_to_spectrograms(file) -> np.ndarray:
    segments = read_audio_segments(file)
    recording = [get_spectrogram(signal) for signal in segments]
    return np.array(recording)


def get_pred_percentage(logits: np.ndarray) -> np.float32:
    logits_exp = np.exp(logits - np.max(logits))
    probabilities = logits_exp / np.sum(logits_exp, axis=-1, keepdims=True)
    max_probability = np.max(probabilities, axis=-1)
    return round(max_probability * 100, 1)


class PhonemeRecognitionService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = load_model("./models/phoneme_model.h5")
        return cls._instance

    def predict(self, spectrograms: np.ndarray):
        predicts = self._model.predict(spectrograms, verbose=0)
        recording = []

        for logits in predicts:
            percentage = get_pred_percentage(logits)
            predicted_class = phonemes[np.argmax(logits)]
            recording.append(
                {
                    "class": predicted_class,
                    "percentage": percentage,
                }
            )

        return recording