import tempfile
from core import convert_audio_to_spectrograms, PhonemeRecognitionService
from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import os
import soundfile as sf

app = Flask(__name__)
CORS(app, resources={r"/*": {"Access-Control-Allow-Origins": "*"}})
model = PhonemeRecognitionService()
type_model = 'vocal'

@app.route("/api/", methods=["POST"])
def most_frequent_phoneme():
    # guarda el audio
    recording = request.files["recording"]
    # obtener espectogramas
    spectrograms = convert_audio_to_spectrograms(recording)
    # llamar model, hacer predicciones
    preds = model.predict(spectrograms)

    # filtrar la clase ruido
    phonemes = list(filter(lambda pred: pred["class"] != "noise", preds))
    preds = phonemes if len(phonemes) > 0 else preds
    phoneme = max(preds, key=lambda x: x["percentage"])
    print("fonema:",phoneme, flush=True)
    return jsonify(
        {
            "pronunciation": "correct",
            "phoneme": phoneme,
        }
    )
def process_audio(file_path):
    # Cargar el audio a su frecuencia original (48,000 Hz)
    audio, sample_rate = librosa.load(file_path, sr=None)

    # Resamplear a 16,000 Hz
    if sample_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    return audio, sample_rate

@app.route("/api/word/<pattern>", methods=["POST"])
def validate_phoneme_pattern(pattern: str):   
    recording = request.files['recording']
    type_model = 'vocal' if pattern in ["a", "e", "i", "o", "u"] else 'p'

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_path = temp_file.name
        recording.save(temp_path)  # Guardar el archivo antes de cerrarlo

    try:
        spectrograms = convert_audio_to_spectrograms(temp_path)
        predictions = model.predict(spectrograms, type_model)
    finally:
        os.remove(temp_path)  # Eliminar el archivo después de usarlo

    phonemes = []
    for prediction in predictions:
        if phonemes and prediction["class"] == phonemes[-1]["class"]:
            phonemes[-1]["percentage"] = max(phonemes[-1]["percentage"], prediction["percentage"])
        else:
            phonemes.append(prediction)

    phonemes = [p for p in phonemes if p["class"] != "noise"]
    start_pattern = next((i for i, p in enumerate(phonemes) if p["class"] == pattern[0]), None)

    if start_pattern is None:
        return jsonify({"word": pattern, "score": 0, "phonemes": []})

    default_phoneme = {"class": "unknown", "percentage": 0.0}
    predicted = phonemes[start_pattern : start_pattern + len(pattern)]
    predicted += [default_phoneme] * (len(pattern) - len(predicted))

    total_percentage = sum(p["percentage"] for p in predicted)
    average = total_percentage / len(predicted) if predicted else 0

    return jsonify({"word": pattern, "score": average, "phonemes": predicted})


    

@app.route('/')
def home():
    return "Running app"


if __name__ == "__main__":
    # pyinstaller --onefile --console --clean server.py --name "backend"
    spectrograms = convert_audio_to_spectrograms("./models/recording.wav")
    model.predict(spectrograms, type_model)
    

    # version app
    #app.run(port=4000, debug=False)

    # version runanyway
    #app.run(port=5000, debug=False, host="0.0.0.0")

    # deploy production
    app.run(debug=False, host="0.0.0.0")
    #print("Ejecuta con Gunicorn: gunicorn -w 2 -t 60 -b 0.0.0.0:5000 app:app")
