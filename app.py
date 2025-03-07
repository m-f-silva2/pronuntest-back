import logging
import tempfile
from core import convert_audio_to_spectrograms, PhonemeRecognitionService
from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import os
import soundfile as sf

app = Flask(__name__)
#CORS(app, resources={r"/*": {"Access-Control-Allow-Origins": "*"}})
#CORS(app, resources={r"/*": {"origins": "*"}})
CORS(app, supports_credentials=True, resources={r"/*": {"origins": "*"}}, headers="*")
#CORS(app, resources={r"/*": {"origins": "*"}})

@app.after_request
def add_cors_headers(response):
    response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

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
    logging.info("validar fonema {pattern}") 
    recording = request.files['recording']
    type_model = 'vocal' if pattern in ["a", "e", "i", "o", "u"] else 'p'

    # Crear archivo temporal sin que se elimine automáticamente
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_path = temp_file.name
        recording.save(temp_path)
    logging.info("validar fonema #1") 
    try:
        # Ahora sí podemos leerlo sin problemas
        spectrograms = convert_audio_to_spectrograms(temp_path)
        predictions = model.predict(spectrograms, type_model)
    finally:
        os.remove(temp_path)  # Eliminar el archivo después de usarlo
    logging.info("validar fonema #2") 
    phonemes = [p for p in predictions if p["class"] != "noise"]
    start_pattern = next((i for i, p in enumerate(phonemes) if p["class"] == pattern[0]), None)

    if start_pattern is None:
        return jsonify({"word": pattern, "score": 0, "phonemes": []})
    logging.info("validar fonema #3") 
    predicted = phonemes[start_pattern : start_pattern + len(pattern)]
    predicted += [{"class": "unknown", "percentage": 0.0}] * (len(pattern) - len(predicted))

    average = sum(p["percentage"] for p in predicted) / len(predicted) if predicted else 0

    return jsonify({"word": pattern, "score": average, "phonemes": predicted})




"""
@app.route("/api/word/<pattern>", methods=["POST"])
def validate_phoneme_pattern(pattern: str):   
    recording = request.files['recording']
    type_model = 'vocal' if pattern in ["a", "e", "i", "o", "u"] else 'p'

    # Guardar temporalmente el archivo para inspeccionarlo
    temp_file_path = "test_recording.wav"
    recording.save(temp_file_path)
    processed_audio, sample_rate = process_audio(temp_file_path)
    # Guardar el audio procesado en un nuevo archivo
    processed_file_path = "processed_recording.wav"
    sf.write(processed_file_path, processed_audio, sample_rate)
    print(f"Processed audio saved at: {processed_file_path}")
    try:
        # Procesar el archivo
        spectrograms = convert_audio_to_spectrograms(processed_file_path)
    finally:
        # Eliminar el archivo temporal
        if os.path.exists(processed_file_path):
            #os.remove(processed_file_path)
            print(f"Archivo temporal {processed_file_path} eliminado.")
    
    if(pattern in ["a", "e", "i", "o", "u"]):
        type_model = 'vocal'

    predictions = model.predict(spectrograms, type_model)
    phoneme = None
    percentage = 0.0
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



def validate_phoneme_pattern(pattern: str):   
    recording = request.files['recording']
    # Guardar temporalmente el archivo para inspeccionarlo
    temp_file_path = "test_recording.wav"
    recording.save(temp_file_path)
    processed_audio, sample_rate = process_audio(temp_file_path)

    # Guardar el audio procesado en un nuevo archivo
    processed_file_path = "processed_recording.wav"
    sf.write(processed_file_path, processed_audio, sample_rate)
    print(f"Processed audio saved at: {processed_file_path}")

    try:
        # Procesar el archivo
        spectrograms = convert_audio_to_spectrograms(processed_file_path)
    finally:
        # Eliminar el archivo temporal
        if os.path.exists(processed_file_path):
            #os.remove(processed_file_path)
            print(f"Archivo temporal {processed_file_path} eliminado.")
    
    if(pattern in ["a", "e", "i", "o", "u"]):
        type_model = 'vocal'

    predictions = model.predict(spectrograms, type_model)
    phoneme = None
    percentage = 0.0
    phonemes = []

    for prediction in predictions:
        if prediction["class"] == phoneme:
            if prediction["percentage"] > percentage:
                percentage = prediction["percentage"]
                phonemes[-1]["percentage"] = prediction["percentage"]
            continue

        phoneme = prediction["class"]
        percentage = prediction["percentage"]
        phonemes.append(prediction)

    
    start_pattern = None
    phonemes = list(filter(lambda pred: pred["class"] != "noise", phonemes))

    for i, phoneme in enumerate(phonemes):
        if phoneme["class"] == pattern[0]:
            start_pattern = i
            break
    print("phonemes",phonemes)
    print("estar",start_pattern," fonema:", pattern)
    print("phoneme:",phoneme)

    if start_pattern == None:
        result = {"word": pattern, "score": 0, "phonemes": []}
        return jsonify(result)

    default_phoneme = {"class": "unknown", "percentage": 0.0}
    predicted = phonemes[start_pattern : start_pattern + len(pattern)]
    predicted = predicted + [default_phoneme] * (len(pattern) - len(predicted))

    total_percentage = sum(phoneme["percentage"] for phoneme in predicted)
    average = total_percentage / len(predicted) if len(predicted) > 0 else 0
    result = {"word": pattern, "score": average, "phonemes": predicted}

    return jsonify(result)
"""


    
@app.route('/test/<pattern>', methods=["POST"])
def test(pattern: str):
    print("request--> ",request)
    res={"word": pattern+" 1", "score": 0, "phonemes": [], "val": request}
    if "recording" in request.files:
        file = request.files["recording"]
        res["val"] = file.filename+" : "+file.content_type
        print("res--> ", res)
        return jsonify({'res': file.filename})
    print("res--> ", res)
    return jsonify({'res': "nada"})


@app.route('/test/', methods=["OPTIONS"])
def optionsTest():
    print("requestOPTIONS--> ",request)
    return 'test', 200


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
    #app.run(debug=False, host="0.0.0.0")
    #print("Ejecuta con Gunicorn: gunicorn -w 2 -t 60 -b 0.0.0.0:5000 app:app")
    from waitress import serve  # Alternativa si no quieres usar Gunicorn
    serve(app, host="0.0.0.0", port=5000)