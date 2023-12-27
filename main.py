from flask import Flask, request, jsonify
import pickle
import numpy as np
import soundfile
import librosa

app = Flask(__name__)


@app.route('/')
def home():
    return "Hello world"


# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))


# DataFlair - Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
    return result


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the audio file from the request
        audio_file = request.files['audio']

        # Save the audio file temporarily
        temp_audio_path = 'temp.wav'
        audio_file.save(temp_audio_path)

        # Extract features from the audio file
        features = extract_feature(temp_audio_path, mfcc=True, chroma=True, mel=True)

        # Reshape the features to match the model input shape
        features = np.reshape(features, (1, -1))

        # Make a prediction
        prediction = model.predict(features)

        # Return the prediction as JSON k
        return jsonify({'emotion': prediction[0]})
    except Exception as e:
        return jsonify({'error1': str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)


#if __name__ == '__main__':
#    app.run(debug=True)



