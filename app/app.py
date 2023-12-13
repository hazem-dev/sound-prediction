from flask import Flask, render_template, request
import os
import librosa
import numpy as np
from joblib import load
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import pickle
import matplotlib.pyplot as plt
import base64

app = Flask(__name__)
# Load the SVM model
svm_model_path = '/app/models/audio_classifier_model_selected_features.joblib'
svm_model = load(svm_model_path)

# Load the VGG model
vgg_model_path = "/app/models/vgg_model.pkl"
with open(vgg_model_path, 'rb') as vgg_model_pkl:
    vgg_model = pickle.load(vgg_model_pkl)

# Define a function to extract audio features from an uploaded file
def extract_features_from_upload(file):
    print("Extracting features from the uploaded audio...")
    audio, sample_rate = librosa.load(file, res_type='kaiser_fast')
    chroma_stft_mean = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate))
    rms_mean = np.mean(librosa.feature.rms(y=audio))
    spectral_centroid_mean = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate))
    spectral_bandwidth_mean = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate))
    zero_crossing_rate_mean = np.mean(librosa.feature.zero_crossing_rate(y=audio))
    mfcc1_mean = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate)[0])
    mfcc2_mean = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate)[1])
    mfcc3_mean = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate)[2])
    mfcc4_mean = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate)[3])

    # Combine extracted features into a list
    features = [chroma_stft_mean, rms_mean, spectral_centroid_mean, spectral_bandwidth_mean, zero_crossing_rate_mean,
                mfcc1_mean, mfcc2_mean, mfcc3_mean, mfcc4_mean]

    print("Features extracted successfully.")
    return features

# Function to preprocess the audio for the VGG model
def preprocess_audio_for_vgg(audio_path):
    print(f"Processing audio for VGG: {audio_path}")

    # Load the audio file
    audio, sample_rate = librosa.load(audio_path, res_type='kaiser_fast')

    # Generate a spectrogram
    plt.figure(figsize=(4, 4))
    plt.specgram(audio, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap='viridis', sides='default', mode='default', scale='dB');
    plt.axis('off')  # Turn off axis labels
    plt.tight_layout()

    # Save the spectrogram as an image
    spect_path = "static/spectrogram.png"
    plt.savefig(spect_path)
    plt.close()

    # Convert the spectrogram image to base64
    with open(spect_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    # Preprocess the spectrogram image for prediction
    img = image.load_img(spect_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make predictions using VGG
    vgg_prediction = vgg_model.predict(img_array)

    return base64_image, vgg_prediction

if not os.path.exists('temp'):
    os.makedirs('temp')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['audio_file']
        if file:
            targets = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
            # Save the uploaded file to the 'temp' directory
            file_path = os.path.join('temp', file.filename)
            file.save(file_path)

            print(f"Uploaded file saved to: {file_path}")

            # Extract audio features for SVM
            features = extract_features_from_upload(file_path)

            # Make a prediction using the SVM model
            svm_prediction = svm_model.predict([features])[0]

            # Preprocess audio for VGG model
            base64_image, vgg_prediction = preprocess_audio_for_vgg(file_path)
            vgg_prediction = targets[list(vgg_prediction[0]).index(max(vgg_prediction[0]))]

            # Render the 'upload.html' template with classification results
            return render_template('upload.html', svm_result_html=f'SVM Classification Result: {targets[svm_prediction]}',
                                   vgg_result_html=f'VGG Classification Result: {vgg_prediction}',
                                   base64_image=base64_image)

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
