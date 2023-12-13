import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Load the data from the CSV file
data = pd.read_csv('static/features_3_sec.csv')  # Replace with the actual path to your CSV file

# Create a label encoder for the 'label' column
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])

# Select only the 9 features you've extracted from the audio
selected_features = data[
    ['chroma_stft_mean', 'rms_mean', 'spectral_centroid_mean', 'spectral_bandwidth_mean', 'zero_crossing_rate_mean',
     'mfcc1_mean', 'mfcc2_mean', 'mfcc3_mean', 'mfcc4_mean']]

# Separate features (X) and the label (y)
X = selected_features
y = data['label']

# Standardize the features (center to mean and scale to unit variance)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest classifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)
print("Model training completed.")

# Evaluate the model on the test data
accuracy = classifier.score(X_test, y_test)
print(f"Model Accuracy: {accuracy}")

# Save the trained model for future use

dump(classifier, '/app/models/audio_classifier_model_selected_features.joblib')
print("Model saved as '/app/models/audio_classifier_model_selected_features.joblib'.")


# Map the numerical labels back to their original names
label_names = label_encoder.inverse_transform(y_test)
print("Label names for the test data:", label_names)
